"""Minimal backup of everything needed to retrain from zero (Satai).

Strategy for "adding a satellite in the future" (Roman, or anything else):

* The model is ~0.42 M parameters (~1.7 MB) and only the LAST linear layer
  depends on the class count — the conv backbone is class-agnostic. So the
  cheapest correct path to N+1 classes is always **retraining from an empty
  init**, not surgically editing a trained checkpoint.
* Retraining needs (1) the *empty* model definition frozen as a checkpoint
  and (2) a *seed sample of the real crops* (the staging area is deletable:
  it can be re-downloaded, but that costs hours of archive transfers).

This script therefore backs up exactly those two things:

* ``empty_model_v1.pt`` — untrained AstroNet-v2 weights (bit-exact restart).
* ``crop_sample/<class>/`` — 100 random crops per class (~16 KB each),
  enough to bootstrap/augment a retrain while the full campaign re-runs.

Everything lands in ``Data/model_backup/`` (gitignored, like all data).
Total size: ~17 MB — it can be copied anywhere (USB, cloud, e-mail).

Usage (from the project root)::

    .venv/bin/python Tools/training/model_backup.py save
    .venv/bin/python Tools/training/model_backup.py verify
    .venv/bin/python Tools/training/model_backup.py restore   # sample -> staging
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Src.utils.constants import (  # noqa: E402
    CLASS_FOLDERS,
    CLASS_NAMES,
    CROPS_STAGING_DIR,
    DATA_DIR,
    NUM_CLASSES,
    PROJECT_ROOT,
    RAND_SEED,
)

__all__ = ["save_backup", "verify_backup", "restore_backup"]

BACKUP_DIR: Path = DATA_DIR / "model_backup"
SAMPLE_DIRNAME: str = "crop_sample"
MANIFEST: str = "backup_manifest.json"

#: Crops per class kept in the backup sample.
CROPS_PER_CLASS: int = 100


def _empty_checkpoint() -> dict:
    """Build the untrained-model checkpoint (training-checkpoint format)."""
    import torch

    from Src.models.psf_cnn import build_model
    model = build_model(device="cpu", num_classes=NUM_CLASSES)
    return {
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "epoch": 0,
        "note": "empty init — bit-exact restart point for retrains",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_backup(crops_per_class: int = CROPS_PER_CLASS) -> Path:
    """Write the empty model + per-class crop sample to the backup folder.

    Args:
        crops_per_class: how many staging crops to sample per class.

    Returns:
        The backup directory.
    """
    import torch

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Empty model (overwrites: it is deterministic by construction).
    ckpt = _empty_checkpoint()
    model_path = BACKUP_DIR / "empty_model_v1.pt"
    torch.save(ckpt, model_path)
    n_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
    print(f"[model] empty init saved: {model_path.name} "
          f"({model_path.stat().st_size / 1e6:.1f} MB, {n_params / 1e6:.2f} M "
          f"params, {NUM_CLASSES} classes)")

    # 2. Crop sample per class (post-quarantine crops only).
    rng = random.Random(RAND_SEED + 123)
    sample_root = BACKUP_DIR / SAMPLE_DIRNAME
    counts: dict[str, int] = {}
    for name in CLASS_FOLDERS:
        staging = CROPS_STAGING_DIR / name
        dest = sample_root / name
        dest.mkdir(parents=True, exist_ok=True)
        files = [p for p in staging.glob("*.npy")
                 if not p.name.startswith(".")]
        rng.shuffle(files)
        kept = 0
        for f in files[:crops_per_class]:
            try:
                arr = np.load(f)
            except Exception:               # noqa: BLE001
                continue
            if arr.shape != (64, 64):
                continue
            np.save(dest / f.name, arr)
            kept += 1
        counts[name] = kept
        try:
            shown = dest.relative_to(PROJECT_ROOT)
        except ValueError:              # custom --backup-dir / tests
            shown = dest
        print(f"[{name}] {kept} crops -> {shown}")

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
        "crop_counts": counts,
        "model": model_path.name,
        "note": "real staging crops are deletable; this sample lets a "
                "retrain bootstrap while the full campaign re-runs",
    }
    (BACKUP_DIR / MANIFEST).write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in BACKUP_DIR.rglob("*") if p.is_file())
    print(f"\nBackup complete: {BACKUP_DIR} "
          f"({total / 1e6:.1f} MB total) — copy this folder anywhere safe.")
    return BACKUP_DIR


def verify_backup() -> bool:
    """Check the backup is complete and the model forward-passes.

    Returns:
        True when every expected piece is present and functional.
    """
    import torch

    from Src.models.psf_cnn import build_model

    ok = True
    model_path = BACKUP_DIR / "empty_model_v1.pt"
    if not model_path.exists():
        print("[model] MISSING — run 'save' first.")
        ok = False
    else:
        ckpt = torch.load(model_path, map_location="cpu",
                          weights_only=False)
        model = build_model(device="cpu",
                            num_classes=int(ckpt.get("num_classes",
                                                     NUM_CLASSES)))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 64, 64))
        print(f"[model] OK — {list(out.shape)} logits from empty init "
              f"({ckpt.get('created', '?')})")

    manifest_path = BACKUP_DIR / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name in CLASS_FOLDERS:
            n = len(list((BACKUP_DIR / SAMPLE_DIRNAME / name).glob("*.npy"))) \
                if (BACKUP_DIR / SAMPLE_DIRNAME / name).exists() else 0
            expected = manifest["crop_counts"].get(name, 0)
            state = "OK" if n == expected and n > 0 else \
                ("empty (no data yet)" if n == 0 and expected == 0 else "MISMATCH")
            if state == "MISMATCH":
                ok = False
            print(f"[{name}] {n}/{expected} sample crops — {state}")
    else:
        print("[manifest] MISSING.")
        ok = False
    print("\nBackup verify:", "PASSED" if ok else "FAILED")
    return ok


def restore_backup() -> int:
    """Copy the backup crop sample back into the staging area.

    Restored files are prefixed ``bk_`` so they never collide with the
    campaign's own crop names; the staging md5 ledger is unaffected
    (it tracks source FITS, not crops).

    Returns:
        Number of crops restored.
    """
    restored = 0
    for name in CLASS_FOLDERS:
        src = BACKUP_DIR / SAMPLE_DIRNAME / name
        if not src.exists():
            continue
        dst = CROPS_STAGING_DIR / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.npy")):
            target = dst / f"bk_{f.name}"
            if not target.exists():
                shutil.copy2(f, target)
                restored += 1
    print(f"Restored {restored} backup crops into staging "
          f"(prefixed 'bk_').")
    return restored


def main() -> None:
    """CLI entry point for the minimal retrain backup."""
    parser = argparse.ArgumentParser(
        description="Minimal backup for future retrains: empty model + "
                    "crop sample (~17 MB)")
    parser.add_argument("command", choices=["save", "verify", "restore"],
                        help="save the backup / verify it / restore the "
                             "crop sample into staging")
    parser.add_argument("--crops-per-class", type=int,
                        default=CROPS_PER_CLASS,
                        help="sample size per class (default 100)")
    args = parser.parse_args()
    if args.command == "save":
        save_backup(crops_per_class=args.crops_per_class)
    elif args.command == "verify":
        raise SystemExit(0 if verify_backup() else 1)
    else:
        restore_backup()


if __name__ == "__main__":
    main()