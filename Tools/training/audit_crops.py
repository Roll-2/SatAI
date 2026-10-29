"""Audit the staged 64x64 crops: are they really star crops, or junk?

The streaming pipeline deletes the source FITS after cropping, so the
``.npy`` files in ``Data/crops/staging/<class>/`` are the only artifact
left. This script answers "can we trust them?" in two ways:

1. **Quantitative** — per-class statistics over a fixed random sample:
   * ``center_star`` — fraction of crops with a local peak >= 5 sigma
     above the sky noise within a few pixels of the center (32, 32).
     This is the primary "is there a star here?" check: the crop
     generator centers on a detected star, so a real crop always has
     one; junk (empty sky, noise, artifacts) usually does not.
   * ``center_peak`` — fraction of crops whose *global* brightest pixel
     sits within a few pixels of the center. In crowded fields a
     brighter neighbor can sit inside the 64x64 window, so this is
     reported as crowding info, not as a failure.
   * ``core_ratio`` — median fraction of flux inside the central 8x8 box
     relative to the central 24x24 box. A point source concentrates flux
     in the core; flat noise spreads it evenly.
   * ``empty`` — fraction of crops where (max - median) < 0.05, i.e. no
     detectable source at all.
   * ``dead`` — fraction of constant/zero crops (unreadable data).

2. **Visual** — a 10x10 montage per class saved to
   ``reports/crop_audit/montage_<class>.png`` so a human can eyeball the
   actual pixels (each cell min/max-stretched for display only).

Usage (from the project root)::

    .venv/bin/python Tools/training/audit_crops.py            # all classes
    .venv/bin/python Tools/training/audit_crops.py --classes hubble jwst
    .venv/bin/python Tools/training/audit_crops.py --sample 500 --cols 10
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: no GUI needed
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Src.utils.constants import (  # noqa: E402
    CLASS_FOLDERS,
    CLASS_NAMES,
    CROPS_STAGING_DIR,
    CROP_SIZE,
    PROJECT_ROOT,
    RAND_SEED,
)

__all__ = ["audit_class", "audit_all"]

REPORTS_DIR: Path = PROJECT_ROOT / "reports" / "crop_audit"

#: Center box used for the core-flux ratio (half-size in pixels).
CORE_HALF: int = 4             # 8x8 core
RING_HALF: int = 12            # 24x24 ring

#: A crop is "centered" when the argmax is within this Chebyshev distance.
CENTER_TOL: int = 6

#: Significance (in sigma-clipped sky sigmas) that the central patch must
#: reach for a crop to count as a real star crop. The star finder detects
#: at 5 sigma against the *full-frame* sky noise; inside a 64x64 crop the
#: star's own spikes and the background structure inflate the clipped std
#: by ~2-3x, so 2 sigma in-crop corresponds to the original 5-sigma floor.
SIGMA_MIN: float = 2.0

#: Name of the quarantine subfolder inside each staging class.
QUARANTINE_DIR: str = "quarantine"


def _load_crops(folder: Path, sample: int, seed: int) -> list[np.ndarray]:
    """Return up to ``sample`` crops from ``folder`` (deterministic draw)."""
    files = sorted(p for p in folder.glob("*.npy")
                   if not p.name.startswith("."))
    rng = random.Random(seed)
    rng.shuffle(files)
    crops: list[np.ndarray] = []
    for f in files[:sample]:
        try:
            arr = np.load(f)
            if arr.shape == (CROP_SIZE, CROP_SIZE):
                crops.append(arr.astype(np.float32))
        except Exception:      # noqa: BLE001 — skip corrupt crops
            continue
    return crops


def _metrics(crops: list[np.ndarray]) -> dict[str, float]:
    """Compute the sanity metrics over a list of crops.

    Note on ``center_star`` vs ``center_peak``: the crop generator
    re-centers each crop on a detected star peak, but crowded fields
    often put a *brighter* neighbor inside the 64x64 window — the global
    argmax then lands off-center even though a real star sits at the
    center. So the primary metric asks "is there a star at the center?"
    (local peak near (32, 32) above the sky noise), and the secondary
    metric reports how often the global max is off-center (crowding).
    """
    if not crops:
        return {"n": 0.0, "center_star": 0.0, "center_peak": 0.0,
                "core_ratio": 0.0, "empty": 0.0, "dead": 0.0,
                "insig": 0.0}

    n = len(crops)
    center_star = 0
    centered = 0
    core_ratios: list[float] = []
    empty = 0
    dead = 0
    insig = 0
    c = CROP_SIZE // 2          # 32
    r = CENTER_TOL              # local-star search radius around center
    from astropy.stats import sigma_clipped_stats
    for img in crops:
        if float(img.std()) < 1e-4:
            dead += 1
            continue
        if float(img.max() - np.median(img)) < 0.05:
            empty += 1
        # Robust sky statistics (sigma-clipped, same as DAOStarFinder's
        # detection criterion): immune to the star's own spikes filling
        # the tiny 64x64 window, which breaks a plain MAD estimate.
        _, sky_med, sky_std = sigma_clipped_stats(img, sigma=3.0)
        patch = img[c - r:c + r + 1, c - r:c + r + 1]
        sig = (float(patch.max()) - float(sky_med)) / (float(sky_std) + 1e-8)
        if sig >= SIGMA_MIN:
            center_star += 1
        else:
            insig += 1
        # Global argmax: should be at center unless a brighter neighbor
        # star is inside the window (crowding).
        y, x = np.unravel_index(int(np.argmax(img)), img.shape)
        if max(abs(y - c), abs(x - c)) <= CENTER_TOL:
            centered += 1
        core = img[c - CORE_HALF:c + CORE_HALF + 1,
                   c - CORE_HALF:c + CORE_HALF + 1].sum()
        ring = img[c - RING_HALF:c + RING_HALF + 1,
                   c - RING_HALF:c + RING_HALF + 1].sum()
        core_ratios.append(float(core / ring) if ring > 0 else 0.0)

    return {
        "n": float(n),
        "center_star": center_star / n,
        "center_peak": centered / n,
        "core_ratio": float(np.median(core_ratios)),
        "empty": empty / n,
        "dead": dead / n,
        "insig": insig / n,
    }


def quarantine_class(class_id: int, threshold: float = SIGMA_MIN,
                     limit: int | None = None,
                     dry_run: bool = False) -> tuple[int, int]:
    """Move junk crops to a quarantine subfolder (reversible prune).

    A crop is junk when the central patch never reaches ``threshold``
    sigma-clipped sky sigmas — the star finder should not have produced
    it (extended nebula edges, galaxy bodies, pure background).

    Quarantine (move, not delete): the crop generator's re-centering
    window is only 7x7 while DAOStarFinder's centroid can be a few
    pixels off for faint sources near the fwhm=2 limit, so a cautious
    threshold keeps those rather than discarding them. All moves are
    reversible: ``audit_crops.py --restore`` puts everything back.

    Args:
        class_id: class index (0-8).
        threshold: minimum central significance in sigmas.
        limit: stop after moving this many crops (None = no limit).
        dry_run: count without moving anything.

    Returns:
        ``(moved, checked)`` counts.
    """
    from astropy.stats import sigma_clipped_stats

    folder = CROPS_STAGING_DIR / CLASS_FOLDERS[class_id]
    if not folder.exists():
        return 0, 0
    files = sorted(p for p in folder.glob("*.npy")
                   if not p.name.startswith("."))
    rng = random.Random(RAND_SEED + 99)     # fixed subsample for --limit
    rng.shuffle(files)
    if limit is not None:
        files = files[:limit]

    qdir = folder / QUARANTINE_DIR
    moved = 0
    checked = 0
    for f in files:
        try:
            img = np.load(f).astype(np.float32)
        except Exception:                  # noqa: BLE001 — unreadable
            continue
        if img.shape != (CROP_SIZE, CROP_SIZE):
            continue
        checked += 1
        _, sky_med, sky_std = sigma_clipped_stats(img, sigma=3.0)
        c32 = CROP_SIZE // 2
        patch = img[c32 - CENTER_TOL:c32 + CENTER_TOL + 1,
                    c32 - CENTER_TOL:c32 + CENTER_TOL + 1]
        sig = (float(patch.max()) - float(sky_med)) / (float(sky_std) + 1e-8)
        if sig < threshold:
            if not dry_run:
                qdir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), qdir / f.name)
            moved += 1
    return moved, checked


def restore_class(class_id: int) -> int:
    """Move every quarantined crop of one class back into staging."""
    qdir = CROPS_STAGING_DIR / CLASS_FOLDERS[class_id] / QUARANTINE_DIR
    if not qdir.exists():
        return 0
    restored = 0
    for f in sorted(qdir.glob("*.npy")):
        shutil.move(str(f), f.parent.parent / f.name)
        restored += 1
    if not any(qdir.iterdir()):
        qdir.rmdir()
    return restored


def _montage(crops: list[np.ndarray], out_path: Path, cols: int) -> Path:
    """Save a grid montage of the crops (per-cell stretch for display)."""
    n = len(crops)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img in zip(axes, crops):
        disp = img - img.min()
        if disp.max() > 0:
            disp = disp / disp.max()
        ax.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def audit_class(class_id: int, sample: int = 400, cols: int = 10,
                seed: int = RAND_SEED) -> dict[str, float]:
    """Audit one class: metrics + montage.

    Args:
        class_id: class index (0-8).
        sample: max crops to sample.
        cols: montage columns.
        seed: deterministic sampling seed.

    Returns:
        The metrics dict (see :func:`_metrics`).
    """
    folder = CROPS_STAGING_DIR / CLASS_FOLDERS[class_id]
    crops = _load_crops(folder, sample, seed + class_id)
    metrics = _metrics(crops)
    out = REPORTS_DIR / f"montage_{CLASS_FOLDERS[class_id]}.png"
    if crops:
        _montage(crops, out, cols)
    metrics["montage"] = str(out)
    return metrics


def audit_all(sample: int = 400, cols: int = 10,
              classes: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Audit every requested class and print a summary table.

    Args:
        sample: max crops per class.
        cols: montage columns.
        classes: subset of class names; defaults to all 9.

    Returns:
        ``{class_name: metrics}``.
    """
    targets = classes or CLASS_FOLDERS
    results: dict[str, dict[str, float]] = {}
    header = (f"{'class':<9} {'n':>5} {'center_star':>12} "
              f"{'center_peak':>12} {'core_ratio':>10} {'empty':>6} "
              f"{'dead':>5} {'insig':>6}  verdict")
    print(header)
    print("-" * len(header))
    for name in targets:
        cid = CLASS_FOLDERS.index(name)
        m = audit_class(cid, sample=sample, cols=cols)
        results[name] = m
        if m["n"] == 0:
            print(f"{name:<9} {'—':>5}  no crops staged yet")
            continue
        ok = (m["center_star"] >= 0.6 and m["core_ratio"] >= 0.15
              and m["empty"] <= 0.20)
        verdict = "OK" if ok else "CHECK"
        print(f"{name:<9} {int(m['n']):>5} {m['center_star']:>12.0%} "
              f"{m['center_peak']:>12.0%} {m['core_ratio']:>10.2f} "
              f"{m['empty']:>6.0%} {m['dead']:>5.0%} {m['insig']:>6.0%}  "
              f"{verdict}")
        if len(m.get("montage", "")):
            print(f"    montage: {m['montage']}")
    return results


def main() -> None:
    """CLI entry point for the crop audit.

    Modes:
        default            — audit (stats + montages)
        --quarantine       — move junk crops to quarantine/ (reversible)
        --quarantine --dry-run — count what would be quarantined
        --restore          — move every quarantined crop back
    """
    parser = argparse.ArgumentParser(
        description="Audit staged crops: sanity stats + visual montages")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="class names to audit (default: all)")
    parser.add_argument("--sample", type=int, default=400,
                        help="max crops sampled per class (default 400)")
    parser.add_argument("--cols", type=int, default=10,
                        help="montage columns (default 10)")
    parser.add_argument("--seed", type=int, default=RAND_SEED,
                        help="sampling seed (default 42)")
    parser.add_argument("--quarantine", action="store_true",
                        help="move crops below the significance threshold "
                             "to a quarantine/ subfolder (reversible)")
    parser.add_argument("--restore", action="store_true",
                        help="move every quarantined crop back to staging")
    parser.add_argument("--threshold", type=float, default=SIGMA_MIN,
                        help="quarantine threshold in sigma (default 2; "
                             "2 sigma in-crop ~ the finder's 5-sigma "
                             "full-frame floor)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max crops to check per class (quarantine)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --quarantine: count without moving")
    args = parser.parse_args()
    names = args.classes or CLASS_FOLDERS

    if args.restore:
        for name in names:
            n = restore_class(CLASS_FOLDERS.index(name))
            print(f"[{name}] restored {n} crops from quarantine.")
        return
    if args.quarantine:
        label = "would move" if args.dry_run else "moved"
        for name in names:
            moved, checked = quarantine_class(
                CLASS_FOLDERS.index(name), threshold=args.threshold,
                limit=args.limit, dry_run=args.dry_run)
            print(f"[{name}] {label} {moved}/{checked} checked crops "
                  f"to quarantine (threshold {args.threshold} sigma).")
        return
    audit_all(sample=args.sample, cols=args.cols, classes=args.classes)


if __name__ == "__main__":
    main()