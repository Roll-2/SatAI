"""Build the 9-class crop dataset (Task 5).

Pipeline:
  1. **Crops** — for every telescope folder in ``Data/fits_raw/<class>/``
     (classes 0-7), run the loader + FITS star finder + crop generator
     and stage the crops in ``Data/crops/staging/<class>/``.
  2. **Altro** — class 8 is generated synthetically by
     :func:`Src.acquisition.altro_generator.generate_altro_crops`.
  3. **Split** — stratified 70% train / 15% val / 15% test per class
     (spec 5.3), materialized into ``Data/crops/{train,val,test}/<class>/``.

Usage (from the project root)::

    python -m Src.data.build_dataset --n-per-class 5000
"""

from __future__ import annotations
import numpy as np

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from Src.preprocessing.anti_saturator import desaturate
from Src.preprocessing.crop_generator import generate_crops_for_image
from Src.preprocessing.loader import load_fits
from Src.preprocessing.star_finder import find_stars_fits
from Src.utils.constants import (
    CLASS_FOLDERS,
    CROPS_STAGING_DIR,
    FITS_RAW_DIR,
    RAND_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

__all__ = ["stage_crops_for_class", "build_dataset"]

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / val / test (spec 5.3)
def stage_crops_for_class(class_id: int,
                          n_per_class: int,
                          max_crops_per_image: int = 20) -> int:
    """Crop every FITS of one telescope class into the staging area.

    Args:
        class_id: class index (0-7 for real telescopes).
        n_per_class: target number of crops (crop generation stops early).
        max_crops_per_image: crops per source image (spec: max 20).

    Returns:
        Number of crops staged for this class.
    """
    import random
    from Src.acquisition.altro_generator import _hubble_psf, _jwst_psf

    folder = CLASS_FOLDERS[class_id]
    fits_dir = FITS_RAW_DIR / folder
    out_dir = CROPS_STAGING_DIR / folder

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for real FITS files
    has_fits = fits_dir.exists() and any(fits_dir.glob("*.fits"))

    # STAGE real FITS crops if available
    if has_fits:
        existing = len(list(out_dir.glob(".npy"))) if out_dir.exists() else 0
        if existing >= n_per_class:
            print(f"[{folder}] staging already has {existing} crops — skipped.")
            return existing

        fits_files = sorted(fits_dir.rglob("*.fits"))
        total = existing
        for idx, fpath in enumerate(tqdm(fits_files, desc=f"Crops/{folder}", unit="img")):
            if total >= n_per_class:
                break
            try:
                # Task 3.2: anti-saturation is mandatory for FITS with bright
                # stars, before star finding and cropping.
                img = desaturate(load_fits(fpath))
                positions = find_stars_fits(img, telescope=folder)
                # Indexed prefix: files with the same stem in different
                # subfolders must not overwrite each other's crops.
                saved = generate_crops_for_image(
                    img, positions, out_dir, prefix=f"{idx:05d}_{fpath.stem}",
                    max_crops=max_crops_per_image)
                total += len(saved)
            except Exception as exc:  # noqa: BLE001 — skip corrupt sources
                print(f"[skip] {fpath.name}: {exc}")
    else:
        total = 0
        print(f"[{folder}] no FITS files found — will generate synthetic crops only.")

    # NEW: Add synthetic Hubble-like PSF crops to the Hubble class (class 0)
    # These contain the 4-vane spider spike pattern that distinguishes HST from others.
    # Run regardless of whether real FITS were available, to ensure the model
    # always sees the distinctive spike pattern during training.
    if folder == "hubble":
        rng = random.Random(RAND_SEED)
        hubble_crops_dir = CROPS_STAGING_DIR / folder / "synthetic_spikes"
        hubble_crops_dir.mkdir(parents=True, exist_ok=True)
        # Count existing synthetic crops
        existing_hubble = len(list(hubble_crops_dir.glob(".npy")))
        n_synthetic = max(0, n_per_class - existing_hubble - total)
        for i in range(n_synthetic):
            crop = _hubble_psf(rng)
            np.save(hubble_crops_dir / f"synth_spike_{i:04d}.npy", crop)
        total += n_synthetic

    # NEW: Add synthetic JWST-like PSF crops to the JWST class (class 1)
    # These contain the 6-hexagonal spike pattern that distinguishes JWST from others.
    # Run regardless of whether real FITS were available, to ensure the model
    # always sees the distinctive spike pattern during training.
    if folder == "jwst":
        rng = random.Random(RAND_SEED + 1)
        jwst_crops_dir = CROPS_STAGING_DIR / folder / "synthetic_spikes"
        jwst_crops_dir.mkdir(parents=True, exist_ok=True)
        # Count existing synthetic crops
        existing_jwst = len(list(jwst_crops_dir.glob(".npy")))
        n_synthetic = max(0, n_per_class - existing_jwst - total)
        for i in range(n_synthetic):
            crop = _jwst_psf(rng)
            np.save(jwst_crops_dir / f"synth_spike_{i:04d}.npy", crop)
        total += n_synthetic

    print(f"[{folder}] staged {total} crops.")
    return total

def _stratified_split(class_id: int, n_per_class: int) -> dict[str, int]:
    """Split one class's staged crops into train/val/test directories."""
    folder = CLASS_FOLDERS[class_id]
    src_dir = CROPS_STAGING_DIR / folder
    # Search recursively to find crops in subdirectories like synthetic_spikes/
    files = sorted(src_dir.glob("**/*.npy")) if src_dir.exists() else []
    if not files:
        print(f"[{folder}] nothing staged — split skipped.")
        return {"train": 0, "val": 0, "test": 0}

    rng = random.Random(RAND_SEED + class_id)
    rng.shuffle(files)
    if len(files) > n_per_class:
        files = files[:n_per_class]           # keep classes balanced

    n_train = int(len(files) * SPLIT_FRACTIONS[0])
    n_val = int(len(files) * SPLIT_FRACTIONS[1])
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    counts = {}
    for split, split_files in splits.items():
        dst = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split] \
            / folder
        dst.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, dst / f.name)
        counts[split] = len(split_files)
    print(f"[{folder}] split: {counts}")
    return counts


def build_dataset(n_per_class: int = 5000) -> dict[str, dict[str, int]]:
    """Run the full Task 5 pipeline: stage crops, build Altro, split.

    Args:
        n_per_class: target crops per class (spec: 5000 x 9 = 45000).

    Returns:
        Per-class split counts.
    """
    # 1-2. Stage crops for the real telescopes + generate the Altro class.
    for class_id in range(8):
        stage_crops_for_class(class_id, n_per_class)

    altro_dir = CROPS_STAGING_DIR / CLASS_FOLDERS[8]
    have_altro = len(list(altro_dir.glob("*.npy"))) if altro_dir.exists() else 0
    if have_altro < n_per_class:
        print(f"\n=== Generating class-8 'Altro' ({n_per_class} crops) ===")
        from Src.acquisition.altro_generator import generate_altro_crops
        counts = generate_altro_crops(n_per_class, out_dir=altro_dir)
        print(f"[Altro] generated: {counts}")

    # 3. Stratified split.
    print("\n=== Stratified split 70/15/15 ===")
    results = {CLASS_FOLDERS[c]: _stratified_split(c, n_per_class)
               for c in range(9)}

    totals = {s: sum(r[s] for r in results.values())
              for s in ("train", "val", "test")}
    print(f"\nTotals: {totals}")
    return results


def main() -> None:
    """CLI entry point for dataset construction."""
    parser = argparse.ArgumentParser(description="Build the SatAi dataset")
    parser.add_argument("--n-per-class", type=int, default=5000,
                        help="target crops per class (spec: 5000)")
    args = parser.parse_args()
    build_dataset(n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()
