"""Build the 9-class crop dataset (Task 5).

Pipeline:
  1. **Crops** — for every telescope folder in ``data/fits_raw/<class>/``
     (classes 0-7), run the loader + FITS star finder + crop generator
     and stage the crops in ``data/crops/staging/<class>/``.
  2. **Altro** — class 8 is generated synthetically by
     :func:`src.acquisition.altro_generator.generate_altro_crops`.
  3. **Split** — stratified 70% train / 15% val / 15% test per class
     (spec 5.3), materialized into ``data/crops/{train,val,test}/<class>/``.

Usage (from the project root)::

    python -m src.data.build_dataset --n-per-class 5000
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from src.preprocessing.anti_saturator import desaturate
from src.preprocessing.crop_generator import generate_crops_for_image
from src.preprocessing.loader import load_fits
from src.preprocessing.star_finder import find_stars_fits
from src.utils.constants import (
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
    folder = CLASS_FOLDERS[class_id]
    fits_dir = FITS_RAW_DIR / folder
    out_dir = CROPS_STAGING_DIR / folder

    if not fits_dir.exists() or not any(fits_dir.glob("*.fits")):
        print(f"[{folder}] no FITS files found in {fits_dir} — skipped. "
              f"Run 'python -m src.acquisition.run_all' first "
              f"(roman/rubin/gaia may need manual drops).")
        return 0

    existing = len(list(out_dir.glob("*.npy"))) if out_dir.exists() else 0
    if existing >= n_per_class:
        print(f"[{folder}] staging already has {existing} crops — skipped.")
        return existing

    fits_files = sorted(fits_dir.rglob("*.fits"))
    total = existing
    for idx, fpath in enumerate(tqdm(fits_files, desc=f"Crops/{folder}",
                                     unit="img")):
        if total >= n_per_class:
            break
        try:
            # Task 3.2: anti-saturation is mandatory for FITS with bright
            # stars, before star finding and cropping.
            img = desaturate(load_fits(fpath))
            positions = find_stars_fits(img)
            # Indexed prefix: files with the same stem in different
            # subfolders must not overwrite each other's crops.
            saved = generate_crops_for_image(
                img, positions, out_dir, prefix=f"{idx:05d}_{fpath.stem}",
                max_crops=max_crops_per_image)
            total += len(saved)
        except Exception as exc:  # noqa: BLE001 — skip corrupt sources
            print(f"[skip] {fpath.name}: {exc}")
    print(f"[{folder}] staged {total} crops.")
    return total


def _stratified_split(class_id: int, n_per_class: int) -> dict[str, int]:
    """Split one class's staged crops into train/val/test directories."""
    folder = CLASS_FOLDERS[class_id]
    src_dir = CROPS_STAGING_DIR / folder
    files = sorted(src_dir.glob("*.npy")) if src_dir.exists() else []
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
        from src.acquisition.altro_generator import generate_altro_crops
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
