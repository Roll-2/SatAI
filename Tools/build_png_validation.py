"""Build a PNG/JPG validation set from the staged crops (inference only).

The model trains on FITS-derived crops, but the app will receive **PNG/JPG
photos** through the UI. This script materializes that reality: it renders
the real (post-quarantine) 64x64 staging crops into photo-like sky images —
mosaics of 8x8 crops (512x512), asinh stretch like astrophoto rendering,
sky background, 8-bit color-ish gray levels and JPEG/PNG compression — and
saves them to ``Data/png_validation/<class>/``.

Everything is bounded: ``--max-mb`` (default 500 MB) is a hard cap checked
before every write, and ``--per-class`` keeps the classes balanced. The
resulting folder is a *validation* asset, not training data (anti-pattern
list: training stays FITS-only).

After the real training finishes, measure the FITS→photo gap end-to-end::

    python Tools/satai.py --input Data/png_validation/hubble/hubble_000.jpg --mode png

Usage (from the project root)::

    .venv/bin/python Tools/build_png_validation.py
    .venv/bin/python Tools/build_png_validation.py --per-class 30 --max-mb 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Src.utils.constants import (  # noqa: E402
    CLASS_FOLDERS,
    CROPS_STAGING_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    RAND_SEED,
)

__all__ = ["build_png_validation"]

#: Output root (gitignored like every other generated data).
OUT_DIR: Path = DATA_DIR / "png_validation"

#: Manifest with the provenance of every rendered image.
MANIFEST: str = "manifest.txt"

#: Mosaic side in crops (8 crops x 64 px = 512 px images).
GRID: int = 8

#: JPEG quality range (photo-like variability).
JPEG_QUALITY: tuple[int, int] = (70, 95)

#: Sky background level added behind the crops (like a real photo's
#: light-polluted sky) and the asinh softening parameter.
SKY_BACKGROUND: float = 0.03
ASINH_ALPHA: float = 3.0


def _load_crops(folder: Path, per_image: int, rng: random.Random
                ) -> list[np.ndarray]:
    """Return ``per_image`` crops for one mosaic (post-quarantine only)."""
    files = [p for p in folder.glob("*.npy") if not p.name.startswith(".")]
    if not files:
        return []
    crops: list[np.ndarray] = []
    for _ in range(per_image * 3):          # tolerate unreadable files
        f = rng.choice(files)
        try:
            arr = np.load(f).astype(np.float32)
        except Exception:                   # noqa: BLE001
            continue
        if arr.shape == (64, 64):
            crops.append(arr)
        if len(crops) >= per_image:
            break
    return crops


def _render_mosaic(crops: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """Compose ``GRID x GRID`` crops into one photo-like image.

    Per-crop min-max stretch (contact-sheet convention), asinh softening
    of the bright cores and a small sky background make the tiles read
    like a rendered astrophoto instead of raw detector values.

    Tiles are separated by ``TILE_MARGIN`` px (and the canvas has a border
    of the same size): with zero margins the stars on tile edges were cut
    in half, producing inference crops centered on partial sources — the
    Predictor's edge-margin gate then discarded them.
    """
    tile, margin = 64, 32
    size = GRID * tile + (GRID + 1) * margin
    canvas = np.full((size, size), SKY_BACKGROUND, dtype=np.float32)
    rng.shuffle(crops)
    needed = GRID * GRID
    while len(crops) < needed:              # tile with repetition if short
        crops = crops + crops
    for k in range(needed):
        img = crops[k]
        lo, hi = float(img.min()), float(img.max())
        norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
        rendered = np.arcsinh(norm * ASINH_ALPHA) / np.arcsinh(ASINH_ALPHA)
        y = margin + (k // GRID) * (tile + margin)
        x = margin + (k % GRID) * (tile + margin)
        canvas[y:y + tile, x:x + tile] = rendered + SKY_BACKGROUND
    # Gentle Poisson-like noise so blobs are not mathematically clean.
    noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    canvas += rng.uniform(0.002, 0.01) * noise_rng.standard_normal(
        (size, size)).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def _save_image(img: np.ndarray, path: Path, fmt: str,
                rng: random.Random) -> int:
    """Write one 8-bit image (JPG or PNG) and return its size in bytes."""
    pil = Image.fromarray((img * 255.0 + 0.5).astype(np.uint8), mode="L")
    if fmt == "jpg":
        pil.save(path, quality=rng.randint(*JPEG_QUALITY))
    else:
        pil.save(path, compress_level=6)
    return path.stat().st_size


def build_png_validation(per_class: int = 20, max_mb: float = 500.0,
                         out_dir: Path | None = None,
                         classes: list[str] | None = None) -> dict[str, int]:
    """Render photo-like mosaics from the staged crops, under a size cap.

    Args:
        per_class: images per class (target, halved when space runs out).
        max_mb: hard cap for the whole output folder in megabytes.
        out_dir: output root (default ``Data/png_validation``).
        classes: subset of classes; defaults to all 9.

    Returns:
        ``{class: images_written}``.
    """
    rng = random.Random(RAND_SEED + 7)
    out = out_dir or OUT_DIR
    budget = max_mb * 1024 * 1024
    used = 0
    results: dict[str, int] = {}
    names = classes or CLASS_FOLDERS
    for name in names:
        staging = CROPS_STAGING_DIR / name
        have = len([p for p in staging.glob("*.npy")
                    if not p.name.startswith(".")])
        if have < GRID * GRID:
            print(f"[{name}] only {have} crops staged — skipped.")
            results[name] = 0
            continue
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        written = 0
        for i in range(per_class):
            crops = _load_crops(staging, GRID * GRID, rng)
            if len(crops) < GRID * GRID:
                break
            img = _render_mosaic(crops, rng)
            fmt = "jpg" if i % 2 == 0 else "png"   # both formats covered
            path = dest / f"{name}_{i:03d}.{fmt}"
            size = _save_image(img, path, fmt, rng)
            if used + size > budget:
                path.unlink(missing_ok=True)
                print(f"[cap] {max_mb} MB budget reached — stopping early.")
                break
            used += size
            written += 1
        results[name] = written
        print(f"[{name}] {written} photo-like images -> {dest}")
    (out / MANIFEST).write_text(
        "Generated by Tools/build_png_validation.py — mosaics of real "
        f"(quarantined-clean) staging crops, GRID={GRID}x{GRID}, "
        f"cap={max_mb} MB, total MB used: {used / 1024 / 1024:.1f}\n")
    print(f"\nDone: {sum(results.values())} images, "
          f"{used / 1024 / 1024:.1f} / {max_mb} MB used -> {out}")
    return results


def main() -> None:
    """CLI entry point for the PNG/JPG validation builder."""
    parser = argparse.ArgumentParser(
        description="Render photo-like PNG/JPG validation set from the "
                    "staged crops (bounded by --max-mb)")
    parser.add_argument("--per-class", type=int, default=20,
                        help="images per class (default 20)")
    parser.add_argument("--max-mb", type=float, default=500.0,
                        help="hard size cap of the whole folder (default 500)")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="subset of classes (default: all)")
    parser.add_argument("--out", default=None,
                        help="output dir (default Data/png_validation)")
    args = parser.parse_args()
    build_png_validation(per_class=args.per_class, max_mb=args.max_mb,
                         out_dir=Path(args.out) if args.out else None,
                         classes=args.classes)


if __name__ == "__main__":
    main()