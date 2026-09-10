"""Crop generator: extracts fixed 64x64 crops centered on star peaks (Task 4.3).

Validation rules from the spec:
  * size is always ``64 x 64``;
  * the pixel ``(32, 32)`` of each crop is the intensity peak;
  * a crop is discarded if the peak lies too close to a border
    (distance < 32 px from any edge);
  * at most 20 crops per source image;
  * crops are saved as individual ``.npy`` files.

The peak re-centering is done with a sub-window local maximum search so
that the brightest pixel ends up exactly at the crop centre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.constants import (
    CROP_CENTER,
    CROP_SIZE,
    EDGE_MARGIN,
    MAX_CROPS_PER_IMAGE,
)

__all__ = ["extract_crops", "generate_crops_for_image", "generate_crops"]


def _recenter_to_peak(image: np.ndarray, x: float, y: float) -> tuple[int, int] | None:
    """Return integer peak coordinates near (x, y), or None if invalid."""
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None

    # Search the local maximum in a small window around the detection.
    r = 3
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    patch = image[y0:y1, x0:x1]
    dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
    return x0 + int(dx), y0 + int(dy)


def extract_crops(image: np.ndarray,
                  positions: np.ndarray,
                  max_crops: int = MAX_CROPS_PER_IMAGE) -> np.ndarray:
    """Extract up to ``max_crops`` centered 64x64 crops from an image.

    Args:
        image: float32 image (any size >= 64x64).
        positions: ``(N, 2)`` array of ``(x, y)`` candidate star centers.
        max_crops: maximum number of crops returned (spec: 20 per image).

    Returns:
        ``(M, 64, 64)`` float32 array with ``0 <= M <= max_crops``.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    half = CROP_SIZE // 2
    crops: list[np.ndarray] = []

    for x, y in positions:
        if len(crops) >= max_crops:
            break
        peak = _recenter_to_peak(img, float(x), float(y))
        if peak is None:
            continue
        px, py = peak

        # Edge validation: the peak must be at least 32 px from every border.
        if px < EDGE_MARGIN or py < EDGE_MARGIN \
                or px >= w - EDGE_MARGIN or py >= h - EDGE_MARGIN:
            continue

        crop = img[py - half:py + half, px - half:px + half]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            continue
        crops.append(crop)

    return np.stack(crops, axis=0) if crops else np.empty(
        (0, CROP_SIZE, CROP_SIZE), dtype=np.float32)


def generate_crops_for_image(image: np.ndarray,
                             positions: np.ndarray,
                             out_dir: str | Path,
                             prefix: str,
                             max_crops: int = MAX_CROPS_PER_IMAGE) -> list[Path]:
    """Extract crops from one image and save them as individual ``.npy`` files.

    Args:
        image: float32 source image.
        positions: ``(N, 2)`` star coordinates from a star finder.
        out_dir: destination directory (created if missing).
        prefix: filename prefix (usually the source file stem).
        max_crops: cap on crops per source image (Task 4.3).

    Returns:
        List of paths of the saved crop files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    crops = extract_crops(image, positions, max_crops=max_crops)
    saved: list[Path] = []
    for i, crop in enumerate(crops):
        fpath = out_path / f"{prefix}_crop{i:02d}.npy"
        np.save(fpath, crop)
        saved.append(fpath)
    return saved


def generate_crops(image_paths: list[str | Path],
                   finder,
                   out_dir: str | Path,
                   max_crops: int = MAX_CROPS_PER_IMAGE) -> int:
    """Run the full find->crop->save pipeline over a list of FITS images.

    Args:
        image_paths: list of FITS file paths.
        finder: callable ``image -> (N, 2)`` positions
            (e.g. :func:`src.preprocessing.star_finder.find_stars_fits`).
        out_dir: destination directory for the ``.npy`` crops.
        max_crops: cap on crops per source image.

    Returns:
        Total number of crops saved.
    """
    from src.preprocessing.anti_saturator import desaturate
    from src.preprocessing.loader import load_fits

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for fpath in tqdm(image_paths, desc="Crops", unit="img"):
        try:
            img = desaturate(load_fits(fpath))
            positions = finder(img)
            prefix = Path(fpath).stem
            total += len(generate_crops_for_image(
                img, positions, out_path, prefix, max_crops=max_crops))
        except Exception as exc:  # noqa: BLE001 — skip corrupt sources
            print(f"[skip] {fpath}: {exc}")
    return total
