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

from Src.preprocessing.star_finder import estimate_fwhm
from Src.utils.constants import (
    CROP_CENTER,
    CROP_FWHM_MAX,
    CROP_FWHM_MIN,
    CROP_SIZE,
    EDGE_MARGIN,
    MAX_CROPS_PER_IMAGE,
    MIN_CROP_SNR,
)

__all__ = ["extract_crops", "generate_crops_for_image", "generate_crops",
           "crop_qc_pass"]


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


def crop_qc_pass(crop: np.ndarray, fwhm_max: float | None = None,
                 min_snr: float | None = None) -> bool:
    """Mandatory quality gates applied to every crop before it is saved.

    A crop is rejected when any of these holds (diagnosis 2026-09-13:
    without these gates 92-97% of the staged crops were extended blobs):

    * the central peak is not significant: SNR < ``min_snr``
      (default :data:`MIN_CROP_SNR`) against the sigma-clipped sky
      noise of the crop itself — below ~10 sigma no diffraction
      structure is visible. Inference relaxes this gate: on an extended
      galaxy the crop's own std IS the source, so the point-source
      10-sigma rule would drop every "Altro" candidate;
    * the source is not point-like: second-moment FWHM estimated on the
      crop is outside ``[CROP_FWHM_MIN, fwhm_max]`` (a hot pixel
      or an extended galaxy/cluster core, never a PSF). Training keeps
      the strict ``CROP_FWHM_MAX``; inference may relax it so extended
      objects still reach the model and get classified as "Altro";
    * the 3x3 core around the peak is completely flat: 8+ of 9 pixels
      within 0.5% of the peak value — a blown star the anti-saturator
      could not repair (no usable PSF information). A small plateau is
      normal: the loader clips at the 99.5th percentile (spec 3.1), so
      bright cores land exactly at 1.0 while the spikes survive below.
    """
    from astropy.stats import sigma_clipped_stats

    c = CROP_SIZE // 2
    peak = float(crop[c, c])

    # 1. Central significance vs the crop's own sky noise.
    snr_floor = MIN_CROP_SNR if min_snr is None else min_snr
    _, med, std = sigma_clipped_stats(crop, sigma=3.0)
    if (peak - float(med)) / (float(std) + 1e-8) < snr_floor:
        return False

    # 2. Point-like size gate (second-moment FWHM at half maximum).
    fmax = CROP_FWHM_MAX if fwhm_max is None else fwhm_max
    fwhm = estimate_fwhm(crop, c, c)
    if fwhm is None or not (CROP_FWHM_MIN <= fwhm <= fmax):
        return False

    # 3. Blown core: the whole 3x3 plateau is at the ceiling.
    core3 = crop[c - 1:c + 2, c - 1:c + 2]
    if int((core3 >= 0.995 * peak).sum()) >= 8:
        return False

    return True


def extract_crops(image: np.ndarray,
                  positions: np.ndarray,
                  max_crops: int = MAX_CROPS_PER_IMAGE,
                  fwhm_max: float | None = None,
                  min_snr: float | None = None) -> np.ndarray:
    """Extract up to ``max_crops`` centered 64x64 crops from an image.

    Every candidate passes the :func:`crop_qc_pass` gates (SNR, FWHM,
    saturation) before being accepted.

    Args:
        image: float32 image (any size >= 64x64).
        positions: ``(N, 2)`` array of ``(x, y)`` candidate star centers.
        max_crops: maximum number of crops returned (spec: 20 per image).
        fwhm_max: optional relaxed FWHM ceiling (inference uses it so
            extended objects reach the model instead of being dropped).
        min_snr: optional relaxed SNR floor (inference: extended sources
            inflate the crop's own std, see :func:`crop_qc_pass`).

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
        if not crop_qc_pass(crop, fwhm_max=fwhm_max, min_snr=min_snr):
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
            (e.g. :func:`Src.preprocessing.star_finder.find_stars_fits`).
        out_dir: destination directory for the ``.npy`` crops.
        max_crops: cap on crops per source image.

    Returns:
        Total number of crops saved.
    """
    from Src.preprocessing.anti_saturator import desaturate
    from Src.preprocessing.loader import load_fits

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
