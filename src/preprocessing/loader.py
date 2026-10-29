"""Unique data loader for FITS and PNG/JPG images (Task 3.1).

Both branches return a ``float32`` numpy array normalized in ``[0, 1]``
(spec 3.3).

* **FITS**: read the first scientific HDU (HDU 0 or the ``SCI`` extension),
  sky-subtract with the image median, divide by the image standard
  deviation, then clamp negatives to 0 and values above the 99.5th
  percentile to 1.0.
* **PNG/JPG** (inference only): grayscale with the ITU-R 601 luma formula
  ``GRAY = 0.2989*R + 0.5870*G + 0.1140*B`` and divide by 255.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from PIL import Image

from Src.utils.constants import SKY_PERCENTILE_CLIP

__all__ = ["load_fits", "load_png", "load_image"]


def load_fits(path: str | np.ndarray) -> np.ndarray:
    """Load a FITS file (or an already-open HDU) and normalize it to [0, 1].

    Uses only the first scientific layer: the primary HDU if it holds data,
    otherwise the first ``SCI`` extension (Task 2.3). Files that contain
    only headers without data are rejected.

    Args:
        path: path to the FITS file, or an ``astropy.io.fits`` HDU object.

    Returns:
        float32 image array in [0, 1].

    Raises:
        ValueError: if no HDU contains image data (header-only file).
    """
    if isinstance(path, (fits.PrimaryHDU, fits.ImageHDU)):
        data = np.asarray(path.data)
    else:
        with fits.open(path, memmap=True) as hdul:
            data = None
            for hdu in hdul:                      # first scientific layer only
                if hdu.data is not None and getattr(hdu, "is_image", False) \
                        and hdu.data.size > 0:
                    data = hdu.data
                    break
            if data is None:
                raise ValueError(f"No image data found in FITS file: {path}")

    if data.ndim == 3:                            # (n, h, w) cube -> first slice
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}")

    img = data.astype(np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        raise ValueError(f"FITS image contains no finite pixels: {path}")

    # Replace NaN/inf with the sky median before computing statistics.
    sky_median = float(np.median(img[finite]))
    img = np.where(finite, img, sky_median)

    # Sky subtraction and normalization (Task 3.1).
    sky_std = float(np.std(img))
    if sky_std > 0:
        img = (img - sky_median) / sky_std

    # Clamp: negatives -> 0, above the 99.5th percentile -> 1.0.
    hi = float(np.percentile(img, SKY_PERCENTILE_CLIP))
    img = np.clip(img, 0.0, 1.0 if hi <= 0 else hi)
    if img.max() > 0:
        img = img / img.max()
    return img.astype(np.float32)


def load_png(path: str | Image.Image) -> np.ndarray:
    """Load a PNG/JPG image and normalize it like the FITS training data.

    Grayscale conversion uses the luminance formula from spec 3.1:
    ``GRAY = 0.2989*R + 0.5870*G + 0.1140*B``.

    Since the model is trained on ``load_fits`` output (sky-subtracted,
    std-normalized, percentile-clipped), a photo must go through the same
    normalization or the domain gap collapses confidence to ~1/N (observed
    2026-09-14: JPEG/gamma backgrounds at ~0.05 vs ~0.01 in training
    crops -> every prediction "Non determinabile").

    Args:
        path: path to the image, or an already-open PIL image.

    Returns:
        float32 2D image array in [0, 1], sky-normalized as in training.
    """
    pil_img = Image.open(path) if isinstance(path, (str, bytes)) else path
    if pil_img.mode not in ("L", "I", "I;16", "F"):
        pil_img = pil_img.convert("RGB")

    arr = np.asarray(pil_img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:      # color -> luma (Task 3.1)
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.max() > 1.5:                           # integer images
        # 8-bit -> /255; 16-bit -> /65535 (spec 3.1 uses /255 for 8-bit
        # PNG/JPG; 16-bit FITS-like PNGs must not be crushed to near-black).
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    # Sky normalization identical to load_fits (Task 3.1): without it the
    # inference distribution never matches the training one.
    sky_median = float(np.median(arr))
    sky_std = float(np.std(arr))
    if sky_std > 0:
        arr = (arr - sky_median) / sky_std
    hi = float(np.percentile(arr, SKY_PERCENTILE_CLIP))
    arr = np.clip(arr, 0.0, 1.0 if hi <= 0 else hi)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.astype(np.float32)


def load_image(path: str, mode: str | None = None) -> np.ndarray:
    """Dispatch to :func:`load_fits` or :func:`load_png` based on extension.

    Args:
        path: image path on disk.
        mode: optional explicit mode, ``"fits"`` or ``"png"``.

    Returns:
        float32 image array in [0, 1].
    """
    if mode is None:
        mode = "fits" if str(path).lower().endswith((".fits", ".fit", ".fts")) \
            else "png"
    return load_fits(path) if mode == "fits" else load_png(path)
