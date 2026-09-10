"""Anti-saturation filter (Task 3.2).

The core of bright stars is saturated; the geometric fingerprint of the
diffraction spikes lives in the *outer wings*, which must be preserved.
Saturated pixels (>= 0.99) are replaced with the median of their
neighbours computed via a 3x3 ``scipy.ndimage.median_filter``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from src.utils.constants import MEDIAN_FILTER_SIZE, SATURATION_THRESHOLD

__all__ = ["desaturate"]


def desaturate(image: np.ndarray,
               threshold: float = SATURATION_THRESHOLD,
               kernel_size: int = MEDIAN_FILTER_SIZE) -> np.ndarray:
    """Replace saturated pixels with the median of their neighbourhood.

    Procedure (Task 3.2):
      1. find every pixel with value >= ``threshold`` (default 0.99);
      2. compute a median filter with a ``kernel_size`` x ``kernel_size``
         kernel (default 3x3) over the whole image;
      3. substitute the saturated pixels with the filtered values.

    Args:
        image: float32 image in [0, 1].
        threshold: saturation threshold.
        kernel_size: size of the median filter kernel.

    Returns:
        New float32 array with saturated pixels repaired.
    """
    img = np.asarray(image, dtype=np.float32)
    mask = img >= threshold
    if not mask.any():
        return img

    filtered = median_filter(img, size=kernel_size, mode="nearest")
    return np.where(mask, filtered, img).astype(np.float32)
