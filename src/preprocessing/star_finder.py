"""Star finders for FITS and PNG/JPG images (Task 4).

* ``find_stars_fits``: :class:`photutils.detection.DAOStarFinder` with
  ``fwhm=2.0`` and ``threshold=5.0 * sky_std``. The stars are sorted by
  decreasing brightness, the 5 brightest and the 10 faintest are dropped,
  and the next 20 (intermediate magnitude) are kept.
* ``find_stars_png``: :func:`skimage.feature.blob_log` with
  ``min_sigma=1, max_sigma=5, num_sigma=10, threshold=0.1``. The top 10
  blobs by intensity are returned.
"""

from __future__ import annotations

import numpy as np

from src.utils.constants import (
    BLOB_MAX_SIGMA,
    BLOB_MIN_SIGMA,
    BLOB_NUM_SIGMA,
    BLOB_THRESHOLD,
    BLOB_TOP_K,
    DAO_DROP_BRIGHTEST,
    DAO_DROP_FAINT,
    DAO_FWHM,
    DAO_KEEP,
    DAO_THRESHOLD_SIGMA,
)

__all__ = ["find_stars_fits", "find_stars_png"]


def find_stars_fits(image: np.ndarray) -> np.ndarray:
    """Detect stars in a preprocessed FITS image with DAOStarFinder.

    Args:
        image: float32 image already normalized by the loader (in [0, 1]).

    Returns:
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates, brightest-first,
        trimmed to the intermediate-magnitude selection of spec 4.1.
        Empty array when fewer than 16 stars are detected (the spec's
        5-brightest + 10-faintest trimming needs at least that many).
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    img = np.asarray(image, dtype=np.float32)
    # Robust sky statistics (sigma-clipped): the 5-sigma threshold of the
    # spec must be measured against the *sky* std, not the global std,
    # which is inflated by the stars themselves and would kill detection.
    _, sky_median, sky_std = sigma_clipped_stats(img, sigma=3.0)
    sky_median, sky_std = float(sky_median), float(sky_std)
    if sky_std <= 0:
        return np.empty((0, 2), dtype=np.float32)

    finder = DAOStarFinder(fwhm=DAO_FWHM,
                           threshold=DAO_THRESHOLD_SIGMA * sky_std)
    table = finder(img - sky_median)
    if table is None or len(table) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Sort by decreasing flux (luminosita' decrescente).
    flux = np.asarray(table["flux"], dtype=np.float64)
    order = np.argsort(flux)[::-1]
    xs = np.asarray(table["xcentroid"], dtype=np.float64)[order]
    ys = np.asarray(table["ycentroid"], dtype=np.float64)[order]

    # Drop the 5 brightest (too saturated) and the 10 faintest (too weak),
    # then keep the next 20 (intermediate magnitude).
    start = min(DAO_DROP_BRIGHTEST, len(xs))
    stop = min(start + DAO_KEEP, max(len(xs) - DAO_DROP_FAINT, start))
    xs, ys = xs[start:stop], ys[start:stop]
    return np.column_stack([xs, ys]).astype(np.float32)


def find_stars_png(image: np.ndarray) -> np.ndarray:
    """Detect star-like blobs in a PNG/JPG image with blob_log.

    Args:
        image: float32 image in [0, 1] (grayscale).

    Returns:
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates of the top-10
        blobs sorted by decreasing intensity (spec 4.2).
    """
    from skimage.feature import blob_log

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 0:
        img = img / img.max()

    blobs = blob_log(img,
                     min_sigma=BLOB_MIN_SIGMA,
                     max_sigma=BLOB_MAX_SIGMA,
                     num_sigma=BLOB_NUM_SIGMA,
                     threshold=BLOB_THRESHOLD)
    if len(blobs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # blob_log returns (y, x, sigma); sort by blob intensity proxy = sigma
    # is not meaningful, so rank by peak pixel value around each blob.
    h, w = img.shape
    scores = []
    for y, x, _sigma in blobs:
        yi, xi = int(round(y)), int(round(x))
        y0, y1 = max(0, yi - 2), min(h, yi + 3)
        x0, x1 = max(0, xi - 2), min(w, xi + 3)
        patch = img[y0:y1, x0:x1]
        scores.append(float(patch.max()) if patch.size else 0.0)

    order = np.argsort(scores)[::-1][:BLOB_TOP_K]
    xs = np.asarray(blobs)[:, 1][order]
    ys = np.asarray(blobs)[:, 0][order]
    return np.column_stack([xs, ys]).astype(np.float32)
