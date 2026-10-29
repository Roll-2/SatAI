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

from Src.utils.constants import (
    BLOB_MAX_SIGMA,
    BLOB_MIN_SIGMA,
    BLOB_NUM_SIGMA,
    BLOB_DEDUP_RADIUS,
    BLOB_SNR_MIN,
    BLOB_THRESHOLD,
    BLOB_TOP_K,
    DAO_DROP_BRIGHTEST,
    DAO_DROP_FAINT,
    DAO_FWHM,
    DAO_KEEP,
    DAO_THRESHOLD_SIGMA,
    FINDER_PARAMS,
)

__all__ = ["find_stars_fits", "find_stars_png", "estimate_fwhm"]


def _finder_params(telescope: str | None) -> tuple[float, float]:
    """Resolve (fwhm, threshold_sigma) for one telescope.

    ``jwst`` images may carry an INSTRUME hint (``jwst_miri``) — callers
    that know the instrument pass the specialized key directly.
    Unknown telescopes fall back to the spec defaults.
    """
    if telescope and telescope in FINDER_PARAMS:
        p = FINDER_PARAMS[telescope]
        return float(p["fwhm"]), float(p["threshold_sigma"])
    return DAO_FWHM, DAO_THRESHOLD_SIGMA


def estimate_fwhm(image: np.ndarray, x: int, y: int) -> float | None:
    """Second-moment FWHM estimate of the source centered at (x, y).

    Uses the pixels at or above the half-maximum level inside a 21x21
    window: for a point source this tracks the PSF width (1.5-8 px for
    the archives used here); for extended blobs it explodes beyond 10 px.
    Returns ``None`` when no significant peak exists in the window.
    """
    h, w = image.shape
    half = 10
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    win = image[y0:y1, x0:x1]
    from astropy.stats import sigma_clipped_stats
    _, med, std = sigma_clipped_stats(win, sigma=3.0)
    peak = float(win.max())
    if peak <= float(med) + float(std):
        return None
    mask = win >= med + 0.5 * (peak - med)
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 1.0                      # single hot pixel -> minimum width
    r2 = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
    return 2.0 * float(np.sqrt(r2.mean()))


def find_stars_fits(image: np.ndarray,
                    telescope: str | None = None) -> np.ndarray:
    """Detect stars in a preprocessed FITS image with DAOStarFinder.

    Args:
        image: float32 image already normalized by the loader (in [0, 1]).
        telescope: optional class folder name; selects the per-telescope
            finder parameters (FWHM + threshold) from
            :data:`Src.utils.constants.FINDER_PARAMS` — the spec default
            (2.0 px, 5 sigma) is wrong for undersampled archives like WISE.

    Returns:
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates, brightest-first,
        trimmed to the intermediate-magnitude selection of spec 4.1.
        Empty array when fewer than 16 stars are detected (the spec's
        5-brightest + 10-faintest trimming needs at least that many).
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    fwhm, thr_sigma = _finder_params(telescope)

    img = np.asarray(image, dtype=np.float32)
    # Robust sky statistics (sigma-clipped): the 5-sigma threshold of the
    # spec must be measured against the *sky* std, not the global std,
    # which is inflated by the stars themselves and would kill detection.
    _, sky_median, sky_std = sigma_clipped_stats(img, sigma=3.0)
    sky_median, sky_std = float(sky_median), float(sky_std)
    if sky_std <= 0:
        return np.empty((0, 2), dtype=np.float32)

    finder = DAOStarFinder(fwhm=fwhm,
                           threshold=thr_sigma * sky_std)
    table = finder(img - sky_median)
    if table is None or len(table) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Sort by decreasing flux (luminosita' decrescente).
    flux = np.asarray(table["flux"], dtype=np.float64)
    order = np.argsort(flux)[::-1]
    xs = np.asarray(table["xcentroid"], dtype=np.float64)[order]
    ys = np.asarray(table["ycentroid"], dtype=np.float64)[order]

    # Keep the BRIGHTEST stars (their spikes carry the 4-vs-6 line
    # signature — the desaturator repairs their cores) and drop the
    # faintest; then keep the next DAO_KEEP (intermediate magnitude).
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

    # SNR gate against the image's own sigma-clipped sky: the model was
    # trained on crops whose peaks stand >= ~10 sigma above their sky, so a
    # texture/noise blob from an arbitrary photo would be out-of-distribution
    # — and the classifier would still label it confidently (bug 2026-09-14:
    # a satellite photo -> "JWST 99.7%").
    from astropy.stats import sigma_clipped_stats
    _, sky_med, sky_std = sigma_clipped_stats(img, sigma=3.0)

    # blob_log returns (y, x, sigma); sort by blob intensity proxy = sigma
    # is not meaningful, so rank by peak pixel value around each blob.
    h, w = img.shape
    passing: list[tuple[float, float, float]] = []   # (peak, y, x)
    for y, x, _sigma in blobs:
        yi, xi = int(round(y)), int(round(x))
        y0, y1 = max(0, yi - 2), min(h, yi + 3)
        x0, x1 = max(0, xi - 2), min(w, xi + 3)
        patch = img[y0:y1, x0:x1]
        peak = float(patch.max()) if patch.size else 0.0
        if (peak - float(sky_med)) / (float(sky_std) + 1e-8) < BLOB_SNR_MIN:
            continue
        passing.append((peak, y, x))

    if not passing:
        return np.empty((0, 2), dtype=np.float32)
    passing.sort(key=lambda t: t[0], reverse=True)

    # Proximity dedup: blob_log reports the same bright plateau several
    # times (once per sigma scale). Multi-crop voting (Task 10) is meant
    # to aggregate DIFFERENT sources; letting one blob vote 6 times turned
    # noise/texture junk into a unanimous confident label (bug 14/09:
    # photo of a satellite -> "JWST 99.7%"). Keep the brightest blob per
    # BLOB_DEDUP_RADIUS neighborhood.
    kept: list[tuple[float, float, float]] = []
    for peak, y, x in passing:
        if all((y - ky) ** 2 + (x - kx) ** 2 >= BLOB_DEDUP_RADIUS ** 2
               for _kp, ky, kx in kept):
            kept.append((peak, y, x))
        if len(kept) >= BLOB_TOP_K:
            break

    return np.asarray([[x, y] for _peak, y, x in kept], dtype=np.float32)
