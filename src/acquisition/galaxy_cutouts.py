"""Legacy Survey FITS cutout downloader (class 8 "Altro" — galaxies).

Downloads real galaxy-field cutouts from the public Legacy Survey
cutout service (https://www.legacysurvey.org). Positions are sampled
inside the survey's sky footprint; the returned files are genuine FITS
images dominated by galaxies rather than point sources, used for the
20% "galaxies without point-like stars" share of the "Altro" class.

Usage::

    from src.acquisition.galaxy_cutouts import download_galaxies
    download_galaxies(n_images=100)
"""

from __future__ import annotations

import random
import time
import urllib.request
from pathlib import Path

from src.utils.constants import FITS_RAW_DIR, RAND_SEED

CUTOUT_URL = ("https://www.legacysurvey.org/viewer/fits-cutout"
              "?ra={ra:.5f}&dec={dec:.5f}&size={size}&layer={layer}")

# SDSS footprint approximations (north + south Galactic caps).
_SDSS_PLATES_NORTH = [(110.0, 245.0, 5.0, 65.0)]      # ra range, dec range
_SDSS_PLATES_SOUTH = [(315.0, 60.0, -10.0, 40.0)]


def _random_sdss_position(rng: random.Random) -> tuple[float, float]:
    """Sample a random RA/Dec inside the (approximate) SDSS footprint."""
    plate = rng.choice(_SDSS_PLATES_NORTH + _SDSS_PLATES_SOUTH)
    ra0, ra1, dec0, dec1 = plate
    if ra0 <= ra1:
        ra = rng.uniform(ra0, ra1)
    else:  # wraps around 0
        ra = rng.uniform(ra0, 360.0) if rng.random() < 0.5 \
            else rng.uniform(0.0, ra1)
    return ra, rng.uniform(dec0, dec1)


def download_galaxies(n_images: int,
                      out_dir: Path | None = None,
                      layer: str = "ls-dr10",
                      size: int = 256) -> int:
    """Download real Legacy Survey FITS cutouts for the "Altro" class.

    Args:
        n_images: target number of cutouts.
        out_dir: destination (default ``data/fits_raw/altro/galaxies``).
        layer: Legacy Survey data layer (``ls-dr10`` or ``ls-dr9``).
        size: cutout size in pixels.

    Returns:
        Number of FITS cutouts written.
    """
    out_path = out_dir or (FITS_RAW_DIR / "altro" / "galaxies")
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RAND_SEED)
    written = 0
    attempts = 0
    while written < n_images and attempts < n_images * 3:
        attempts += 1
        ra, dec = _random_sdss_position(rng)
        url = CUTOUT_URL.format(ra=ra, dec=dec, size=size, layer=layer)
        dest = out_path / f"galaxy_{written:04d}.fits"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SatAi/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            # Basic validation: the service returns a FITS on success.
            if not blob.startswith(b"SIMPLE  ="):
                continue
            dest.write_bytes(blob)
            written += 1
        except Exception as exc:  # noqa: BLE001 — network resilience
            print(f"[LegacySurvey] ({ra:.2f}, {dec:.2f}): {exc}")
            time.sleep(1.0)
    return written
