"""IRSA downloader for real Spitzer and WISE FITS images (Task 2.1).

IRSA implements the IVOA Simple Image Access v2 protocol, reachable from
astroquery via ``Irsa.query_sia``. Collection IDs are discovered at runtime
with ``Irsa.list_collections()`` so the downloader survives IRSA renaming
collections: it prefers ``spitzer*`` / ``wise*`` collections and filters
images by their band/energy metadata.

Usage::

    from src.acquisition.irsa_downloader import download_spitzer
    download_spitzer(n_images=200)
"""

from __future__ import annotations

import io
import random
import urllib.request
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from src.acquisition._common import download_images
from src.utils.constants import FITS_RAW_DIR, RAND_SEED

#: Dense star fields (Galactic plane / Magellanic Clouds) for PSF hunting.
STAR_FIELDS: list[tuple[str, str]] = [
    ("M22", "M22"),
    ("M11", "M11"),
    ("NGC 104", "47Tuc"),
    ("M15", "M15"),
    ("omega Cen", "omegacen"),
]


def _pick_collections(available, starts_with: tuple[str, ...]) -> list[str]:
    """Return IRSA collection IDs starting with the given prefixes.

    ``Irsa.list_collections()`` returns an astropy ``Table`` with a
    ``collection`` column, but older versions returned a list or an ASCII
    table string — every shape is normalized into a token stream here.
    """
    import re

    if isinstance(available, str):
        text = available
    elif hasattr(available, "colnames"):          # astropy Table
        if "collection" in available.colnames:
            text = "\n".join(str(v) for v in available["collection"])
        else:
            text = str(available)
    else:
        try:
            text = "\n".join(str(x) for x in available)
        except TypeError:
            text = str(available)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
    pattern = "^(?:" + "|".join(starts_with) + ")"
    return sorted({t for t in tokens if re.match(pattern, t, re.IGNORECASE)})


def _download_mission(mission: str,
                      prefixes: tuple[str, ...],
                      n_images: int,
                      out_dir: Path) -> int:
    """Generic IRSA SIAv2 loop for one mission (Spitzer or WISE)."""
    from astroquery.ipac.irsa import Irsa

    collections = _pick_collections(Irsa.list_collections(), prefixes)
    if not collections:
        print(f"[IRSA] no collection found for {mission}; "
              f"available: {list(Irsa.list_collections())}")
        return 0

    random.seed(RAND_SEED)
    fields = STAR_FIELDS[:]
    random.shuffle(fields)

    written = 0
    idx = 0
    while written < n_images and idx < len(fields) * 10:
        name, tag = fields[idx % len(fields)]
        idx += 1
        try:
            # SkyCoord.from_name resolves via SESAME. NOTE: SkyCoord(name)
            # would NOT resolve — it tries to parse the string as an angle.
            coord = SkyCoord.from_name(name)
            # SIA2 convention: pos is a (center, radius) tuple — a bare
            # SkyCoord raises 'radius should be provided in the pos tuple'.
            table = Irsa.query_sia(
                pos=(coord, 8 * u.arcmin),
                collection=collections,
                data_type="image",
                maxrec=50,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[IRSA {mission}] {name}: {exc}")
            continue
        if table is None or len(table) == 0:
            continue

        colnames = table.colnames
        url_col = next((c for c in colnames if "access" in c.lower()
                        and "url" in c.lower()), None) \
            or next((c for c in colnames if "url" in c.lower()), None)
        if url_col is None:
            continue

        def _fetch(table=table, url_col=url_col):
            hdulists = []
            for url in table[url_col][:30]:
                try:
                    with urllib.request.urlopen(str(url), timeout=120) as resp:
                        blob = resp.read()
                    # SIA2 rows may include previews/metadata that are not
                    # FITS at all — validate the magic bytes first.
                    if not blob.startswith(b"SIMPLE  ="):
                        continue
                    hdulists.append(fits.open(io.BytesIO(blob)))
                except Exception as exc:  # noqa: BLE001
                    print(f"[IRSA {mission}] download failed: {exc}")
            return hdulists

        written += download_images(_fetch, out_dir, f"{tag}", n_images - written)
    return written


def download_spitzer(n_images: int, out_dir: Path | None = None) -> int:
    """Download real Spitzer FITS images from IRSA.

    Args:
        n_images: target number of FITS files.
        out_dir: destination (default ``data/fits_raw/spitzer``).

    Returns:
        Number of FITS files written.
    """
    return _download_mission("Spitzer", ("spitzer",), n_images,
                             out_dir or FITS_RAW_DIR / "spitzer")


def download_wise(n_images: int, out_dir: Path | None = None) -> int:
    """Download real WISE/AllWISE FITS images from IRSA.

    Args:
        n_images: target number of FITS files.
        out_dir: destination (default ``data/fits_raw/wise``).

    Returns:
        Number of FITS files written.
    """
    return _download_mission("WISE", ("wise",), n_images,
                             out_dir or FITS_RAW_DIR / "wise")
