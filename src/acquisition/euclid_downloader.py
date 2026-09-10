"""Euclid downloader for real FITS images (Task 2.1).

Verified live against the ESA Euclid Science Archive (astroquery 0.4.11):

* ``ESASky.get_maps`` does **not** support Euclid downloads yet
  ("EUCLID-VIS does not yet support downloading of fits files"), so this
  module uses the dedicated ``astroquery.esa.euclid`` TAP service.
* Downloadable products live in ``ivoa.obscore`` with a real
  ``file_name=`` inside ``access_url``; ``Euclid.get_product(file_name=...)``
  fetches them.
* Public Q1 coverage is limited to the Deep Fields (EDF-North, EDF-South,
  EDF-Fornax) — globular clusters / LMC have zero rows (verified).
* **Size warning**: VIS Level-2 calibrated frames are ~7 GB each
  (2066x2048). A handful already contains thousands of stars, which is
  enough for the 5000-crop class target.

Usage::

    from src.acquisition.euclid_downloader import download_euclid
    download_euclid(n_images=3)
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

from src.utils.constants import FITS_RAW_DIR, RAND_SEED

#: Public Q1 fields as (tag, ra_min, ra_max, dec_min, dec_max) ADQL boxes.
_EDF_BOXES: list[tuple[str, float, float, float, float]] = [
    ("edfn", 253.0, 298.0, 60.0, 71.0),    # EDF-North
    ("edfs", 35.0, 85.0, -54.0, -43.0),    # EDF-South
]

_ADQL = """SELECT TOP {top} obs_id, instrument_name, access_url
FROM ivoa.obscore
WHERE dataproduct_type='image'
  AND instrument_name IN ('VIS', 'NISP')
  AND access_url LIKE '%.fits%'
  AND s_ra BETWEEN {ra0} AND {ra1}
  AND s_dec BETWEEN {dec0} AND {dec1}"""


def _query_frame_urls(ra_range: tuple[float, float],
                      dec_range: tuple[float, float],
                      top: int) -> list[tuple[str, str]]:
    """Return (instrument, file_name) pairs for downloadable EDF frames."""
    from astroquery.esa.euclid import Euclid

    ra0, ra1 = ra_range
    dec0, dec1 = dec_range
    query = _ADQL.format(top=top, ra0=ra0, ra1=ra1, dec0=dec0, dec1=dec1)
    job = Euclid.launch_job(query)
    table = job.get_results()

    out: list[tuple[str, str]] = []
    for row in range(len(table)):
        url = str(table["access_url"][row])
        if "file_name=" not in url or not url.rstrip().endswith(".fits"):
            continue
        fname = url.split("file_name=")[1]
        instrument = str(table["instrument_name"][row])
        out.append((instrument, fname))
    return out


def download_euclid(n_images: int, out_dir: Path | None = None) -> int:
    """Download real Euclid VIS/NISP calibrated frames from the EDFs.

    Args:
        n_images: target number of FITS files. WARNING: ~7 GB per file —
            3-5 frames are already enough for the 5000-crop class target.
        out_dir: destination (default ``data/fits_raw/euclid``).

    Returns:
        Number of FITS files written.
    """
    from astroquery.esa.euclid import Euclid

    out_path = out_dir or FITS_RAW_DIR / "euclid"
    out_path.mkdir(parents=True, exist_ok=True)

    random.seed(RAND_SEED)
    box = random.choice(_EDF_BOXES)
    tag, ra0, ra1, dec0, dec1 = box

    written = len(list(out_path.glob("*.fits")))       # resume support
    if written:
        print(f"[Euclid] {written} files already present — resuming.")
    if written >= n_images:
        return written

    try:
        frames = _query_frame_urls((ra0, ra1), (dec0, dec1),
                                   top=(n_images - written) * 2 + 4)
    except Exception as exc:  # noqa: BLE001
        print(f"[Euclid] TAP query failed: {exc}")
        return written

    random.shuffle(frames)
    for instrument, fname in frames:
        if written >= n_images:
            break
        dest = out_path / fname
        if dest.exists():
            written += 1
            continue
        try:
            print(f"[Euclid] downloading {instrument} frame "
                  f"({fname[:44]}...) — ~7 GB, be patient")
            Euclid.get_product(file_name=fname, output_file=str(dest))
            if dest.exists() and dest.stat().st_size > 1_000_000:
                written += 1
            else:
                dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[Euclid] download failed: {exc}")
            dest.unlink(missing_ok=True)
            # A failed streaming download can leave a partial file behind;
            # fall back to a direct fetch of the same product URL.
            try:
                url = ("https://eas.esac.esa.int/sas-dd/data"
                       f"?RETRIEVAL_TYPE=FILE&release=sedm&file_name={fname}")
                req = urllib.request.Request(url,
                                             headers={"User-Agent":
                                                      "SatAi/1.0"})
                with urllib.request.urlopen(req, timeout=3600) as resp, \
                        open(dest, "wb") as fh:
                    while chunk := resp.read(1 << 22):
                        fh.write(chunk)
                if dest.exists() and dest.stat().st_size > 1_000_000:
                    written += 1
            except Exception as exc2:  # noqa: BLE001
                print(f"[Euclid] direct fetch failed: {exc2}")
                dest.unlink(missing_ok=True)
    return written
