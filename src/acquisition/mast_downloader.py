"""MAST downloader for real Hubble and JWST FITS images (Task 2.1).

Uses ``astroquery.mast.Observations`` cone searches: every downloaded
product is a genuine FITS file from the Barbara A. Mikulski Archive for
Space Telescopes.

Usage::

    from src.acquisition.mast_downloader import download_hubble
    download_hubble(n_images=200)
"""

from __future__ import annotations

import random
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

from src.acquisition._common import download_images
from src.utils.constants import FITS_RAW_DIR, RAND_SEED

#: Well-known deep fields crowded with point sources, ideal for PSFs.
HUBBLE_FIELDS: list[tuple[str, str]] = [
    ("IC 1613", "IC1613"),            # Local Group dwarf galaxy
    ("IC 418", "IC418"),              # planetary nebula field
    ("M33", "M33"),                   # Local Group spiral
    ("NGC 6822", "NGC6822"),
    ("LMC", "LMC"),
]

JWST_FIELDS: list[tuple[str, str]] = [
    ("LMC", "LMC"),
    ("SMC", "SMC"),
    ("NGC 6822", "NGC6822"),
    ("IC 1613", "IC1613"),
    ("M33", "M33"),
]

#: Data-product extensions kept: DRZ/DRC (resampled, clean) and FLT
#: (calibrated, s/w-corrected). RAW/UNCAL are intentionally excluded.
_HST_EXT = ["DRZ", "DRC", "FLT"]
_JWST_EXT = ["CAL", "I2D"]


def _download_collection(collection: str,
                         fields: list[tuple[str, str]],
                         extensions: list[str],
                         n_images: int,
                         out_dir: Path,
                         radius_arcmin: float = 4.0) -> int:
    """Cone-search MAST around ``fields`` and download ``n_images`` FITS."""
    random.seed(RAND_SEED)
    order = fields[:]
    random.shuffle(order)
    # Resume support: files already on disk count toward the target so a
    # re-run does not re-download (or overwrite) existing data.
    out_dir.mkdir(parents=True, exist_ok=True)
    written = len(list(out_dir.glob("*.fits")))
    if written:
        print(f"[MAST {collection}] {written} files already present — resuming.")
    idx = 0

    while written < n_images and idx < len(order) * 10:
        name, tag = order[idx % len(order)]
        idx += 1
        try:
            # Pass the OBJECT NAME (a plain string): MAST resolves it via
            # SESAME. A SkyCoord(name) would try to *parse* the string as
            # coordinates and fail with 'Cannot parse ... for attribute ra'.
            table = Observations.query_criteria(
                coordinates=name,
                radius=f"{radius_arcmin} arcmin",
                obs_collection=collection,
                dataproduct_type="image",
            )
            if len(table) == 0:
                continue
            products = Observations.get_product_list(table[:20])
            filtered = Observations.filter_products(
                products,
                productSubGroupDescription=extensions,
                extension="fits",
            )
            if len(filtered) == 0:
                continue
            # astroquery does NOT create the download dir itself.
            dl_dir = out_dir / "_dl"
            dl_dir.mkdir(parents=True, exist_ok=True)
            manifest = Observations.download_products(
                filtered[: max(n_images - written, 1)],
                download_dir=str(dl_dir),
                flat=True,
                cache=False,
            )
        except Exception as exc:  # noqa: BLE001 — network errors keep going
            print(f"[MAST {collection}] {name}: {exc}")
            continue

        # The manifest lists what actually landed on disk (COMPLETE only).
        local_paths = []
        if manifest is not None and len(manifest) and "Local Path" in manifest.colnames:
            for row in manifest:
                if "Status" in manifest.colnames \
                        and str(row["Status"]) != "COMPLETE":
                    continue
                local_paths.append(str(row["Local Path"]))
        for src in local_paths:
            if written >= n_images:
                break
            try:
                from src.acquisition._common import has_image_data
                src_path = Path(src)
                if not has_image_data(src_path):
                    continue
                out = out_dir / f"{tag}_{written:04d}.fits"
                src_path.replace(out)
                written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[MAST {collection}] move failed: {exc}")
    return written


def download_hubble(n_images: int, out_dir: Path | None = None) -> int:
    """Download real Hubble (``obs_collection='HST'``) FITS images.

    Args:
        n_images: target number of FITS files.
        out_dir: destination (default ``data/fits_raw/hubble``).

    Returns:
        Number of FITS files written.
    """
    return _download_collection("HST", HUBBLE_FIELDS, _HST_EXT, n_images,
                                out_dir or FITS_RAW_DIR / "hubble")


def download_jwst(n_images: int, out_dir: Path | None = None) -> int:
    """Download real JWST (``obs_collection='JWST'``) FITS images.

    Args:
        n_images: target number of FITS files.
        out_dir: destination (default ``data/fits_raw/jwst``).

    Returns:
        Number of FITS files written.
    """
    return _download_collection("JWST", JWST_FIELDS, _JWST_EXT, n_images,
                                out_dir or FITS_RAW_DIR / "jwst")
