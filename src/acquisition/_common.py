"""Shared helpers for the archive downloaders (Task 2)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

from astropy.io import fits

#: JWST EXP_TYPE fragments that mark a spectroscopic product (grisms,
#: slit/slitless spectra, MRS/LRS). Imaging-only classes must reject them.
_SPECTRO_EXP_TYPE = re.compile(r"SPEC|GRISM|MRS|LRS|WFSS", re.IGNORECASE)

#: HST grism/prism FILTER values (ACS G800L, WFC3 G102/G141, ...).
_HST_GRATING_FILTER = re.compile(r"^G\d+[LM]?$", re.IGNORECASE)


def is_spectroscopy_product(path: Path) -> bool:
    """True when a FITS file is a spectroscopic product, not an image.

    Checks the EXP_TYPE keyword (JWST: NRS_*, MIR_MRS, MIR_LRS-*,
    NIS_WFSS, NRC_GRISM, ...) and the FILTER keyword (HST gratings:
    G800L, G102, G141; JWST slit gratings like P750L). Files whose
    header cannot be read count as spectroscopic (rejected).
    """
    try:
        with fits.open(path, memmap=True) as hdul:
            hdr0 = hdul[0].header
            exp_type = str(hdr0.get("EXP_TYPE") or "")
            if _SPECTRO_EXP_TYPE.search(exp_type):
                return True
            for hdu in hdul:
                filt = str(hdu.header.get("FILTER") or "")
                # HST FILTER may be a semicolon-separated list.
                for token in filt.split(";"):
                    token = token.strip()
                    if _HST_GRATING_FILTER.match(token):
                        return True
                    if token.upper() in {"P750L", "LRS", "MRS"}:
                        return True
    except OSError:
        return True
    return False


def has_image_data(path: Path) -> bool:
    """Return True if a FITS file contains at least one 2-D image HDU.

    Header-only files must be rejected (Task 2.3).
    """
    try:
        with fits.open(path, memmap=True) as hdul:
            return any(
                hdu.data is not None
                and getattr(hdu, "is_image", False)
                and getattr(hdu.data, "ndim", 0) >= 2
                for hdu in hdul
            )
    except OSError:
        return False


def save_hdulist(hdulist: fits.HDUList, out_dir: Path, stem: str) -> Path:
    """Write an HDUList to ``out_dir`` with an auto-numbered unique stem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.fits"
    n = 1
    while out.exists():
        out = out_dir / f"{stem}_{n}.fits"
        n += 1
    hdulist.writeto(out, overwrite=True)
    return out


def download_images(
    fetch: Callable[[], object],
    out_dir: Path,
    prefix: str,
    target_count: int,
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """Persist every image HDUList produced by ``fetch()`` until the target.

    ``fetch`` must return either a single HDUList or an iterable of them.
    Corrupt or header-only files are skipped. Returns the number of FITS
    files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    items = fetch()
    if items is None:
        return 0
    if isinstance(items, fits.HDUList):
        items = [items]

    for hdulist in items:
        if written >= target_count or (is_cancelled and is_cancelled()):
            break
        try:
            if not isinstance(hdulist, fits.HDUList):
                continue
            if not any(
                hdu.data is not None
                and getattr(hdu.data, "ndim", 0) >= 2
                for hdu in hdulist
            ):
                continue  # header-only product -> reject (Task 2.3)
            hdulist.verify("fix")
            save_hdulist(hdulist, out_dir, f"{prefix}_{written:04d}")
            written += 1
        except Exception as exc:  # noqa: BLE001 — keep going on bad products
            print(f"[skip] {prefix}: {exc}")
    return written


def iter_batches(seq: Iterable, size: int):
    """Yield lists of ``size`` items from ``seq``."""
    batch: list = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
