"""Shared helpers for the archive downloaders (Task 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from astropy.io import fits


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
