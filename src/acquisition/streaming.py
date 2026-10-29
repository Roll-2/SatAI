"""Streaming acquisition: download -> crop 64x64 -> delete FITS immediately.

Low-disk mode (peak usage ~1 GB even for a full campaign): source FITS
files are temporary raw material and are removed as soon as their crops
are extracted. Only the tiny ``.npy`` crops (16 KB each) survive, so the
final dataset occupies under 1 GB for the 45,000-crop target.

Because every source FITS is deleted after cropping, classes handled here
never appear in ``Data/fits_raw/`` at the end: the crops are written
*directly* into ``Data/crops/staging/<class>/`` and the stratified split
of :mod:`Src.data.build_dataset` can then run on its own (it detects the
already-staged crops and skips re-cropping).
Already-seen source FITS (tracked by md5 across batches) are skipped,
so overlapping archive results do not inflate the dataset with
duplicate crops.

Euclid is excluded by default in streaming mode: its Q1 VIS frames are
~7 GB each — a disk-constrained machine cannot hold them.

Usage (from the project root)::

    python -m Src.acquisition.streaming --per-class 300
    python -m Src.acquisition.streaming --telescope hubble
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path

from Src.utils.constants import CROPS_STAGING_DIR, FITS_RAW_DIR

#: Classes downloadable with the streaming pipeline (small FITS only).
#: Euclid is excluded on purpose: ~7 GB per frame (see module docstring).
STREAMABLE: list[str] = ["hubble", "jwst", "spitzer", "wise"]

#: Max crops per source image (spec 4.3).
MAX_CROPS_PER_IMAGE: int = 20

#: Name of the per-class dedup ledger (inside the staging folder).
SEEN_LEDGER: str = ".seen_md5"


def _load_seen(staging: Path) -> set[str]:
    """Load the md5 ledger of already-cropped sources (resume-safe dedup)."""
    ledger = staging / SEEN_LEDGER
    if ledger.exists():
        return {line for line in ledger.read_text().split() if line}
    return set()


def _save_seen(staging: Path, seen: set[str]) -> None:
    """Persist the md5 ledger so a killed run never duplicates crops."""
    staging.mkdir(parents=True, exist_ok=True)
    (staging / SEEN_LEDGER).write_text("\n".join(sorted(seen)) + "\n")


def _crops_from_dir(src_dir: Path, out_dir: Path, tag: str,
                    delete_fits: bool, seen: set[str] | None = None) -> int:
    """Crop every FITS in ``src_dir`` into ``out_dir``; delete sources if asked.

    Args:
        src_dir: folder containing the just-downloaded FITS batch.
        out_dir: destination staging directory for the ``.npy`` crops.
        tag: prefix for crop filenames (the telescope folder name).
        delete_fits: when True, every source FITS is deleted right after
            its crops are extracted (and astroquery's ``_dl`` temp dir is
            removed as well).

    Returns:
        Number of crops extracted from this batch.
    """
    from Src.preprocessing.anti_saturator import desaturate
    from Src.preprocessing.crop_generator import generate_crops_for_image
    from Src.preprocessing.loader import load_fits
    from Src.preprocessing.star_finder import find_stars_fits

    seen = seen if seen is not None else set()
    fits_files = sorted(p for p in src_dir.rglob("*.fits")
                        if "_dl" not in p.parts)
    total, dup = 0, 0
    for idx, fpath in enumerate(fits_files):
        try:
            # Different archives overlap: an identical source would only
            # duplicate crops, so it is skipped by content hash.
            digest = hashlib.md5(fpath.read_bytes()).hexdigest()
            if digest in seen:
                dup += 1
                continue
            seen.add(digest)   # ledger first: no infinite re-processing
            # Spectroscopic products (JWST grisms/LRS, HST grating FILTERs)
            # carry no PSF: reject before any processing.
            from Src.acquisition._common import is_spectroscopy_product
            if is_spectroscopy_product(fpath):
                continue
            # Task 3.2 anti-saturation -> Task 4.1 star finding (with the
            # per-telescope FWHM/threshold) -> Task 4.3 cropping with the
            # mandatory QC gates: the same chain build_dataset uses.
            img = desaturate(load_fits(fpath))
            positions = find_stars_fits(img, telescope=tag)
            saved = generate_crops_for_image(
                img, positions, out_dir,
                prefix=f"{tag}_{idx:04d}_{fpath.stem}",
                max_crops=MAX_CROPS_PER_IMAGE)
            total += len(saved)
        except Exception as exc:  # noqa: BLE001 — skip corrupt sources
            print(f"  [skip] {fpath.name}: {exc}")
        finally:
            if delete_fits:
                fpath.unlink(missing_ok=True)
    if delete_fits:
        shutil.rmtree(src_dir / "_dl", ignore_errors=True)
    if dup:
        print(f"  [dedup] {dup} duplicate source(s) skipped.")
    return total


def _stream_class(name: str, n_images: int, batch_size: int,
                  delete_fits: bool, staging_root: Path | None) -> int:
    """Run one class: download in batches, crop each batch, delete FITS.

    Args:
        name: telescope folder name (one of :data:`STREAMABLE`).
        n_images: total source images to pull (across all batches).
        batch_size: FITS files per download batch.
        delete_fits: delete each source FITS after cropping.
        staging_root: optional override of the crops staging root
            (tests use a temp dir; production uses the default).

    Returns:
        Total crops staged for this class.
    """
    if name == "hubble":
        from Src.acquisition.mast_downloader import download_hubble as dl
    elif name == "jwst":
        from Src.acquisition.mast_downloader import download_jwst as dl
    elif name == "spitzer":
        from Src.acquisition.irsa_downloader import download_spitzer as dl
    elif name == "wise":
        from Src.acquisition.irsa_downloader import download_wise as dl
    else:  # pragma: no cover — guarded by STREAMABLE
        raise ValueError(f"class not streamable: {name}")

    staging = (staging_root or CROPS_STAGING_DIR) / name
    tmp_dir = FITS_RAW_DIR / name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(staging)

    existing = len(list(staging.glob("*.npy"))) if staging.exists() else 0
    if existing:
        print(f"[{name}] {existing} crops already staged — continuing.")
    target_crops = n_images * MAX_CROPS_PER_IMAGE
    if existing >= target_crops:
        print(f"[{name}] target already reached ({existing} crops).")
        return existing

    total_crops = existing
    # Resume point derived from the dedup ledger: N processed sources
    # means N // batch_size download batches were already completed.
    # Floor division can only round *down*, so a resumed run replays at
    # worst one deduped batch — it never skips unseen data.
    batch = len(seen) // batch_size if batch_size > 0 else 0
    if batch:
        print(f"[{name}] resuming at batch {batch} "
              f"({len(seen)} sources already in ledger).")
    t_start = time.time()
    while total_crops < target_crops and batch < 200:
        remaining_images = max(n_images - batch * batch_size, 1)
        t0 = time.time()
        got = dl(min(batch_size, remaining_images),
                 out_dir=tmp_dir, offset=batch)
        if got == 0:
            print(f"[{name}] batch {batch}: no new files — stopping.")
            break
        n = _crops_from_dir(tmp_dir, staging, name, delete_fits, seen)
        _save_seen(staging, seen)
        total_crops = len(list(staging.glob("*.npy")))
        elapsed = time.time() - t0
        rate = got / elapsed if elapsed > 0 else 0.0
        done_images = min((batch + 1) * batch_size, n_images)
        eta = (n_images - done_images) / rate if rate > 0 else 0.0
        print(f"[{name}] batch {batch}: +{got} FITS -> +{n} crops "
              f"(total {total_crops}) | {elapsed:.0f}s, "
              f"{rate:.1f} img/s, ETA {eta / 60:.0f} min")
        batch += 1

    if delete_fits:
        # Sources are deleted per batch; clean any stragglers.
        for f in tmp_dir.rglob("*.fits"):
            f.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir / "_dl", ignore_errors=True)
    print(f"[{name}] streaming done: {total_crops} crops staged "
          f"in {(time.time() - t_start) / 60:.1f} min.")
    return total_crops


def _install_socket_timeout(seconds: int = 180) -> None:
    """Force a socket timeout on every requests call of this process.

    astroquery passes ``timeout=None`` to ``requests``, so a stalled
    archive connection would block forever and silently freeze the
    campaign (observed live). With a real timeout a stalled socket
    raises and the downloader's error handling moves on to the next
    file. Long streaming downloads are unaffected: requests' timeout is
    per-read, not total.
    """
    import requests

    original = requests.Session.request

    def patched(self, *args, **kwargs):
        kwargs.setdefault("timeout", seconds)
        return original(self, *args, **kwargs)

    requests.Session.request = patched  # type: ignore[method-assign]


def stream_dataset(per_class: int,
                   batch_size: int = 10,
                   telescopes: list[str] | None = None,
                   delete_fits: bool = True,
                   staging_root: Path | None = None) -> dict[str, int]:
    """Stream-crop every requested class and return staged-crop counts.

    Args:
        per_class: number of source images per telescope.
        batch_size: FITS files per download batch (disk peak ~ batch size).
        telescopes: subset to stream; defaults to all :data:`STREAMABLE`.
        delete_fits: delete source FITS after cropping (low-disk mode).
        staging_root: optional override of the crops staging root (tests).

    Returns:
        Map ``{class_name: staged_crops}``.
    """
    _install_socket_timeout()
    targets = telescopes if telescopes is not None else list(STREAMABLE)
    results: dict[str, int] = {}
    for name in targets:
        if name not in STREAMABLE:
            print(f"[{name}] not streamable (Euclid frames are ~7 GB; "
                  f"Roman/Rubin/Gaia have no public data) — skipped.")
            results[name] = 0
            continue
        print(f"\n=== Streaming {name}: {per_class} images -> crops ===")
        try:
            results[name] = _stream_class(name, per_class, batch_size,
                                          delete_fits, staging_root)
        except Exception as exc:  # noqa: BLE001 — keep other classes going
            print(f"[{name}] FAILED: {exc}")
            results[name] = 0
    return results


def main() -> None:
    """CLI entry point for the low-disk streaming pipeline."""
    parser = argparse.ArgumentParser(
        description="Stream-download -> crop 64x64 -> delete FITS "
                    "(peak disk ~1 batch, not the full archive)")
    parser.add_argument("--per-class", type=int, default=300,
                        help="source images per telescope (default 300)")
    parser.add_argument("--telescope", choices=STREAMABLE, default=None,
                        help="stream a single archive instead of all")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="FITS files per download batch (default 10)")
    parser.add_argument("--keep-fits", action="store_true",
                        help="do not delete source FITS after cropping")
    parser.add_argument("--weakest-first", action="store_true",
                        help="stream spitzer/wise before hubble/jwst "
                             "(prioritize the poorest classes)")
    args = parser.parse_args()
    telescopes = ([args.telescope] if args.telescope
                  else None)
    if telescopes is None and args.weakest_first:
        telescopes = ["spitzer", "wise", "hubble", "jwst"]
    stream_dataset(per_class=args.per_class,
                   batch_size=args.batch_size,
                   telescopes=telescopes,
                   delete_fits=not args.keep_fits)


if __name__ == "__main__":
    main()
