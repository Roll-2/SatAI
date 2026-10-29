"""Run all archive downloads (Task 2 entry point).

Usage (from the project root)::

    python -m Src.acquisition.run_all                     # all archives
    python -m Src.acquisition.run_all --telescope hubble  # one archive
    python -m Src.acquisition.run_all --altro             # class 8 synthetic
    python -m Src.acquisition.run_all --n 100             # per-archive count
"""

from __future__ import annotations

import argparse

from Src.utils.constants import FITS_RAW_DIR

# (folder, downloader module attribute, missing-data note)
MISSING_NOTE = {
    "roman": "Roman has no public data before launch (2027) — folder left empty.",
    "rubin": "Rubin RSP access is credential-gated until the public release — "
             "drop FITS files manually into Data/fits_raw/rubin/.",
    "gaia": "Gaia publishes catalogs only (no 2-D image products) — drop FITS "
            "files manually into Data/fits_raw/gaia/ if you have them.",
}
DOWNLOADABLE = ["hubble", "jwst", "spitzer", "wise", "euclid"]


def main() -> None:
    """Download FITS data for every requested telescope."""
    parser = argparse.ArgumentParser(description="SatAi data acquisition")
    parser.add_argument("--telescope", choices=DOWNLOADABLE + list(MISSING_NOTE),
                        default=None, help="download a single archive")
    parser.add_argument("--n", type=int, default=200,
                        help="number of FITS files per archive")
    parser.add_argument("--altro", action="store_true",
                        help="generate the synthetic class-8 'Altro' crops")
    parser.add_argument("--altro-n", type=int, default=5000)
    args = parser.parse_args()

    targets = [args.telescope] if args.telescope else \
        DOWNLOADABLE + list(MISSING_NOTE)

    for name in targets:
        if name in MISSING_NOTE:
            print(f"[{name}] {MISSING_NOTE[name]}")
            (FITS_RAW_DIR / name).mkdir(parents=True, exist_ok=True)
            continue
        print(f"\n=== Downloading {name} ({args.n} files) ===")
        try:
            if name == "hubble":
                from Src.acquisition.mast_downloader import download_hubble
                n = download_hubble(args.n)
            elif name == "jwst":
                from Src.acquisition.mast_downloader import download_jwst
                n = download_jwst(args.n)
            elif name == "spitzer":
                from Src.acquisition.irsa_downloader import download_spitzer
                n = download_spitzer(args.n)
            elif name == "wise":
                from Src.acquisition.irsa_downloader import download_wise
                n = download_wise(args.n)
            else:  # euclid
                from Src.acquisition.euclid_downloader import download_euclid
                n = download_euclid(args.n)
            print(f"[{name}] wrote {n} FITS files.")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}")

    if args.altro:
        print(f"\n=== Generating class-8 'Altro' crops ({args.altro_n}) ===")
        from Src.acquisition.galaxy_cutouts import download_galaxies
        from Src.acquisition.altro_generator import generate_altro_crops

        n_gal = max(10, int(args.altro_n * 0.20 // 8))   # enough cutouts
        got = download_galaxies(n_gal)
        print(f"[LegacySurvey] wrote {got} galaxy cutouts.")
        counts = generate_altro_crops(args.altro_n)
        print(f"[Altro] crops generated: {counts}")


if __name__ == "__main__":
    main()
