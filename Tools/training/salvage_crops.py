"""Salvataggio dei crop gia' generati (decisione 2026-09-13: NON ripartire da zero).

La diagnosi precedente doveva una correzione: contando l'FWHM sull'intero
crop 64x64 (invece che sulla finestra centrale 21x21) i campi affollati
gonfiavano la stima e facevano sembrare junk il 92-97% dei crop. La
misura corretta dice che **~2.200 dei 4.120 crop sono stelle vere** e il
probe lineare/CNN raggiunge 47-53% di validation contro 25% del caso
(4 classi): il segnale e' reale e il modello potra' appoggiarvisi.

Questo script copia i crop che passano i gate QC (SNR >= 10 sigma,
FWHM finestra 1.5-14 px) da ``Data/crops/staging_quarantine/<classe>/``
a ``Data/crops/staging/<classe>/`` e sposta i restanti nella classe
"Altro" (sono oggetti estesi reali: esattamente il 20% galassie della
spec 5.2, senza bisogno di download).

Uso::

    .venv/bin/python Tools/training/salvage_crops.py            # default: tutto
    .venv/bin/python Tools/training/salvage_crops.py --dry-run  # solo conteggi
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Src.utils.constants import CROPS_STAGING_DIR  # noqa: E402

QUARANTINE = CROPS_STAGING_DIR.parent / "staging_quarantine"
#: FWHM accettato sulla finestra centrale (vedi constants.CROP_FWHM_*).
FWHM_MIN, FWHM_MAX = 1.5, 14.0
#: SNR minimo del picco centrale (vedi constants.MIN_CROP_SNR).
SNR_MIN = 10.0


def window_fwhm(img: np.ndarray) -> float | None:
    """FWHM della sorgente centrata, misurato sulla finestra 21x21.

    La versione full-crop gonfiava la stima nei campi affollati: qui il
    calcolo e' limitato al vicinato del picco (centro del crop).
    """
    from astropy.stats import sigma_clipped_stats

    c = img.shape[0] // 2
    half = 10
    win = img[c - half:c + half + 1, c - half:c + half + 1]
    _, med, std = sigma_clipped_stats(win, sigma=3.0)
    pw = float(win.max())
    if pw <= float(med) + float(std):
        return None
    mask = win >= med + 0.5 * (pw - med)
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 1.0
    return 2.0 * float(np.sqrt(
        ((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))


def crop_is_star(img: np.ndarray) -> bool:
    """Gate QC del salvataggio: stella compatta e significativa."""
    from astropy.stats import sigma_clipped_stats

    c = img.shape[0] // 2
    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    snr = (float(img[c, c]) - float(med)) / (float(std) + 1e-8)
    if snr < SNR_MIN:
        return False
    fwhm = window_fwhm(img)
    return fwhm is not None and FWHM_MIN <= fwhm <= FWHM_MAX


def salvage(dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Move star crops back to staging, extended ones to Altro."""
    stats: dict[str, dict[str, int]] = {}
    altro_dir = CROPS_STAGING_DIR / "altro"
    n_altro = 0
    for tel in ["hubble", "jwst", "spitzer", "wise"]:
        src_dir = QUARANTINE / tel
        dst_dir = CROPS_STAGING_DIR / tel
        ok = ext = 0
        for f in sorted(src_dir.glob("*.npy")):
            try:
                img = np.load(f).astype(np.float32)
            except Exception:  # noqa: BLE001 — crop corrotto: lascia dov'e'
                continue
            if img.shape != (64, 64):
                continue
            if crop_is_star(img):
                ok += 1
                if not dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst_dir / f.name)
            else:
                ext += 1
                if not dry_run:
                    altro_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, altro_dir / f"alt_real_{tel}_{f.name}")
                    n_altro += 1
        stats[tel] = {"stelle": ok, "estesi": ext}
        print(f"[{tel}] stelle={ok} estesi->Altro={ext}")
    if not dry_run and n_altro:
        print(f"[altro] +{n_altro} oggetti estesi reali")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="conta senza copiare")
    # Accettato per simmetria con streaming --weakest-first: il gate QC del
    # salvage è indipendente dall'ordine, quindi il flag qui è un no-op.
    parser.add_argument("--weakest-first", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    salvage(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
