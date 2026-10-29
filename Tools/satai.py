"""SatAi CLI (Task 11).

Base command (spec 11.1)::

    python Tools/satai.py --input <path_immagine> --mode <fits|png>

Extra options (spec 11.3):

* ``--save-crops`` — save the analyzed crops for manual inspection;
* ``--verbose`` — print the per-crop probability matrix;
* ``--batch-dir <dir>`` — process a whole folder and write a CSV report;
* ``--backend auto|coreml|pytorch`` — choose the inference engine
  (``auto`` prefers CoreML/Neural Engine on Apple Silicon).

Output block follows spec 11.2 exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow ``python Tools/satai.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Src.utils.constants import BEST_MODEL_PATH, CLASS_NAMES, DISPLAY_NAMES  # noqa: E402

# Reordered per spec 11.2 (Hubble, JWST, Euclid, Altro, ...).
_OUTPUT_ORDER = [0, 1, 3, 4, 5, 6, 7, 2, 8]


def _print_banner() -> None:
    """Print the CLI banner."""
    print("=== ASTRO-NET v2.0 ===")


def _print_result(result, file_label: str, verbose: bool) -> None:
    """Print one prediction in the exact spec 11.2 format."""
    print(f"File: {file_label}")
    print(f"Crop analizzati: {result.n_crops}")

    if not result.is_success:
        print(f"Esito: {result.status_message}")
        return

    name = DISPLAY_NAMES[result.class_id]
    print(f"Telescopio rilevato: {name}")
    print(f"Confidenza: {result.confidence * 100:.1f}%")
    print("Distribuzione probabilita':")
    probs = result.probabilities
    for cid in _OUTPUT_ORDER:
        print(f"  {CLASS_NAMES[cid]:<8} {probs[cid] * 100:>6.1f}%")

    if verbose and result.per_crop_probs is not None:
        print("\nProbabilita' per singolo crop:")
        for i, row in enumerate(result.per_crop_probs):
            top = int(row.argmax())
            print(f"  crop {i:02d}: {CLASS_NAMES[top]:<8} "
                  f"{row[top] * 100:>6.1f}%  "
                  f"[{' '.join(f'{p:.2f}' for p in row)}]")


def _process_one(args, predictor) -> dict:
    """Run the pipeline on one input path and print the result."""
    _print_banner()
    path = Path(args.input)
    result = predictor.predict(str(path), mode=args.mode,
                               save_crops_dir=args.save_crops)
    _print_result(result, path.name, args.verbose)
    row = {
        "file": path.name,
        "status": result.status_message,
        "class_id": "" if result.class_id is None else result.class_id,
        "class_name": "" if result.class_id is None
        else CLASS_NAMES[result.class_id],
        "confidence": f"{result.confidence:.4f}",
        "n_crops": result.n_crops,
    }
    return row


def _process_batch(args, predictor) -> None:
    """Process a whole folder (``--batch-dir``) and write a CSV (Task 11.3)."""
    folder = Path(args.batch_dir)
    if not folder.is_dir():
        print(f"Cartella non trovata: {folder}")
        return

    exts = {".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in exts)
    if not files:
        print(f"Nessuna immagine trovata in {folder}")
        return

    csv_path = folder / "predictions.csv"
    print(f"Elaboro {len(files)} immagini da {folder} -> {csv_path}")

    rows = []
    for f in files:
        _print_banner()
        result = predictor.predict(str(f), save_crops_dir=args.save_crops)
        _print_result(result, f.name, args.verbose)
        print()
        rows.append({
            "file": f.name,
            "status": result.status_message,
            "class_id": "" if result.class_id is None else result.class_id,
            "class_name": "" if result.class_id is None
            else CLASS_NAMES[result.class_id],
            "confidence": f"{result.confidence:.4f}",
            "n_crops": result.n_crops,
        })

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV salvato: {csv_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="SatAi",
        description="AstroNet-v2 — classificatore di telescopi via analisi PSF")
    parser.add_argument("--input", type=str, default=None,
                        help="percorso dell'immagine (FITS o PNG/JPG)")
    parser.add_argument("--mode", choices=["fits", "png"], default=None,
                        help="forza la modalita' di lettura")
    parser.add_argument("--save-crops", type=str, default=None,
                        help="cartella dove salvare i 10 crop analizzati")
    parser.add_argument("--verbose", action="store_true",
                        help="stampa le probabilita' di ogni singolo crop")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="elabora un'intera cartella e produce un CSV")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH),
                        help="checkpoint PyTorch da usare")
    parser.add_argument("--backend", choices=["auto", "coreml", "pytorch"],
                        default="auto",
                        help="motore di inferenza (default: CoreML se "
                             "disponibile, altrimenti PyTorch)")
    args = parser.parse_args()

    if not args.input and not args.batch_dir:
        parser.error("specificare --input <path> oppure --batch-dir <dir>")

    from Src.inference.runtime import backend_report, create_predictor

    report = backend_report(args.model)
    if args.backend == "coreml" and not report["coreml_usable"]:
        print("CoreML non disponibile su questa macchina/export:")
        print(f"  {report['reason'] or 'motivo sconosciuto'}")
        sys.exit(1)
    if not Path(args.model).exists() and not report["coreml_usable"]:
        print(f"Modello non trovato: {args.model}")
        print("Addestra prima il modello: python -m Src.training.train")
        sys.exit(1)

    try:
        predictor, backend = create_predictor(
            prefer_coreml=args.backend != "pytorch",
            model_path=args.model)
    except Exception as exc:                 # noqa: BLE001 — CLI boundary
        print(f"Impossibile caricare il modello: {exc}")
        sys.exit(1)

    if args.backend == "coreml" and backend != "coreml":
        print("CoreML richiesto ma non è stato possibile caricarlo.")
        sys.exit(1)

    print(f"Motore: {predictor.backend_label}")

    if args.batch_dir:
        _process_batch(args, predictor)
    else:
        _process_one(args, predictor)


if __name__ == "__main__":
    main()
