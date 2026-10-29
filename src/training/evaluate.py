"""Evaluation on the test set (Task 9).

Produces: top-1 accuracy, the 9x9 confusion matrix (PNG + text) and
per-class precision / recall / F1. The critical confusions of spec 9.2
(Roman vs Euclid, Hubble vs Rubin, class 8 vs Hubble) trigger targeted
remediation hints.

Usage (from the project root)::

    python -m Src.training.evaluate
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Src.data.dataset import PSFDataset
from Src.models.psf_cnn import build_model
from Src.utils.constants import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_WORKERS,
    PNG_SAMPLES_DIR,
    RAND_SEED,
)
from Src.utils.device import configure_hardware, dataloader_kwargs

__all__ = ["evaluate", "classification_report"]

# Spec 9.2 critical pairs: (class_a, class_b, threshold, remediation)
CRITICAL_PAIRS = [
    (2, 3, 0.15,
     "Roman vs Euclid > 15%: aumentare i filtri in Conv3 e Conv4."),
    (0, 4, 0.15,
     "Hubble vs Rubin confusi: aumentare il dropout."),
    (8, 0, 0.15,
     "Classe 8 confusa con Hubble: aggiungere piu' campioni di rumore puro."),
]


def classification_report(confusion: np.ndarray) -> str:
    """Format the per-class precision/recall/F1 report (Task 9.1).

    Args:
        confusion: 9x9 confusion matrix (rows = true, cols = predicted).

    Returns:
        Formatted report string.
    """
    lines = [f"{'class':<10} {'precision':>9} {'recall':>9} {'f1':>9} "
             f"{'support':>8}"]
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        lines.append(f"{CLASS_NAMES[c]:<10} {precision:>9.3f} {recall:>9.3f} "
                     f"{f1:>9.3f} {int(confusion[c].sum()):>8d}")
    return "\n".join(lines)


def _save_confusion_png(confusion: np.ndarray, out_path) -> None:
    """Save the confusion matrix as a PNG heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("AstroNet-v2 — Confusion Matrix (test set)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2
                    else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(model_path=BEST_MODEL_PATH,
             split: str = "test",
             batch_size: int | None = None) -> dict:
    """Evaluate the saved model and print/save all Task 9 metrics.

    Args:
        model_path: checkpoint produced by :func:`Src.training.train.train`.
        split: dataset split to evaluate (default ``test``).
        batch_size: override the default batch size.

    Returns:
        Dict with ``accuracy``, ``confusion`` (9x9 numpy array) and the
        text ``report``.
    """
    device = configure_hardware()
    torch.manual_seed(RAND_SEED)

    dataset = PSFDataset(split=split, augment_enabled=False)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No crops found for split '{split}'. "
            "Run 'python -m Src.data.build_dataset' first.")

    kwargs = dataloader_kwargs(NUM_WORKERS)
    if len(dataset) < 2 * kwargs["batch_size"]:
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
    kwargs["batch_size"] = batch_size or kwargs["batch_size"]
    loader = DataLoader(dataset, shuffle=False, **kwargs)

    model = build_model(device=device)
    checkpoint = torch.load(model_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    correct, seen = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    accuracy = correct / max(seen, 1)
    report = classification_report(confusion)

    print(f"\n=== ASTRO-NET v2.0 — Test evaluation ({split}) ===")
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%  ({correct}/{seen})")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "          " + " ".join(f"{n[:8]:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(confusion):
        print(f"{CLASS_NAMES[i]:>9} " + " ".join(f"{v:>8d}" for v in row))
    print("\n" + report)

    # Spec 9.2 — critical confusion analysis.
    print("\nAnalisi confusioni critiche (soglia 15%):")
    for a, b, threshold, remediation in CRITICAL_PAIRS:
        row_sum = confusion[a].sum()
        rate = confusion[a, b] / row_sum if row_sum else 0.0
        status = "OK" if rate <= threshold else "SUPERATA"
        print(f"  {CLASS_NAMES[a]} -> {CLASS_NAMES[b]}: {rate * 100:.1f}% "
              f"[{status}]")
        if rate > threshold:
            print(f"    -> {remediation}")

    # Persist artifacts.
    PNG_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PNG_SAMPLES_DIR / "confusion_matrix.png"
    _save_confusion_png(confusion, png_path)
    txt_path = BEST_MODEL_PATH.parent / "evaluation_report.txt"
    txt_path.write_text(
        f"Top-1 Accuracy: {accuracy * 100:.2f}% ({correct}/{seen})\n\n"
        f"Confusion matrix:\n{confusion}\n\n{report}\n")
    print(f"\nSaved: {png_path}\nSaved: {txt_path}")

    return {"accuracy": accuracy, "confusion": confusion, "report": report}


if __name__ == "__main__":
    evaluate()
