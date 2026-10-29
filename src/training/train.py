"""Training loop (Task 8).

Hyperparameters (Task 8.1): batch 256, Adam lr 0.001 betas (0.9, 0.999)
eps 1e-8 weight decay 1e-5, max 40 epochs, early stopping patience 7,
ReduceLROnPlateau(factor 0.1, patience 5) on val_loss.

Loss (Task 8.2): CrossEntropyLoss with inverse-frequency class weights.

Usage (from the project root)::

    python -m Src.training.train
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Src.data.dataset import PSFDataset, compute_class_weights
from Src.models.psf_cnn import build_model
import gc

from Src.utils.constants import (
    ADAM_BETAS,
    ADAM_EPS,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    LAST_MODEL_PATH,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    NUM_WORKERS,
    RAND_SEED,
    WEIGHT_DECAY,
)
from Src.utils.device import configure_hardware, dataloader_kwargs

__all__ = ["train", "main"]


def _make_dataloaders(device: torch.device,
                      batch_size: int = BATCH_SIZE
                      ) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Build train/val loaders and the inverse-frequency class weights."""
    train_set = PSFDataset(split="train", augment_enabled=True,
                           augment_seed=RAND_SEED)
    val_set = PSFDataset(split="val", augment_enabled=False)

    kwargs = dataloader_kwargs(NUM_WORKERS)
    kwargs["batch_size"] = batch_size
    # Small datasets must not spawn 8 workers with persistent_workers.
    if len(train_set) < 2 * kwargs["batch_size"]:
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False

    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)

    weights = compute_class_weights(train_set.labels, num_classes=9)
    weights = weights.to(device)
    print(f"Dataset: {len(train_set)} train / {len(val_set)} val crops")
    print(f"Class weights: {weights.cpu().numpy().round(3).tolist()}")
    return train_loader, val_loader, weights


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch over a loader; returns (mean_loss, accuracy, n)."""
    model.train() if train else model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "val  ",
                                   leave=False, unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(train):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1), seen


def train(max_epochs: int = MAX_EPOCHS,
          batch_size: int = BATCH_SIZE) -> dict:
    """Run the full training loop (Task 8.3).

    Args:
        max_epochs: epoch cap (spec default 40; lower it for pilot runs).
        batch_size: batch size (spec default 256).

    Returns:
        History dict with ``train_loss``, ``val_loss``, ``val_accuracy``
        lists and the best validation accuracy.
    """
    # Task 0.4 — before any tensor operation.
    device = configure_hardware()
    torch.manual_seed(RAND_SEED)

    train_loader, val_loader, class_weights = _make_dataloaders(
        device, batch_size=batch_size)

    model = build_model(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AstroNet-v2: {n_params:,} parameters on {device}")

    # Task 8.2 — inverse-frequency class weights.
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=ADAM_BETAS,
                                 eps=ADAM_EPS,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [],
                                       "val_accuracy": []}
    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, _ = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        # Task 8.3 — scheduler on val_loss, checkpoint on val_accuracy.
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        "class_names": CLASS_NAMES},
                       BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "class_names": CLASS_NAMES},
                   LAST_MODEL_PATH)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{max_epochs} "
              f"({time.time() - t0:.0f}s) | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {lr_now:.2e} "
              f"{'*' if improved else ''}")

        # Task 8.1 — early stopping.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: no val_accuracy improvement for "
                  f"{EARLY_STOPPING_PATIENCE} epochs.")
            break

        # Libera la cache MPS dopo ogni epoca per evitare accumulo di
        # memoria GPU durante training prolungato (il modello è piccolo,
        # ma i DataLoader con worker possono accumulare buffer temporanei).
        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    print(f"\nBest val_accuracy: {best_val_acc:.4f}")
    print(f"Best model:    {BEST_MODEL_PATH}")
    print(f"Last model:    {LAST_MODEL_PATH}")
    history["best_val_accuracy"] = best_val_acc
    return history


def main() -> None:
    """CLI entry point (supports short pilot runs)."""
    parser = argparse.ArgumentParser(description="AstroNet-v2 training")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help="epoch cap (default 40; use 3 for a pilot)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="batch size (default 256)")
    args = parser.parse_args()
    train(max_epochs=args.max_epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
