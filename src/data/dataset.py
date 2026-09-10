"""PyTorch dataset for PSF crops (Tasks 5.3, 6 and 8.2).

``PSFDataset`` loads individual ``.npy`` crops from
``data/crops/<split>/<class_folder>/`` and applies the per-class
augmentation policy from :mod:`src.data.augmentation` on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import Augmenter
from src.utils.constants import (
    CLASS_FOLDERS,
    CROP_SIZE,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

__all__ = ["PSFDataset", "load_split", "compute_class_weights", "SPLIT_DIRS"]

SPLIT_DIRS: dict[str, Path] = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}


def load_split(split: str = "train") -> tuple[list[Path], list[int]]:
    """List all crop files and labels for one split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``(paths, labels)`` where labels are class indices 0-8.
    """
    base = SPLIT_DIRS[split]
    paths: list[Path] = []
    labels: list[int] = []
    for class_id, folder in enumerate(CLASS_FOLDERS):
        class_dir = base / folder
        if not class_dir.exists():
            continue
        found = sorted(class_dir.glob("*.npy"))
        paths.extend(found)
        labels.extend([class_id] * len(found))
    return paths, labels


class PSFDataset(Dataset):
    """Dataset of 64x64 grayscale PSF crops with per-class augmentation."""

    def __init__(self,
                 split: str = "train",
                 augment_enabled: bool | None = None,
                 paths: list[Path] | None = None,
                 labels: list[int] | None = None,
                 augment_seed: int | None = None):
        """Build the dataset.

        Args:
            split: split name, used when ``paths`` is not given.
            augment_enabled: force augmentation on/off; defaults to
                ``True`` for train and ``False`` for val/test.
            paths: explicit crop paths (overrides ``split``).
            labels: explicit labels aligned with ``paths``.
            augment_seed: seed for the augmenter RNG.
        """
        if paths is None or labels is None:
            paths, labels = load_split(split)
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = paths
        self.labels = labels
        if augment_enabled is None:
            augment_enabled = split == "train"
        self.augmenter = Augmenter(enabled=augment_enabled, seed=augment_seed)

    def __len__(self) -> int:
        """Number of crops in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image_tensor, label)`` for one crop.

        The image is a float32 tensor of shape ``(1, 64, 64)`` in [0, 1].
        """
        crop = np.load(self.paths[idx]).astype(np.float32)
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            crop = np.resize(crop, (CROP_SIZE, CROP_SIZE))
        crop = self.augmenter(crop, self.labels[idx])
        tensor = torch.from_numpy(crop).unsqueeze(0)   # (1, 64, 64)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


def compute_class_weights(labels: list[int],
                          num_classes: int = 9) -> torch.Tensor:
    """Compute inverse-frequency class weights (Task 8.2).

    Weights are normalized to mean 1 so the loss scale stays comparable
    across datasets. Classes with zero samples get the maximum weight.

    Args:
        labels: list of class indices from the training split.
        num_classes: total number of classes (9).

    Returns:
        float32 tensor of shape ``(num_classes,)`` for CrossEntropyLoss.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] == 0:
            weights[c] = counts.max() if counts.max() > 0 else 1.0
        else:
            weights[c] = total / (num_classes * counts[c])
    weights /= max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)
