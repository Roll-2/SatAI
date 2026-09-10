"""AstroNet-v2 — PSF classifier CNN (Task 7).

Layer-by-layer spec (Task 7.2):

=================  ====================================  ==============
Livello            Tipo                                  Output
=================  ====================================  ==============
Input              1 channel, 64x64                      (1, 64, 64)
Conv1+BN1+Pool1    32 filtri 3x3 same, ReLU, MaxPool2x2  (32, 32, 32)
Conv2+BN2+Pool2    64 filtri 3x3 same, ReLU, MaxPool2x2  (64, 16, 16)
Conv3+BN3+Pool3    128 filtri 3x3 same, ReLU, MaxPool2x2 (128, 8, 8)
Conv4+BN4+Pool4    256 filtri 3x3 same, ReLU, MaxPool2x2 (256, 4, 4)
GAP                Global Average Pooling                (256,)
Dropout1           tasso 0.4                             (256,)
Dense1             128 nodi, ReLU                        (128,)
Dropout2           tasso 0.3                             (128,)
Dense2             9 nodi, Linear                        (9,)
=================  ====================================  ==============

Notes (Task 7.3):
* **no softmax** in the model — CrossEntropyLoss applies it internally;
* BatchNorm is mandatory after every Conv and **before** the ReLU (the
  explicit ordering of spec 7.3, which supersedes the table's column
  order);
* Xavier/Glorot uniform initialization for all Conv and Linear layers;
* no transfer learning, no giant dense layers before GAP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.constants import NUM_CLASSES

__all__ = ["AstroNetV2", "build_model"]


class ConvBlock(nn.Sequential):
    """Conv2d(3x3, same) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)."""

    def __init__(self, in_ch: int, out_ch: int):
        """Build the block.

        Args:
            in_ch: input channels.
            out_ch: number of 3x3 filters.
        """
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class AstroNetV2(nn.Module):
    """AstroNet-v2: ~0.42 M parameters (~1.7 MB in float32)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        """Build the network.

        Args:
            num_classes: number of output logits (9 per spec).
        """
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),      # -> (32, 32, 32)
            ConvBlock(32, 64),     # -> (64, 16, 16)
            ConvBlock(64, 128),    # -> (128, 8, 8)
            ConvBlock(128, 256),   # -> (256, 4, 4)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)      # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),                   # Dropout1
            nn.Linear(256, 128),                 # Dense1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),                   # Dropout2
            nn.Linear(128, num_classes),         # Dense2 (logits, no softmax)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier/Glorot uniform initialization (Task 7.3)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 1, 64, 64)`` grayscale batch in [0, 1].

        Returns:
            ``(B, 9)`` raw logits (no softmax).
        """
        x = self.features(x)
        x = self.gap(x).flatten(1)              # (B, 256)
        return self.classifier(x)               # (B, 9) logits


def build_model(device: torch.device | str = "cpu",
                num_classes: int = NUM_CLASSES) -> AstroNetV2:
    """Create the model, initialize it and move it to ``device``.

    Args:
        device: target device (mps / cuda / cpu).
        num_classes: number of classes.

    Returns:
        Ready-to-train :class:`AstroNetV2` instance.
    """
    model = AstroNetV2(num_classes=num_classes)
    return model.to(device)
