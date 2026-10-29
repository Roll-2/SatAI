"""Execution backends for AstroNet-v2 inference.

The preprocessing + multi-crop decision pipeline is shared; only the final
``(N, 64, 64) -> (N, 9)`` probability step changes:

* :class:`CoreMLBackend` — the same trained weights exported to CoreML,
  loaded with ``ComputeUnit.CPU_AND_NE`` so macOS can schedule the network
  on the Apple Neural Engine (ANE).
* :class:`TorchBackend` — the original PyTorch checkpoint on MPS/CUDA/CPU.
  This is the fallback for non-Apple machines and for anything CoreML
  cannot load.

Both expose ``predict_proba(crops)`` returning an ``(N, 9)`` float32 array
of softmax probabilities, so the caller never needs to know the backend.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from Src.utils.constants import (
    CORE_ML_INPUT_NAME,
    CORE_ML_OUTPUT_NAME,
    CROP_SIZE,
    NUM_CLASSES,
)

if TYPE_CHECKING:            # torch is imported lazily: the CoreML path
    import torch             # must not pay its ~200 MB import cost

__all__ = [
    "CoreMLBackend",
    "TorchBackend",
    "coreml_runtime_available",
    "softmax",
]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for one vector of logits."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - float(z.max())
    exp = np.exp(z)
    return (exp / exp.sum()).astype(np.float32)


class TorchBackend:
    """PyTorch fallback backend (MPS on Apple Silicon, CUDA, or CPU)."""

    name = "pytorch"

    def __init__(self, model: "torch.nn.Module",
                 device=None):
        """Move ``model`` to the best device and put it in eval mode."""
        from Src.utils.device import get_device

        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()

    @property
    def label(self) -> str:
        """Human-readable backend name for diagnostics/UI."""
        device = str(self.device)
        if device == "mps":
            return "PyTorch · Metal (MPS)"
        if device == "cuda":
            return "PyTorch · CUDA"
        return "PyTorch · CPU"

    def predict_proba(self, crops: np.ndarray) -> np.ndarray:
        """Return ``(N, 9)`` softmax probabilities for 64x64 crops."""
        import torch

        with torch.no_grad():
            batch = torch.from_numpy(
                np.asarray(crops, dtype=np.float32)).unsqueeze(1).to(self.device)
            logits = self.model(batch)
            return torch.softmax(logits, dim=1).cpu().numpy()


class CoreMLBackend:
    """CoreML backend, configured to use the Apple Neural Engine.

    The exported model has a fixed batch of 1 (the shape CoreML schedules
    most reliably on the ANE), so the ten crops are evaluated in a short
    loop. Each call is sub-millisecond on the M4 Pro.
    """

    name = "coreml"

    def __init__(self, model_path: str | Path,
                 compute_units: str = "CPU_AND_NE"):
        """Load an ``.mlpackage`` and compile it for CPU + Neural Engine.

        Args:
            model_path: path to ``AstroNetV2.mlpackage``.
            compute_units: ``"CPU_AND_NE"`` (default, ANE enabled) or
                ``"ALL"`` (CPU + GPU + ANE, useful for comparison).
        """
        import coremltools as ct

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if compute_units == "ALL":
            units = ct.ComputeUnit.ALL
        else:
            units = ct.ComputeUnit.CPU_AND_NE
        self.compute_units = compute_units
        self._model = ct.models.MLModel(str(self.model_path),
                                        compute_units=units)

    @property
    def label(self) -> str:
        """Human-readable backend name for diagnostics/UI."""
        return "CoreML · Apple Neural Engine"

    def predict_proba(self, crops: np.ndarray) -> np.ndarray:
        """Return ``(N, 9)`` softmax probabilities for 64x64 crops."""
        crops = np.asarray(crops, dtype=np.float32)
        if crops.ndim != 3 or crops.shape[1:] != (CROP_SIZE, CROP_SIZE):
            raise ValueError(
                f"expected (N, {CROP_SIZE}, {CROP_SIZE}) crops, "
                f"got {crops.shape}")
        probs = np.empty((crops.shape[0], NUM_CLASSES), dtype=np.float32)
        for i, crop in enumerate(crops):
            batch = crop.reshape(1, 1, CROP_SIZE, CROP_SIZE)
            raw = self._model.predict({CORE_ML_INPUT_NAME: batch})
            probs[i] = softmax(raw[CORE_ML_OUTPUT_NAME])
        return probs


def coreml_runtime_available(model_path: str | Path) -> bool:
    """True when CoreML can run this model on an Apple Silicon Mac.

    Checks the platform first (no pointless ``coremltools`` import on
    Linux/Windows), then the minimum macOS deployment target used by the
    converter (macOS 13), then that the package exists.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return False
    if major < 13:
        return False
    if not Path(model_path).exists():
        return False
    if importlib.util.find_spec("coremltools") is None:
        return False
    # Do NOT import coremltools here: importing it pulls in torch (~200 MB),
    # and this check also runs when the UI just asks which engine it will use.
    # CoreMLBackend imports it on first real inference; create_predictor()
    # catches any load failure and falls back to PyTorch.
    return True
