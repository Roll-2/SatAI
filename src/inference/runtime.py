"""Predictor factory: CoreML first on Apple Silicon, PyTorch otherwise.

The desktop app calls :func:`create_predictor` with its defaults, so the
CoreML/Neural-Engine copy is the *main* inference path. PyTorch remains the
fallback when:

* the machine is not an Apple Silicon Mac (or macOS < 13);
* ``Models/AstroNetV2.mlpackage`` is missing;
* the CoreML package is older than the PyTorch checkpoint (stale export);
* CoreML fails to load for any reason.

The CLI can force either backend for comparison/debugging.
"""

from __future__ import annotations

from pathlib import Path

from Src.inference.backends import (
    CoreMLBackend,
    TorchBackend,
    coreml_runtime_available,
)
from Src.inference.predictor import Predictor, load_model
from Src.utils.constants import BEST_MODEL_PATH, CORE_ML_MODEL_PATH

__all__ = ["create_predictor", "coreml_usable", "backend_report"]


def coreml_usable(model_path: str | Path | None = None) -> bool:
    """True when the CoreML copy should be preferred over PyTorch.

    A checkpoint newer than the CoreML package means the package is stale;
    in that case we deliberately use the freshly trained PyTorch model.
    """
    checkpoint = Path(model_path or BEST_MODEL_PATH)
    if not coreml_runtime_available(CORE_ML_MODEL_PATH):
        return False
    if checkpoint.resolve() != Path(BEST_MODEL_PATH).resolve():
        return False
    if not checkpoint.exists():
        return False
    if CORE_ML_MODEL_PATH.stat().st_mtime < checkpoint.stat().st_mtime:
        return False
    return True


def create_predictor(prefer_coreml: bool = True,
                     model_path: str | Path | None = None,
                     device=None) -> tuple[Predictor, str]:
    """Build a predictor and return ``(predictor, backend_name)``.

    Args:
        prefer_coreml: try the CoreML/ANE copy first (default).
        model_path: PyTorch checkpoint for the fallback path.
        device: optional PyTorch device override.

    Returns:
        ``("coreml")`` when the Neural-Engine package was used, otherwise
        ``("pytorch")``.
    """
    checkpoint = Path(model_path or BEST_MODEL_PATH)

    if prefer_coreml and coreml_usable(checkpoint):
        try:
            return Predictor(backend=CoreMLBackend(CORE_ML_MODEL_PATH)), "coreml"
        except Exception:          # noqa: BLE001 — fall back transparently
            pass

    model = load_model(checkpoint, device=device)
    return Predictor(model, device=device), "pytorch"


def backend_report(model_path: str | Path | None = None) -> dict:
    """Return diagnostics used by tests/`--backend` CLI output."""
    checkpoint = Path(model_path or BEST_MODEL_PATH)
    runtime = coreml_runtime_available(CORE_ML_MODEL_PATH)
    usable = coreml_usable(checkpoint)
    reason = ""
    if not runtime:
        reason = "CoreML runtime or AstroNetV2.mlpackage unavailable"
    elif checkpoint.resolve() != Path(BEST_MODEL_PATH).resolve():
        reason = "custom checkpoint: CoreML package belongs to the default model"
    elif CORE_ML_MODEL_PATH.stat().st_mtime < checkpoint.stat().st_mtime:
        reason = "CoreML package is older than the PyTorch checkpoint"
    return {
        "coreml_runtime": runtime,
        "coreml_usable": usable,
        "coreml_model": str(CORE_ML_MODEL_PATH),
        "checkpoint": str(checkpoint),
        "reason": reason,
    }
