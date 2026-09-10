"""Multi-crop inference pipeline (Task 10).

Flow: load (FITS or PNG/JPG) -> anti-saturation -> star/blob finding ->
top-10 64x64 crops -> single batched forward -> 10x9 probability matrix
-> arithmetic mean over axis 0 -> decision.

Decision rules (spec 10.1):
* ``max(mean_probs) < 0.55``  -> "Non determinabile";
* ``argmax == 8``             -> "Telescopio sconosciuto";
* otherwise                   -> the telescope display name.

Error handling (spec 10.2): fewer than 3 valid crops, image smaller than
64x64 and corrupt files each produce a dedicated status message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.preprocessing.anti_saturator import desaturate
from src.preprocessing.crop_generator import extract_crops
from src.preprocessing.loader import load_image
from src.preprocessing.star_finder import find_stars_png
from src.utils.constants import (
    ALTRO_CLASS_ID,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CROP_SIZE,
    DISPLAY_NAMES,
    MIN_CROPS_FOR_INFERENCE,
    TOP_CROPS_FOR_INFERENCE,
)
from src.utils.device import get_device

__all__ = ["PREDICTION_STATUS", "PredictionResult", "Predictor", "load_model"]

#: Status messages required by spec 10.1 / 10.2.
PREDICTION_STATUS = {
    "ok": "ok",
    "no_determinabile": "Non determinabile",
    "telescopio_sconosciuto": "Telescopio sconosciuto",
    "insufficiente": "Immagine insufficiente per l'analisi",
    "troppo_piccola": "Immagine troppo piccola",
    "errore_lettura": "Errore di lettura del file",
}


class PredictionResult:
    """Container for one multi-crop prediction (Task 10.1)."""

    def __init__(self, status: str, class_id: int | None = None,
                 confidence: float = 0.0, probabilities: np.ndarray | None = None,
                 n_crops: int = 0, per_crop_probs: np.ndarray | None = None,
                 status_message: str | None = None):
        """Store the prediction outcome.

        Args:
            status: one of the :data:`PREDICTION_STATUS` keys.
            class_id: predicted class index (None on error states).
            confidence: max mean probability.
            probabilities: mean probability vector (9,).
            n_crops: number of crops actually analyzed.
            per_crop_probs: (n_crops, 9) matrix for ``--verbose``.
            status_message: human-readable status line.
        """
        self.status = status
        self.class_id = class_id
        self.confidence = confidence
        self.probabilities = probabilities
        self.n_crops = n_crops
        self.per_crop_probs = per_crop_probs
        self.status_message = status_message or PREDICTION_STATUS.get(
            status, status)

    @property
    def is_success(self) -> bool:
        """True when a telescope class was actually decided."""
        return self.status == "ok"


def load_model(model_path,
               device: torch.device | None = None) -> torch.nn.Module:
    """Load a trained AstroNet-v2 checkpoint for inference.

    Args:
        model_path: path to ``astro_net_v2_best.pt`` (or ``_last``).
        device: target device (defaults to the auto-selected one).

    Returns:
        Model in eval mode.
    """
    from src.models.psf_cnn import build_model

    device = device or get_device()
    model = build_model(device=device)
    checkpoint = torch.load(model_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class Predictor:
    """Multi-crop predictor over a trained AstroNet-v2 model (Task 10)."""

    def __init__(self, model: torch.nn.Module,
                 device: torch.device | None = None):
        """Wrap a model for multi-crop inference.

        Args:
            model: trained AstroNet-v2 (see :func:`load_model`).
            device: execution device (defaults to auto-selection).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_path: str, mode: str | None = None,
                save_crops_dir: str | None = None) -> PredictionResult:
        """Run the complete Task 10 pipeline on one image.

        Args:
            image_path: FITS or PNG/JPG path.
            mode: ``"fits"`` or ``"png"`` (default: from extension).
            save_crops_dir: if set, the analyzed crops are saved here
                for manual inspection (``--save-crops``).

        Returns:
            :class:`PredictionResult` with the aggregated decision.
        """
        try:
            img = load_image(image_path, mode=mode)
        except Exception:  # noqa: BLE001 — corrupt/unreadable file
            return PredictionResult("errore_lettura")

        # Spec 10.2 — image too small.
        if img.shape[0] < CROP_SIZE or img.shape[1] < CROP_SIZE:
            return PredictionResult("troppo_piccola")

        # Task 3.2 — anti-saturation (mandatory before blob finding).
        img = desaturate(img)

        positions = find_stars_png(img)
        crops = extract_crops(img, positions,
                              max_crops=TOP_CROPS_FOR_INFERENCE)

        # Spec 10.2 — too few valid crops.
        if len(crops) < MIN_CROPS_FOR_INFERENCE:
            return PredictionResult("insufficiente", n_crops=len(crops))

        if save_crops_dir is not None:
            out = Path(save_crops_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(crops):
                np.save(out / f"crop_{i:02d}.npy", crop)

        batch = torch.from_numpy(crops).unsqueeze(1).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()   # (10, 9)
        mean_probs = probs.mean(axis=0)                      # axis 0 (spec)

        return self._decide(mean_probs, probs)

    # ------------------------------------------------------------------
    def _decide(self, mean_probs: np.ndarray,
                per_crop: np.ndarray) -> PredictionResult:
        """Apply the Task 10.1 decision rules to the mean probabilities."""
        top = int(np.argmax(mean_probs))
        confidence = float(mean_probs[top])

        if confidence < CONFIDENCE_THRESHOLD:
            return PredictionResult("no_determinabile", class_id=top,
                                    confidence=confidence,
                                    probabilities=mean_probs,
                                    n_crops=per_crop.shape[0],
                                    per_crop_probs=per_crop)
        if top == ALTRO_CLASS_ID:
            return PredictionResult("telescopio_sconosciuto", class_id=top,
                                    confidence=confidence,
                                    probabilities=mean_probs,
                                    n_crops=per_crop.shape[0],
                                    per_crop_probs=per_crop)
        return PredictionResult("ok", class_id=top, confidence=confidence,
                                probabilities=mean_probs,
                                n_crops=per_crop.shape[0],
                                per_crop_probs=per_crop)
