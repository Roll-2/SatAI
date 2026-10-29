"""Multi-crop inference pipeline (Task 10).

Two-stage flow:
  Stage 1: Fast star/blob finding over the full image (Photutils blob_log
  or DAOStarFinder).  High recall — we want to catch every spike-carrying
  star even if it means a few extra false positives.
  Stage 2: Crop only the detected candidates, run AstroNet-v2 on those
  crops, aggregate probabilities, apply decision rules.

Decision rules (spec 10.1):
* ``max(mean_probs) < 0.55``  ->  "Non determinabile"
* ``argmax == 8``               ->  "Telescopio sconosciuto"
* otherwise                     ->  the telescope display name

Error handling (spec 10.2): fewer than 3 valid crops, image smaller than
64x64 and corrupt files each produce a dedicated status message.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from Src.inference.backends import TorchBackend
from Src.preprocessing.anti_saturator import desaturate
from Src.preprocessing.crop_generator import extract_crops
from Src.preprocessing.loader import load_image
from Src.preprocessing.star_finder import find_stars_fits, find_stars_png
from Src.utils.constants import (
    ALTRO_CLASS_ID,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CROP_SIZE,
    DISPLAY_NAMES,
    INFERENCE_FWHM_MAX,
    INFERENCE_MIN_SNR,
    MIN_CROPS_FOR_INFERENCE,
    OOD_MEDIAN_SNR_MIN,
    TOP_CROPS_FOR_INFERENCE,
)
if TYPE_CHECKING:            # lazy: CoreML inference never imports torch
    import torch

__all__ = ["PREDICTION_STATUS", "PredictionResult", "Predictor", "load_model"]

#: Status messages required by spec 10.1 / 10.2.
PREDICTION_STATUS = {
    "ok": "ok",
    "no_determinabile": "Non determinabile",
    "telescopio_sconosciuto": "Telescopio sconosciuto",
    "insufficiente": "Immagine insufficiente per l'analisi",
    "non_telescopio": "Nessuna sorgente nitida: non sembra una foto "
                      "astronomica da telescopio",
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
               device=None):
    """Load a trained AstroNet-v2 checkpoint for inference.

    Args:
        model_path: path to ``astro_net_v2_best.pt`` (or ``_last``).
        device: target device (defaults to the auto-selected one).

    Returns:
        Model in eval mode.
    """
    import torch

    from Src.models.psf_cnn import build_model
    from Src.utils.device import get_device

    device = device or get_device()
    model = build_model(device=device)
    checkpoint = torch.load(model_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class Predictor:
    """Multi-crop predictor over a trained AstroNet-v2 model (Task 10)."""

    def __init__(self, model=None, device=None, backend=None):
        """Wrap a model (or a ready backend) for multi-crop inference.

        Args:
            model: trained AstroNet-v2 (see :func:`load_model`). Ignored
                when ``backend`` is given.
            device: execution device (defaults to auto-selection).
            backend: optional execution backend exposing
                ``predict_proba(crops)`` (CoreML or PyTorch).
        """
        if backend is None:
            if model is None:
                raise ValueError("pass either a model or a backend")
            backend = TorchBackend(model, device)
        self._backend = backend
        self.model = getattr(backend, "model", None)
        self.device = getattr(backend, "device", None)

    @property
    def backend_name(self) -> str:
        """``"coreml"`` or ``"pytorch"`` for diagnostics."""
        return str(getattr(self._backend, "name", "unknown"))

    @property
    def backend_label(self) -> str:
        """Human-readable backend label (shown by the UI)."""
        return str(getattr(self._backend, "label", self.backend_name))

    # ------------------------------------------------------------------
    def predict(self, image_path: str, mode: str | None = None,
                save_crops_dir: str | None = None) -> PredictionResult:
        """Run the complete Task 10 pipeline on one image.

        Two-stage pipeline:
          1. Fast star/blob finding over the full image (high recall).
          2. Crop only detected candidates, run AstroNet-v2, aggregate.

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

        # Task 3.2 — anti-saturation (mandatory before star finding).
        img = desaturate(img)

        # Stage 1: Fast star finding (high recall).
        # For FITS images, use find_stars_fits with per-telescope parameters.
        # For PNG/JPG, use find_stars_png.
        is_fits = str(image_path).lower().endswith((".fits", ".fit", ".fts"))
        if is_fits:
            positions = find_stars_fits(img, telescope=None)
        else:
            positions = find_stars_png(img)

        # Stage 2: Crop only the detected candidates.
        # Use relaxed inference FWHM max so extended objects reach the model;
        # use reduced SNR floor so faint spikes are not prematurely dropped.
        crops = extract_crops(img, positions,
                              max_crops=TOP_CROPS_FOR_INFERENCE,
                              fwhm_max=INFERENCE_FWHM_MAX,
                              min_snr=INFERENCE_MIN_SNR)

        # Spec 10.2 — too few valid crops.
        if len(crops) < MIN_CROPS_FOR_INFERENCE:
            return PredictionResult("insufficiente", n_crops=len(crops))

        # OOD gate (bug 14/09: photo of a satellite -> "JWST 99.7%").
        # The classifier is trained ONLY on telescope PSF crops: on photos of
        # arbitrary objects it stays confidently wrong. Measured statistic:
        # junk (noise/texture plateaus) has crop-SNR median <= 13 while the
        # real distribution starts at >= 23.7 on every class — 16 splits.
        # The message says WHY, so the UI can show it verbatim.
        from astropy.stats import sigma_clipped_stats
        snrs = []
        for c in crops:
            _, c_med, c_std = sigma_clipped_stats(c, sigma=3.0)
            peak = float(c[CROP_SIZE // 2, CROP_SIZE // 2])
            snrs.append((peak - float(c_med)) / (float(c_std) + 1e-8))
        if float(np.median(snrs)) < OOD_MEDIAN_SNR_MIN:
            ood_msg = "L'immagine non contiene sorgenti puntiformi abbastanza " \
                      "nitide: probabilmente non e' una foto astronomica da " \
                      "telescopio (o e' troppo compressa/sfocata)."
            return PredictionResult(
                "non_telescopio", n_crops=len(crops),
                status_message=ood_msg)

        if save_crops_dir is not None:
            out = Path(save_crops_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(crops):
                np.save(out / f"crop_{i:02d}.npy", crop)

        # CoreML (ANE) or PyTorch, behind one interface.
        probs = self._backend.predict_proba(crops)           # (n_crops, 9)
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