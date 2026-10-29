"""JS bridge: the only surface the UI can talk to.

Exposed to JavaScript as ``window.pywebview.api``. Every method returns a
plain dict of JSON-native types (str/bool/float/int/list) and never raises —
failures come back as ``{"ok": False, "message": ...}`` so the UI can always
render them.

Inference backend
-----------------
On an Apple Silicon Mac the app loads ``Models/AstroNetV2.mlpackage`` and
lets CoreML schedule the network on the Neural Engine (CPU_AND_NE). The
PyTorch checkpoint ``Models/astro_net_v2_best.pt`` is the fallback for
non-Apple hardware, older macOS, or a stale/missing CoreML export. The
choice is made once, lazily, on the first photo.

Memory policy
-------------
After each analysis we drop every heavy artifact (crop tensors, MPS cache,
previous uploads) but keep the last *result* (~3 KB of probabilities) and
the last uploaded *path* (a string). Those two are required by the
"Perché proprio…?" and "Cosa vedo nella foto?" chips; deleting them made
both chips unusable. Only the current photo remains on disk, never a
growing upload directory. The progress log is removed after every run.
"""

from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path

import numpy as np

from Src.inference.runtime import coreml_usable, create_predictor
from Src.utils.constants import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DISPLAY_NAMES,
    PROJECT_ROOT,
    UI_UPLOADS_DIR,
)
from . import settings

#: Reject uploads above ~150 MB (base64-inflated) — the UI targets photos.
MAX_UPLOAD_B64_BYTES = 200_000_000

#: Phases reported to the UI progress log, in order.
SAT_PHASES = ("load_model", "star_finder", "neural_net", "done")

#: The two user-approved sentences (and nothing else).
SENTENCE_OK = "La foto che mi hai mostrato è stata scattata da {name}."
SENTENCE_NO = ("Mi dispiace, la foto che mi hai mandato non sembra scattata "
               "da nessuno dei satelliti nel mio database.")

_PROGRESS_LOG = Path("/tmp/satai_ui_progress.log")


def _report_progress(phase: str) -> None:
    """Append one phase to the progress log (best-effort, never raises)."""
    try:
        with open(_PROGRESS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "phase": phase}) + "\n")
    except OSError:
        pass


def _remove_file(path: Path) -> None:
    """Unlink one file, ignoring races/permission errors."""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


class AppBridge:
    """Methods exposed to the UI via ``window.pywebview.api``."""

    def __init__(self, model_path: str | Path | None = None,
                 prefer_coreml: bool = True,
                 upload_dir: str | Path | None = None) -> None:
        """Start with no model: it loads lazily on the first photo.

        Args:
            model_path: override the PyTorch checkpoint (tests).
            prefer_coreml: use the Neural-Engine copy when available.
            upload_dir: override the upload folder (tests).
        """
        self._model_path = Path(model_path) if model_path else BEST_MODEL_PATH
        self._prefer_coreml = prefer_coreml
        self._upload_dir = Path(upload_dir) if upload_dir else UI_UPLOADS_DIR
        self._predictor = None
        self._backend = "unloaded"
        self._lock = threading.Lock()   # one inference at a time
        self._last_result = None        # last PredictionResult (~3 KB)
        self._last_image_path: str | None = None
        self._visual_window = None      # optional 3D viewer window
        # Remove stale uploads from a previous session (no result to keep).
        self._cleanup_uploads(keep=None)

    # ------------------------------------------------------------------
    # Upload receiver: the UI sends the dropped file as base64
    # ------------------------------------------------------------------
    def save_upload(self, name: str, data_b64: str) -> dict:
        """Persist a photo sent from the UI and return its server path.

        HTML drag-and-drop gives the JS side File objects without paths,
        so the UI reads the bytes and ships them here as base64.
        """
        try:
            import base64
            import uuid

            if not data_b64 or len(data_b64) > MAX_UPLOAD_B64_BYTES:
                return {"ok": False,
                        "message": "File troppo grande (max ~150 MB)."}
            safe_name = Path(name or "photo").name or "photo"
            self._upload_dir.mkdir(parents=True, exist_ok=True)
            path = (self._upload_dir
                    / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}")
            path.write_bytes(base64.b64decode(data_b64))
            return {"ok": True, "path": str(path)}
        except Exception as exc:              # noqa: BLE001
            return {"ok": False, "message": f"Salvataggio fallito: {exc}"}

    # ------------------------------------------------------------------
    # Optional 3D viewer, in its own window on the left
    # ------------------------------------------------------------------
    def open_visual(self) -> dict:
        """Open ``Visuals/index.html`` in a dedicated window on the left.

        The SatAi window is pinned to the right edge of the screen. This
        second pywebview window occupies the free space to its left, so the
        video never covers the classifier UI. Creating it lazily keeps the
        app's initial memory footprint unchanged (no hidden WebView at
        startup).
        """
        try:
            import webview

            page = PROJECT_ROOT / "Visuals" / "index.html"
            if not page.exists():
                return {"ok": False,
                        "message": "Visualizzatore non trovato: "
                                   f"{page}"}
            if self._visual_window is not None:
                try:
                    self._visual_window.show()
                    self._visual_window.restore()
                    return {"ok": True, "reused": True}
                except Exception:          # noqa: BLE001 — recreate below
                    self._visual_window = None

            x, y, width, height = self._left_viewer_bounds()
            window = webview.create_window(
                title="SatAi · Visual",
                # A file:// URI is required for a child window created after
                # start; a bare path loads a blank WebView on macOS.
                url=page.as_uri(),
                width=width,
                height=height,
                x=x,
                y=y,
                resizable=True,
                min_size=(700, 560),
                background_color="#05070d",
                text_select=True,
            )
            if window is None:
                return {"ok": False,
                        "message": "Impossibile creare la finestra visuale."}
            self._visual_window = window
            window.events.closed += self._on_visual_closed
            return {"ok": True, "reused": False}
        except Exception as exc:              # noqa: BLE001
            return {"ok": False,
                    "message": f"Visual non disponibile: {exc}"}

    def _on_visual_closed(self) -> None:
        """Forget the closed viewer so the next click creates a fresh one."""
        self._visual_window = None

    @staticmethod
    def _left_viewer_bounds() -> tuple[int, int, int, int]:
        """Return (x, y, width, height) for the left-hand viewer window."""
        try:
            from AppKit import NSScreen

            screen = NSScreen.mainScreen().frame()
            sw, sh = int(screen.size.width), int(screen.size.height)
        except Exception:                     # noqa: BLE001
            sw, sh = 1440, 900

        app_left = sw - 430 - 24              # SatAi's left edge
        width = max(700, min(1110, app_left - 48))
        height = max(600, min(880, sh - 80))
        x = 24
        y = max(24, (sh - height) // 2)
        return x, y, width, height

    # ------------------------------------------------------------------
    # Backend diagnostics (header badge in the UI)
    # ------------------------------------------------------------------
    def engine(self) -> dict:
        """Return which engine the app will use (without loading weights)."""
        if self._predictor is not None:
            return {"ok": True, "backend": self._backend,
                    "label": self._predictor.backend_label}
        use_coreml = (self._prefer_coreml
                      and coreml_usable(self._model_path))
        label = ("CoreML · Apple Neural Engine" if use_coreml
                 else "PyTorch · fallback")
        return {"ok": True,
                "backend": "coreml" if use_coreml else "pytorch",
                "label": label}

    # ------------------------------------------------------------------
    # Main entry point: analyze one photo
    # ------------------------------------------------------------------
    def analyze(self, image_path: str) -> dict:
        """Classify one image and return the UI verdict.

        Returns:
            Dict with keys ``ok``, ``identified``, ``sentence``,
            ``confidence`` (0..1) and ``top`` (list of up to three
            ``{"name", "pct"}`` rows for the probability bars).
        """
        path = (image_path or "").strip()
        if not path or not Path(path).exists():
            return {"ok": False, "identified": False, "sentence": "",
                    "confidence": 0.0, "top": [],
                    "message": "File non trovato."}

        with self._lock:                      # one analysis at a time
            try:
                self._ensure_model()
            except Exception as exc:          # noqa: BLE001
                self._cleanup_uploads(keep=None)
                return {"ok": False, "identified": False, "sentence": "",
                        "confidence": 0.0, "top": [],
                        "message": f"Errore caricamento modello: {exc}"}

            try:
                _report_progress("star_finder")
                result = self._predictor.predict(path)
                _report_progress("neural_net")
                _report_progress("done")
            except Exception as exc:          # noqa: BLE001
                self._last_result = None
                self._last_image_path = None
                self._cleanup_uploads(keep=None)
                self._release_memory()
                return {"ok": False, "identified": False, "sentence": "",
                        "confidence": 0.0, "top": [],
                        "message": f"Errore durante l'analisi: {exc}"}

            # Keep the tiny state the UI chips need; drop everything heavy.
            # The current photo stays on disk (identify_object reads it), all
            # older uploads are removed so the folder never accumulates.
            self._last_result = result
            self._last_image_path = path
            self._cleanup_uploads(keep=Path(path))
            self._release_memory()

        return self._result_to_payload(result)

    # ------------------------------------------------------------------
    # "Perché proprio [satellite]?" — computed from the real prediction
    # ------------------------------------------------------------------
    def why(self) -> dict:
        """Return the truthful explanation for the last classification."""
        result = self._last_result
        if result is None:
            return {"ok": False,
                    "message": "Analizza prima una foto."}
        engine_note = (f"Motore di inferenza: "
                       f"{self._predictor.backend_label}."
                       if self._predictor is not None else "")
        if not result.is_success:
            return {"ok": True,
                    "message": "Questa foto non è stata attribuita a nessun "
                               "satellite del database, quindi non c'è un "
                               "perché da spiegare: prova con una foto di "
                               "una zona di cielo con stelle puntiformi."
                               + (" " + engine_note if engine_note else "")}

        probs = np.asarray(result.probabilities, dtype=float)
        order = np.argsort(probs)[::-1]
        first, second = int(order[0]), int(order[1])
        name = DISPLAY_NAMES[result.class_id]
        n_crops = result.n_crops

        parts = [
            f"Ho analizzato {n_crops} ritagli da 64×64 pixel centrati sulle "
            "stelle più brillanti dell'immagine. Ogni ritaglio mostra la "
            "figura di diffrazione (PSF) prodotta dall'ottica del telescopio.",
            f"Nel voto complessivo la classe '{CLASS_NAMES[first]}' ha "
            f"ottenuto il {probs[first] * 100:.1f}%, mentre la seconda "
            f"classificata è '{CLASS_NAMES[second]}' con il "
            f"{probs[second] * 100:.1f}% (margine del "
            f"{(probs[first] - probs[second]) * 100:.1f} punti).",
            engine_note,
        ]
        if result.per_crop_probs is not None:
            votes = int(np.sum(
                result.per_crop_probs.argmax(axis=1) == result.class_id))
            parts.append(f"{votes} ritagli su {n_crops} concordano sulla "
                         f"stessa classe.")
        if result.confidence >= 0.80:
            parts.append("La confidenza è alta: la firma geometrica "
                         "(numero e angolo dei punte di diffrazione) è "
                         " nettamente riconoscibile.")
        elif result.confidence >= 0.65:
            parts.append("La confidenza è buona, anche se qualche tratto "
                         "della PSF è ambiguo.")
        else:
            parts.append("La confidenza è moderata: la decisione è presa "
                         "sopra la soglia, ma i punte sono meno netti del "
                         "solito (immagine compressa o stelle deboli).")
        return {"ok": True, "message": " ".join(p for p in parts if p)}

    # ------------------------------------------------------------------
    # "Cosa vedo nella foto?" — OpenRouter chip (only when a key is set)
    # ------------------------------------------------------------------
    def identify_object(self) -> dict:
        """Send the last photo to OpenRouter and return its description."""
        if self._last_image_path is None:
            return {"ok": False,
                    "message": "Analizza prima una foto."}
        if not Path(self._last_image_path).exists():
            return {"ok": False,
                    "message": "La foto non è più disponibile: analizzala "
                               "di nuovo."}
        from .openrouter_client import OpenRouterError, identify_object
        try:
            text, model_used = identify_object(self._last_image_path)
        except OpenRouterError as exc:
            return {"ok": False, "message": str(exc)}
        except Exception as exc:              # noqa: BLE001
            return {"ok": False,
                    "message": f"Errore inatteso: {exc}"}
        return {"ok": True, "message": text, "model": model_used}

    # ------------------------------------------------------------------
    # OpenRouter key management
    # ------------------------------------------------------------------
    def has_api_key(self) -> bool:
        """Expose key presence so the UI can show/hide the chip."""
        return settings.has_api_key()

    def save_key(self, key: str) -> dict:
        """Persist (or clear with '') the OpenRouter API key locally."""
        settings.set_api_key(key)
        return {"ok": True, "has_key": settings.has_api_key()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load the predictor once; it stays resident for later photos."""
        if self._predictor is not None:
            return
        _report_progress("load_model")
        if not self._model_path.exists() and not (
                self._prefer_coreml and coreml_usable(self._model_path)):
            raise RuntimeError(
                "Modello non ancora addestrato: completa prima il training "
                "(python -m Src.training.train).")
        try:
            self._predictor, self._backend = create_predictor(
                prefer_coreml=self._prefer_coreml,
                model_path=self._model_path)
        except Exception as exc:              # noqa: BLE001
            raise RuntimeError(str(exc)) from exc

    def _cleanup_uploads(self, keep: Path | None) -> None:
        """Delete uploads except ``keep`` (the photo the chips still use)."""
        folder = self._upload_dir
        if not folder.exists():
            return
        keep_resolved = keep.resolve() if keep is not None else None
        try:
            entries = list(folder.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.name == ".gitkeep":
                continue
            try:
                if entry.is_file():
                    if keep_resolved is None or entry.resolve() != keep_resolved:
                        entry.unlink()
            except OSError:
                pass

    def _release_memory(self) -> None:
        """Drop Python garbage and (on PyTorch/MPS) the MPS allocator cache."""
        gc.collect()
        if self._predictor is not None and self._predictor.backend_name == "pytorch":
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:                 # noqa: BLE001 — best effort
                pass
        # Bounded progress log: it is a diagnostic, not state.
        _remove_file(_PROGRESS_LOG)

    def _result_to_payload(self, result) -> dict:
        """Convert a PredictionResult into the JSON dict the UI renders."""
        identified = result.is_success            # status "ok" only
        if identified:
            sentence = SENTENCE_OK.format(
                name=DISPLAY_NAMES[result.class_id])
        else:
            # "Altro", "Non determinabile" and error states all get the
            # single agreed not-in-database sentence.
            sentence = SENTENCE_NO

        top: list[dict] = []
        if result.probabilities is not None:
            probs = np.asarray(result.probabilities, dtype=float)
            for cid in np.argsort(probs)[::-1][:3]:
                top.append({"name": CLASS_NAMES[int(cid)],
                            "pct": round(float(probs[int(cid)]) * 100.0, 1)})

        return {
            "ok": True,
            "identified": identified,
            "sentence": sentence,
            "confidence": round(float(result.confidence) * 100.0, 1)
            if result.probabilities is not None else 0.0,
            "top": top,
            "message": result.status_message or "",
        }
