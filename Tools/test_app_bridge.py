"""Headless verification of the desktop-app bridge (no window opened).

Covers both inference paths and the cleanup contract:

* PyTorch fallback — a temporary untrained checkpoint is used, so the real
  trained model is never touched.
* CoreML/Neural Engine — the real exported package is exercised on a real
  validation image when available.
* ``why()`` and ``identify_object()`` must keep working after ``analyze()``.
  A previous cleanup bug nulled their state and silently disabled both UI
  chips; this test now guards that regression.
* Uploads are written to a temporary folder, never to ``Data/ui_uploads``.

Run: ``.venv/bin/python Tools/test_app_bridge.py``
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from App import settings
from App.bridge import SENTENCE_NO, AppBridge
from Src.inference.runtime import backend_report
from Src.utils.constants import CORE_ML_MODEL_PATH, PROJECT_ROOT

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, cond: bool, extra: str = "") -> None:
    """Record one pass/fail line."""
    (PASSED if cond else FAILED).append(label)
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  [{extra}]" if extra else ""))


def make_sky_png(path: Path) -> None:
    """Write a synthetic 300x300 PNG sky: gaussian stars on faint noise."""
    from PIL import Image

    rng = np.random.default_rng(7)
    img = rng.normal(0.02, 0.01, (300, 300)).clip(0, 1)
    yy, xx = np.mgrid[:300, :300]
    for cy, cx, amp in [(150, 150, 1.0), (60, 200, 0.8), (230, 80, 0.7)]:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
    arr = (img.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").convert("RGB").save(path)


def make_dummy_checkpoint(path: Path) -> None:
    """Save an untrained AstroNet-v2 state_dict to a temporary path."""
    import torch

    from Src.models.psf_cnn import build_model

    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(device="cpu")
    torch.save({"model_state_dict": model.state_dict()}, path)


def upload(bridge: AppBridge, png: Path) -> dict:
    """Send a file through the same base64 path the UI uses."""
    return bridge.save_upload(
        png.name, base64.b64encode(png.read_bytes()).decode("ascii"))


def test_pytorch_fallback(tmp: Path) -> None:
    """Analyze with a temporary PyTorch checkpoint and check state retention."""
    print("\n--- PyTorch fallback ---")
    ckpt = tmp / "dummy.pt"
    make_dummy_checkpoint(ckpt)
    updir = tmp / "uploads_pt"
    updir.mkdir()
    bridge = AppBridge(model_path=ckpt, prefer_coreml=False,
                       upload_dir=updir)

    engine = bridge.engine()
    check("engine() reports pytorch", engine.get("backend") == "pytorch")
    check("why() before analyze is graceful", bridge.why().get("ok") is False)

    png = tmp / "sky.png"
    make_sky_png(png)
    up = upload(bridge, png)
    check("save_upload ok", bool(up.get("ok"))
          and Path(up.get("path", "")).exists())

    res = bridge.analyze(up["path"])
    check("analyze ok", res.get("ok") is True)
    sentence = res.get("sentence", "")
    if res.get("identified"):
        check("sentence ok-form",
              sentence.startswith("La foto che mi hai mostrato"), sentence[:60])
        check("confidence in 0..100",
              55.0 <= res.get("confidence", 0) <= 100.0,
              str(res.get("confidence")))
    else:
        check("sentence no-form", sentence == SENTENCE_NO, sentence[:60])
    top = res.get("top", [])
    check("top rows <=3", len(top) <= 3, str(len(top)))
    check("top pct floats", all(isinstance(t["pct"], float) for t in top))

    # Regression guards: cleanup must not null the state the chips use.
    why = bridge.why()
    check("why() after analyze works", why.get("ok") is True
          and len(why.get("message", "")) > 40)
    check("_last_result retained", bridge._last_result is not None)
    check("_last_image_path retained", bridge._last_image_path is not None)
    check("current upload kept on disk", Path(up["path"]).exists())
    check("exactly one upload retained", len(list(updir.iterdir())) == 1,
          str([p.name for p in updir.iterdir()]))

    # A second analysis must replace the old upload, not accumulate it.
    up2 = upload(bridge, png)
    bridge.analyze(up2["path"])
    check("second analysis keeps exactly one upload",
          len(list(updir.iterdir())) == 1,
          str([p.name for p in updir.iterdir()]))
    check("old upload removed", not Path(up["path"]).exists())

    missing = bridge.analyze(str(tmp / "nope.png"))
    check("missing file -> graceful error", missing.get("ok") is False
          and bool(missing.get("message")))

    # A checkpoint that fails to load must not leave uploads behind either.
    bad_dir = tmp / "uploads_bad"
    bad_dir.mkdir()
    bad = AppBridge(model_path=tmp / "missing.pt", prefer_coreml=False,
                    upload_dir=bad_dir)
    bad_up = upload(bad, png)
    bad_res = bad.analyze(bad_up["path"])
    check("missing checkpoint -> graceful error",
          bad_res.get("ok") is False
          and "Errore caricamento modello" in bad_res.get("message", ""),
          bad_res.get("message", "")[:70])
    check("failed model load cleans uploads",
          len(list(bad_dir.iterdir())) == 0,
          str([p.name for p in bad_dir.iterdir()]))

    had_key = settings.has_api_key()
    if had_key:
        print("SKIP  identify_object no-key path (a real key is configured)")
    else:
        ident = bridge.identify_object()
        check("identify_object no longer says 'Analizza prima'",
              "Analizza prima" not in ident.get("message", ""))
        check("identify_object reaches the no-key guard",
              "Nessuna chiave" in ident.get("message", ""),
              ident.get("message", "")[:70])

    check("engine() after load still pytorch",
          bridge.engine().get("backend") == "pytorch")
    r5 = bridge.save_upload("big.png", "A" * 210_000_000)
    check("oversize upload rejected", r5.get("ok") is False)


def test_key_roundtrip(tmp: Path) -> None:
    """Key save/clear must not damage a real user key."""
    print("\n--- OpenRouter key roundtrip ---")
    had_key = settings.has_api_key()
    if had_key:
        print("SKIP  key roundtrip (a real key is configured)")
        return
    raw = (settings.SETTINGS_PATH.read_text(encoding="utf-8")
           if settings.SETTINGS_PATH.exists() else None)
    try:
        settings.set_api_key("sk-or-v1-TEST_KEY_123")
        check("save_key/has_api_key", settings.has_api_key() is True)
        chain = settings.get_models()
        check("fallback chain loaded", len(chain) == 4
              and chain[0].startswith("google/gemma"), str(chain))
        settings.set_api_key("")
        check("clear key", settings.has_api_key() is False)
    finally:
        if raw is not None:
            settings.SETTINGS_PATH.write_text(raw, encoding="utf-8")
        else:
            settings.set_api_key("")


def test_coreml(tmp: Path) -> None:
    """Exercise the CoreML/ANE path on a real validation image."""
    print("\n--- CoreML / Neural Engine ---")
    report = backend_report()
    if not report["coreml_usable"]:
        print(f"SKIP  CoreML: {report['reason'] or 'not available'}")
        return

    updir = tmp / "uploads_coreml"
    updir.mkdir()
    bridge = AppBridge(upload_dir=updir)
    engine = bridge.engine()
    check("engine() reports coreml", engine.get("backend") == "coreml",
          str(engine))

    images = sorted((PROJECT_ROOT / "Data" / "png_validation"
                     / "hubble").glob("*.png"))
    image = images[0] if images else tmp / "sky.png"
    if not image.exists():
        make_sky_png(image)
    up = upload(bridge, image)
    res = bridge.analyze(up["path"])
    check("CoreML analyze ok", res.get("ok") is True,
          res.get("message", "")[:70])
    check("CoreML backend actually used", bridge._backend == "coreml")
    check("CoreML top rows <=3", len(res.get("top", [])) <= 3)
    why = bridge.why()
    check("CoreML why() works", why.get("ok") is True)
    check("CoreML label visible in why()",
          "CoreML" in why.get("message", ""),
          why.get("message", "")[:80])
    check("CoreML upload retained for identify",
          Path(up["path"]).exists())
    if settings.has_api_key():
        print("SKIP  identify_object on CoreML (real key configured)")
    else:
        ident = bridge.identify_object()
        check("CoreML identify_object reaches no-key guard",
              "Nessuna chiave" in ident.get("message", ""),
              ident.get("message", "")[:70])


def main() -> int:
    """Run both backend checks and return the process exit code."""
    print("=" * 62)
    print("=== SatAi APP BRIDGE — PyTorch fallback + CoreML/ANE ===")
    print("=" * 62)
    tmp = Path(tempfile.mkdtemp(prefix="satai_app_test_"))
    try:
        test_pytorch_fallback(tmp)
        test_key_roundtrip(tmp)
        test_coreml(tmp)
    except Exception as exc:                  # noqa: BLE001 — test harness
        import traceback

        traceback.print_exc()
        FAILED.append(f"unexpected-exception: {exc}")
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 62}")
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed checks:")
        for name in FAILED:
            print(f"  - {name}")
    print("=" * 62)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
