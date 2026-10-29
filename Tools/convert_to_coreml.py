"""Export the trained AstroNet-v2 checkpoint to CoreML (Apple Neural Engine).

This does **not** retrain and does not replace the PyTorch checkpoint: it
creates a second, identical-weights model at ``Models/AstroNetV2.mlpackage``
that CoreML can schedule on the ANE.

The network is deliberately simple (conv/BN/ReLU/max-pool/GAP/linear), so
the whole graph converts to ANE-friendly CoreML ops. BatchNorm is folded
into the convolutions by the converter, and the global average pool becomes
a ``reduce_mean``. Input is fixed at ``(1, 1, 64, 64)`` float32; output is
``(1, 9)`` raw logits (softmax stays in the app, exactly like the PyTorch
path).

Usage (from the project root)::

    .venv/bin/python Tools/convert_to_coreml.py
    .venv/bin/python Tools/convert_to_coreml.py --force
    .venv/bin/python Tools/convert_to_coreml.py --verify 64

Exit code 0 means: converted, saved, and the CoreML logits agree with the
PyTorch logits on real test crops within 1e-3 (argmax must match on all).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from Src.models.psf_cnn import build_model
from Src.utils.constants import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CORE_ML_INPUT_NAME,
    CORE_ML_MODEL_PATH,
    CORE_ML_OUTPUT_NAME,
    CROP_SIZE,
    TEST_DIR,
)

MAX_ABS_DIFF = 1e-3


def load_torch_model(checkpoint: Path) -> torch.nn.Module:
    """Load the trained checkpoint on CPU in eval mode."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model = build_model(device="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _real_crops(limit: int) -> list[np.ndarray]:
    """Pick up to ``limit`` real test crops, else random tensors."""
    files = sorted(TEST_DIR.glob("**/*.npy"))
    random.Random(42).shuffle(files)
    crops: list[np.ndarray] = []
    for path in files[:limit]:
        try:
            crop = np.load(path).astype(np.float32)
        except (OSError, ValueError):
            continue
        if crop.shape == (CROP_SIZE, CROP_SIZE):
            crops.append(np.clip(crop, 0.0, 1.0))
    if not crops:
        rng = np.random.default_rng(42)
        crops = [rng.random((CROP_SIZE, CROP_SIZE), dtype=np.float32)
                 for _ in range(min(limit, 8))]
    return crops


def convert(checkpoint: Path, output: Path, force: bool,
            verify: int = 32) -> int:
    """Convert and save one checkpoint; return the process exit code."""
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1
    if output.exists():
        if not force:
            print(f"Output already exists: {output}\nUse --force to rebuild it.")
            return 1
        shutil.rmtree(output)

    import coremltools as ct

    print(f"Loading PyTorch checkpoint: {checkpoint}")
    model = load_torch_model(checkpoint)
    example = torch.rand(1, 1, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.eval()

    print("Converting to CoreML (mlprogram, macOS 13+, float32) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=CORE_ML_INPUT_NAME,
                              shape=(1, 1, CROP_SIZE, CROP_SIZE),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name=CORE_ML_OUTPUT_NAME, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )
    mlmodel.author = "SatAi"
    mlmodel.short_description = (
        "AstroNet-v2 — 9-class telescope PSF classifier (64x64 grayscale)")
    mlmodel.version = "2.0"
    mlmodel.user_defined_metadata["class_names"] = ", ".join(CLASS_NAMES)
    mlmodel.user_defined_metadata["source_checkpoint"] = checkpoint.name
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))

    size_mb = sum(p.stat().st_size for p in output.rglob("*")
                  if p.is_file()) / (1024 * 1024)
    print(f"Saved: {output}  ({size_mb:.2f} MB)")

    # ------------------------------------------------------------------
    # Verification: same weights -> same logits/argmax on real crops.
    # ------------------------------------------------------------------
    crops = _real_crops(max(1, verify))
    coreml = ct.models.MLModel(str(output),
                               compute_units=ct.ComputeUnit.CPU_AND_NE)
    max_diff = 0.0
    wrong = 0
    torch_logits: list[np.ndarray] = []
    core_logits: list[np.ndarray] = []
    with torch.no_grad():
        for crop in crops:
            x = crop.reshape(1, 1, CROP_SIZE, CROP_SIZE)
            a = model(torch.from_numpy(x)).numpy()
            b = np.asarray(coreml.predict({CORE_ML_INPUT_NAME: x})[
                CORE_ML_OUTPUT_NAME], dtype=np.float64)
            torch_logits.append(a.reshape(-1))
            core_logits.append(b.reshape(-1))
            max_diff = max(max_diff, float(np.abs(a - b).max()))
            if int(a.argmax()) != int(b.argmax()):
                wrong += 1

    print(f"Verified on {len(crops)} real crops:")
    print(f"  max |PyTorch - CoreML| logit difference: {max_diff:.3e}")
    print(f"  argmax agreement: {len(crops) - wrong}/{len(crops)}")

    # Latency, one crop at a time (the app's shape).
    x = crops[0].reshape(1, 1, CROP_SIZE, CROP_SIZE)
    for _ in range(5):
        coreml.predict({CORE_ML_INPUT_NAME: x})
    t0 = time.perf_counter()
    for _ in range(100):
        coreml.predict({CORE_ML_INPUT_NAME: x})
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"  CoreML latency: {ms:.3f} ms/crop (CPU+Neural Engine)")

    if max_diff > MAX_ABS_DIFF or wrong:
        print(f"FAILED: conversion mismatch (tolerance {MAX_ABS_DIFF:g}).")
        return 1
    print("OK: CoreML model is numerically equivalent to PyTorch.")
    return 0


def main() -> int:
    """Parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH,
                        help="PyTorch checkpoint to export")
    parser.add_argument("--output", type=Path, default=CORE_ML_MODEL_PATH,
                        help="destination .mlpackage")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing .mlpackage")
    parser.add_argument("--verify", type=int, default=32,
                        help="number of real crops used to compare logits")
    args = parser.parse_args()
    return convert(args.checkpoint, args.output, args.force, args.verify)


if __name__ == "__main__":
    sys.exit(main())
