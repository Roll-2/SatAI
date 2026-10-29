"""Build the compact data file consumed by the Blender scene.

The visualization is not a drawing of an idealized CNN: every kernel summary,
filter norm, connection and activation field comes from the trained checkpoint
``Models/astro_net_v2_best.pt`` and one real Hubble test crop.

Output: ``Visuals/network_data.json``

Run with the project virtualenv::

    .venv/bin/python Visuals/build_network_data.py

The activation tensors are quantized to uint8 and base64-encoded so the file
stays small enough to be embedded in the Blender MCP session.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.models.psf_cnn import build_model
from Src.utils.constants import CLASS_NAMES, CROP_SIZE, TEST_DIR
CHECKPOINT = PROJECT_ROOT / "Models" / "astro_net_v2_best.pt"
# Highest-confidence real Hubble crop in the test split (p(Hubble)=0.9991).
# Synthetic spikes are deliberately excluded: this is a real instrument crop.
CROP = (PROJECT_ROOT / "Data" / "crops" / "test" / "hubble"
        / "hubble_0009_M33_0009_crop02.npy")
OUT = PROJECT_ROOT / "Visuals" / "network_data.json"

#: 5 classes with real training data; 4 classes reserved for future missions.
REAL_CLASSES = {"Hubble", "JWST", "Spitzer", "WISE", "Altro"}


def b64_u8(array: np.ndarray) -> str:
    """Base64-encode a uint8 array in C order."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii")


def q8(array: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Quantize a non-negative activation tensor to uint8 in 0..255."""
    a = np.asarray(array, dtype=np.float32)
    pos = a[a > 0]
    scale = float(np.percentile(pos, percentile)) if pos.size else 1.0
    scale = max(scale, 1e-8)
    return np.round(np.clip(a / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalise(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Normalize a non-negative vector with a percentile ceiling."""
    v = np.asarray(values, dtype=np.float32)
    hi = float(np.percentile(v, percentile)) if v.size else 1.0
    return np.clip(v / max(hi, 1e-8), 0.0, 1.0)


def main() -> int:
    """Extract the model summaries and write the JSON bundle."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    crop = np.load(CROP).astype(np.float32)
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"unexpected crop shape: {crop.shape}")

    # Hooks capture post-ReLU/post-pool activations, not the pre-activation
    # tensors: these are the values the next layer actually receives.
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().cpu()
        return hook

    model.features[0].register_forward_hook(capture("conv1"))
    model.features[1].register_forward_hook(capture("conv2"))
    model.features[2].register_forward_hook(capture("conv3"))
    model.features[3].register_forward_hook(capture("conv4"))
    model.gap.register_forward_hook(capture("gap"))
    model.classifier[2].register_forward_hook(capture("dense1"))

    x = torch.from_numpy(crop)[None, None]
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].numpy()

    sd = checkpoint["model_state_dict"]
    block_specs = [
        ("conv1", "features.0", 32, (32, 32)),
        ("conv2", "features.1", 64, (16, 16)),
        ("conv3", "features.2", 128, (8, 8)),
        ("conv4", "features.3", 256, (4, 4)),
    ]

    blocks = []
    for name, prefix, channels, spatial in block_specs:
        weight = sd[f"{prefix}.0.weight"].float().numpy()   # (Cout, Cin, 3, 3)
        bn = sd[f"{prefix}.1.weight"].float().numpy()
        act = captured[name][0].numpy()                     # (C, H, W)
        if act.shape != (channels, *spatial):
            raise ValueError(f"{name}: activation shape {act.shape}")

        # One 3x3 summary kernel per output filter: signed mean over input
        # channels. The dots in the Blender filter volume use these values.
        kernels = weight.mean(axis=1)                       # (Cout, 3, 3)
        # A filter is 3x3 x Cin. For the 3D scene we expose three real depth
        # slices (input channels split into low/mid/high groups), so the
        # filter volume is a tiny volumetric kernel, not a flat card.
        cin = weight.shape[1]
        groups = np.array_split(np.arange(cin), min(3, cin))
        slice_kernels = []
        for group in groups:
            slice_kernels.append(weight[:, group, :, :].mean(axis=1))
        all_slices = np.concatenate([s.ravel() for s in slice_kernels])
        dot_ceiling = float(np.percentile(np.abs(all_slices), 99.5))
        dot_ceiling = max(dot_ceiling, 1e-8)

        filter_norm = np.sqrt(np.sum(weight.astype(np.float64) ** 2, axis=(1, 2, 3)))
        filter_norm = normalise(filter_norm)

        channel_act = act.mean(axis=(1, 2))
        channel_act = normalise(channel_act, 99.5)

        filters = []
        for i in range(channels):
            filters.append({
                "kernel": [round(float(v) / dot_ceiling, 4) for v in kernels[i].ravel()],
                "slices": [
                    [round(float(v) / dot_ceiling, 4) for v in sk[i].ravel()]
                    for sk in slice_kernels
                ],
                "norm": round(float(filter_norm[i]), 4),
                "act": round(float(channel_act[i]), 4),
                "bn": round(float(abs(bn[i])), 4),
            })

        blocks.append({
            "name": name,
            "shape": [channels, spatial[0], spatial[1]],
            "activation": b64_u8(q8(act)),
            "filters": filters,
        })

    gap = captured["gap"][0].numpy().reshape(-1)
    dense1 = captured["dense1"][0].numpy().reshape(-1)
    gap_norm = normalise(gap, 99.5)
    dense1_norm = normalise(dense1, 99.5)

    # Connections are the actual strongest weight links, not random fibers.
    # For conv transitions, aggregate |w| over the 3x3 kernel and keep the two
    # strongest input channels per output channel. For the dense head, keep
    # the strongest four/four/eight inputs per node.
    def top_connections(signed_matrix: np.ndarray, top_k: int) -> list[list[float]]:
        """Keep the strongest links, preserving the signed weight direction."""
        w = np.asarray(signed_matrix, dtype=np.float32)
        magnitude = np.abs(w)
        ceiling = float(np.percentile(magnitude, 99.5))
        ceiling = max(ceiling, 1e-8)
        out: list[list[float]] = []
        for dst in range(w.shape[0]):
            order = np.argsort(magnitude[dst])[::-1][:top_k]
            for src in order:
                out.append([
                    int(src), int(dst),
                    round(float(min(1.0, magnitude[dst, src] / ceiling)), 4),
                    1 if float(w[dst, src]) >= 0 else -1,
                ])
        return out

    def conv_connections(weight_tensor: torch.Tensor, top_k: int = 2) -> list[list[float]]:
        # (Cout, Cin, 3, 3) -> (Cout, Cin), signed sum over the kernel.
        w = weight_tensor.float().numpy()
        return top_connections(w.sum(axis=(2, 3)), top_k)

    conv1_w = sd["features.0.0.weight"]
    conv2_w = sd["features.1.0.weight"]
    conv3_w = sd["features.2.0.weight"]
    conv4_w = sd["features.3.0.weight"]
    dense1_w = sd["classifier.1.weight"].numpy()
    dense2_w = sd["classifier.4.weight"].numpy()
    bn4_norm = normalise(np.abs(np.asarray(sd["features.3.1.weight"])))

    # Input -> Conv1: one representative source per filter, with the filter
    # norm and kernel sign carrying the weight information. The Blender scene
    # places the source at the corresponding point of the input image.
    input_conv1 = []
    k1 = conv1_w.float().numpy().mean(axis=1)
    norm1 = normalise(np.sqrt(np.sum(conv1_w.float().numpy() ** 2, axis=(1, 2, 3))))
    for dst in range(32):
        input_conv1.append([
            0, dst, round(float(norm1[dst]), 4),
            1 if float(k1[dst].sum()) >= 0 else -1,
        ])

    connections = {
        "input_conv1": input_conv1,
        "conv1_conv2": conv_connections(conv2_w, 8),
        "conv2_conv3": conv_connections(conv3_w, 8),
        "conv3_conv4": conv_connections(conv4_w, 8),
        "conv4_gap": [
            [i, i, round(float(bn4_norm[i]), 4),
             1 if float(sd["features.3.1.weight"][i]) >= 0 else -1]
            for i in range(256)
        ],
        "gap_dense1": top_connections(dense1_w, 8),
        "dense1_dense2": top_connections(dense2_w, 16),
    }

    # Test support makes the "5 real / 4 future" distinction explicit.
    support = []
    for class_name in CLASS_NAMES:
        folder = TEST_DIR / class_name.lower()
        support.append(len(list(folder.glob("*.npy"))) if folder.exists() else 0)

    input_display = q8(crop, 99.8)
    predicted = int(np.argmax(probabilities))

    payload = {
        "meta": {
            "checkpoint": CHECKPOINT.name,
            "epoch": int(checkpoint.get("epoch", -1)),
            "val_accuracy": float(checkpoint.get("val_accuracy", 0.0)),
            "crop": str(CROP.relative_to(PROJECT_ROOT)),
            "predicted_class": CLASS_NAMES[predicted],
            "confidence": round(float(probabilities[predicted]), 6),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "real_classes": [name for name in CLASS_NAMES if name in REAL_CLASSES],
        },
        "input": {
            "shape": [64, 64],
            "pixels": b64_u8(input_display),
        },
        "blocks": blocks,
        "gap": {
            "activation": b64_u8(np.round(gap_norm * 255).astype(np.uint8)),
        },
        "dense1": {
            "activation": b64_u8(np.round(dense1_norm * 255).astype(np.uint8)),
        },
        "connections": connections,
        "classes": {
            "names": CLASS_NAMES,
            "probabilities": [round(float(p), 6) for p in probabilities],
            "support": support,
            "real": [name in REAL_CLASSES for name in CLASS_NAMES],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")
    print(f"crop: {CROP.name}")
    print(f"prediction: {CLASS_NAMES[predicted]} {probabilities[predicted]:.4f}")
    print(f"parameters: {payload['meta']['parameters']:,}")
    print("connections:", {k: len(v) for k, v in connections.items()})
    print("support:", dict(zip(CLASS_NAMES, support)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
