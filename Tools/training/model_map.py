#!/usr/bin/env python3
"""Model-structure check: HOW the neurons of AstroNet-v2 connect.

Answers three questions with measured evidence from the live model:

1. SHAPE MAP   — what every layer receives and emits (real forward hooks,
                 not documentation).
2. WIRING      — which neurons feed which: per-layer parameter matrices
                 (who talks to whom and how many wires).
3. TRAINABILITY— receptive fields (what each neuron can "see" of the PSF)
                 and a gradient-flow probe (does every layer receive a
                 learning signal from the loss?).

Usage::

    .venv/bin/python Tools/training/model_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from Src.models.psf_cnn import build_model

LINE = "-" * 78


def main() -> None:
    """Run the instrumentation and print the connection map."""
    torch.manual_seed(0)
    model = build_model(device="cpu").eval()
    print("ASTRONET-v2 — LIVE MODEL CONNECTION MAP")
    print(LINE)

    # ------------------------------------------------------------------
    # 1. SHAPE MAP via forward hooks on every LEAF layer (execution order)
    # ------------------------------------------------------------------
    print("\n[1] SHAPE MAP (measured with a real forward pass)")
    print(LINE)
    print(f"{'layer':<26} {'input':<17} {'output':<17} type")

    shapes: list[tuple[str, tuple, tuple, str]] = []
    handles = []

    def make_hook(name, module):
        def hook(_m, inp, out):
            shapes.append((name, tuple(inp[0].shape), tuple(out.shape),
                           type(module).__name__))
        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:   # leaves only
            handles.append(module.register_forward_hook(make_hook(name, module)))

    x = torch.rand(1, 1, 64, 64)
    model(x)
    for h in handles:
        h.remove()

    for name, inp, out, kind in shapes:
        print(f"{name:<26} {str(inp):<17} {str(out):<17} {kind}")

    # ------------------------------------------------------------------
    # 2. WIRING: parameter accounting (who feeds whom)
    # ------------------------------------------------------------------
    print("\n[2] WIRING (learned neuron connections: weights + biases)")
    print(LINE)
    print(f"{'parameter':<26} {'shape':<18} {'connections':>12}  meaning")
    total = 0
    conv_dense = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "running_" in name or "num_batches" in name:
            continue                                    # BN statistics
        if "weight" in name and param.dim() == 4:
            cout, cin, kh, kw = param.shape
            n = param.numel()
            conv_dense += n
            print(f"{name:<26} {str(tuple(param.shape)):<18} {n:>12,}  "
                  f"{cout} filters, each wired to {cin} ch x {kh}x{kw} px")
        elif "weight" in name and param.dim() == 2:
            n = param.numel()
            conv_dense += n
            print(f"{name:<26} {str(tuple(param.shape)):<18} {n:>12,}  "
                  f"dense: {param.shape[1]} neurons -> {param.shape[0]} neurons")
    biases = total - conv_dense
    print(f"\n  conv+dense wires: {conv_dense:,}   biases/BN params: "
          f"{biases:,}   TOTAL learnable: {total:,}")
    print("  ReLU/MaxPool/GAP/Dropout: fixed wiring, zero learned weights.")

    # ------------------------------------------------------------------
    # 3a. RECEPTIVE FIELDS: what each neuron can see of the PSF
    # ------------------------------------------------------------------
    print(f"\n[3a] RECEPTIVE FIELD (how much of the 64x64 PSF a neuron sees)")
    print(LINE)
    rf, jump = 1, 1
    stages = [
        ("Conv1 3x3 s1", 3, 1), ("Pool1 2x2 s2", 2, 2),
        ("Conv2 3x3 s1", 3, 1), ("Pool2 2x2 s2", 2, 2),
        ("Conv3 3x3 s1", 3, 1), ("Pool3 2x2 s2", 2, 2),
        ("Conv4 3x3 s1", 3, 1), ("Pool4 2x2 s2", 2, 2),
    ]
    print(f"{'layer':<16} {'RF (input px)':>14} {'jump':>6}   note")
    for name, k, s in stages:
        rf += (k - 1) * jump
        jump *= s
        note = ""
        if name.startswith("Conv3"):
            note = "RF 18 px: neighboring spike arms enter view"
        if name.startswith("Conv4"):
            note = "RF 38 px: full cross-spike geometry"
        if name.startswith("Pool4"):
            note = "GAP averages these 4x4 cells -> shift/rot robustness"
        print(f"{name:<16} {rf:>14} {jump:>6}   {note}")
    print("\n  Hubble-style spikes span ~26 px radius in the crops; Conv4's")
    print("  38 px receptive field covers the whole cross, and GAP's global")
    print("  vote makes the readout insensitive to where the star sits.")

    # ------------------------------------------------------------------
    # 3b. GRADIENT FLOW: is every layer actually trainable?
    # ------------------------------------------------------------------
    print("\n[3b] GRADIENT FLOW PROBE (one real backward pass)")
    print(LINE)
    model.train()
    batch = torch.rand(16, 1, 64, 64)
    labels = torch.randint(0, 9, (16,))
    loss = torch.nn.functional.cross_entropy(model(batch), labels)
    loss.backward()

    print(f"{'parameter':<34} {'|grad| norm':>12}  status")
    dead = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name:<34} {'None':>12}  NOT CONNECTED")
            dead += 1
            continue
        norm = float(param.grad.norm())
        if norm == 0:
            dead += 1
            print(f"{name:<34} {norm:>12.2e}  ZERO GRAD")
    print(f"  loss = {loss.item():.4f}")
    print(f"  -> {'ALL parameter tensors receive learning signal' if dead == 0 else f'{dead} DEAD TENSORS'}")

    # ------------------------------------------------------------------
    # 4. CAPACITY SANITY (overfitting risk)
    # ------------------------------------------------------------------
    print(f"\n[4] CAPACITY SANITY (samples per parameter)")
    print(LINE)
    for n_samples in (45000, 4500):
        ratio = n_samples / total
        verdict = "comparable to ResNet50-on-ImageNet (0.05): OK with regularization" \
            if ratio <= 0.1 else "comfortable"
        print(f"  {n_samples:>6} samples / {total:,} params = "
              f"{ratio:.2f} samples per parameter ({verdict})")
    print("  For reference: ImageNet trains ResNet50 at 0.05 samples/param.")
    print("  AstroNet-v2 relies on GAP + dropout + per-class augmentation.")


if __name__ == "__main__":
    main()
