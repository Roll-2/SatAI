"""Neuron visualization toolkit for AstroNet-v2.

Answers, with pictures, the questions "what do the neurons do and what
did they learn?":

  1. **Filters**     — the 32 first-layer 3x3 weight grids (what each
     Stage-1 detector prefers to see).
  2. **Activations** — for one crop, the top-K most active feature maps
     of every conv stage ("this neuron lit up here").
  3. **Saliency**    — gradient of the winning class score w.r.t. the
     input pixels: which pixels drove the decision (for PSFs this
     should be the spikes and rings, not the background).
  4. **Dense1 map**  — heatmap of the 128 evidence-combiner neurons x
     256 GAP votes, plus each neuron's top-3 inputs.
  5. **Judges**      — the 9 class scores for the crop (bar chart).

Works on any checkpoint (or the untrained model). After training re-run
with ``--checkpoint Models/astro_net_v2_best.pt`` and compare with
the untrained run: the same pictures become the "after" state.

Usage (from the project root)::

    python Tools/visualize_neurons.py --input Data/crops/staging/hubble/xxx.npy
    python Tools/visualize_neurons.py --checkpoint Models/astro_net_v2_best.pt
    python Tools/visualize_neurons.py --input photo.png --mode png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI: headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Src.models.psf_cnn import build_model  # noqa: E402
from Src.utils.constants import CLASS_FOLDERS, MODELS_DIR  # noqa: E402

STAGES = ["stage1", "stage2", "stage3", "stage4"]  # conv block outputs
TOPK = 8                                            # maps shown per stage


def pick_device() -> torch.device:
    """MPS first, then CUDA, then CPU (same priority as training)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_input(path: Path | None, mode: str) -> tuple[np.ndarray, str]:
    """Load a 64x64 float32 [0,1] crop from .npy, PNG/JPG, or a demo star."""
    if path is None:
        # Demo star: 4-arm diffraction cross so every stage has something
        # to react to even without real data.
        y, x = np.mgrid[0:64, 0:64]
        r2 = (x - 32.0) ** 2 + (y - 32.0) ** 2
        img = np.exp(-r2 / 8.0).astype(np.float32)
        for angle in (0.0, np.pi / 2):
            u, v = np.cos(angle), np.sin(angle)
            arm = np.exp(-((x - 32.0) * v - (y - 32.0) * u) ** 2 / 1.5) \
                * np.exp(-np.abs((x - 32.0) * u + (y - 32.0) * v) / 6.0)
            img += arm.astype(np.float32) * 0.8
        img = np.clip(img / img.max(), 0, 1)
        return img, "demo synthetic 4-arm star"

    if mode == "fits" or path.suffix == ".npy":
        img = np.load(path).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        return np.clip(img, 0, 1), path.name

    from Src.preprocessing.loader import load_image  # PNG/JPG path
    img = load_image(str(path), mode="png").astype(np.float32)
    return np.clip(img, 0, 1), path.name


def fig_filters(model: AstroNetV2, out: Path) -> None:
    """Save the 32 Stage-1 3x3 kernels as one annotated grid."""
    w = model.features[0][0].weight.detach().cpu().numpy()[:, 0]  # (32,3,3)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    fig.suptitle("Stage 1 — the 32 'flashlight' detectors (3x3 weight grids)",
                 fontsize=11)
    for i, ax in enumerate(axes.flat):
        ax.imshow(w[i], cmap="coolwarm",
                  vmin=-np.abs(w).max(), vmax=np.abs(w).max())
        ax.set_title(f"n{i}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "1_filters_conv1.png", dpi=130)
    plt.close(fig)


def fig_activations(acts: dict[str, torch.Tensor], name: str,
                    out: Path) -> None:
    """Save top-K channel maps of every conv stage for one crop."""
    fig, axes = plt.subplots(len(STAGES), TOPK, figsize=(14, 9))
    fig.suptitle(f"Activation maps for: {name}\n"
                 f"(top {TOPK} most active neurons per stage)", fontsize=11)
    for row, stage in enumerate(STAGES):
        a = acts[stage][0].detach().cpu().numpy()          # (C, H, W)
        energy = a.sum(axis=(1, 2))
        order = np.argsort(energy)[::-1][:TOPK]
        vmax = a[order].max() or 1.0
        for col in range(TOPK):
            ax = axes[row, col]
            if col < len(order):
                ch = order[col]
                ax.imshow(a[ch], cmap="inferno", vmin=0, vmax=vmax)
                ax.set_title(f"n{ch} e={energy[ch]:.0f}", fontsize=7)
            ax.axis("off")
        axes[row, 0].set_ylabel(stage, fontsize=9)
    for row, stage in enumerate(STAGES):
        axes[row, 0].text(-0.35, 0.5, stage, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", fontsize=10)
    fig.tight_layout(rect=(0.03, 0, 1, 0.94))
    fig.savefig(out / "2_activations.png", dpi=130)
    plt.close(fig)


def fig_saliency(model: AstroNetV2, x: torch.Tensor, probs: np.ndarray,
                 pred: int, name: str, out: Path) -> None:
    """Gradient-of-score saliency: which pixels drove the decision."""
    xg = x.clone().requires_grad_(True)
    scores = model(xg)
    scores[0, pred].backward()
    sal = xg.grad[0, 0].detach().cpu().numpy()
    sal = np.abs(sal) / (np.abs(sal).max() or 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.suptitle(f"Saliency — pixels that drove the '{CLASS_FOLDERS[pred]}' "
                 f"decision ({probs[pred] * 100:.1f}% conf.)", fontsize=11)
    axes[0].imshow(x[0, 0].cpu(), cmap="gray")
    axes[0].set_title("input crop", fontsize=9)
    axes[1].imshow(sal, cmap="hot")
    axes[1].set_title("|gradient| (evidence map)", fontsize=9)
    axes[2].bar(range(len(CLASS_FOLDERS)), probs, color="steelblue")
    axes[2].set_xticks(range(len(CLASS_FOLDERS)))
    axes[2].set_xticklabels(CLASS_FOLDERS, rotation=45, ha="right",
                            fontsize=7)
    axes[2].set_title("the 9 judges' scores", fontsize=9)
    for a in axes[:2]:
        a.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "3_saliency_and_judges.png", dpi=130)
    plt.close(fig)


def fig_dense_map(model: AstroNetV2, out: Path) -> list[tuple[int, ...]]:
    """Dense1 x GAP-vote weight heatmap; return each neuron's top-3 inputs."""
    w = model.classifier[1].weight.detach().cpu().numpy()  # (128, 256)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(w, cmap="coolwarm", aspect="auto",
                   vmin=-np.abs(w).max(), vmax=np.abs(w).max())
    ax.set_xlabel("GAP vote (which Stage-4 detector)")
    ax.set_ylabel("Dense1 evidence-combiner neuron")
    ax.set_title("Dense1 weight map — how the 128 combiners read the "
                 "256 votes")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "4_dense1_weight_map.png", dpi=130)
    plt.close(fig)
    return [tuple(np.argsort(np.abs(w[n]))[::-1][:3]) for n in range(128)]


def write_report(model: AstroNetV2, tops: list[tuple[int, ...]],
                 probs: np.ndarray, pred: int, name: str,
                 trained: bool, out: Path) -> None:
    """Plain-text companion: numbers that the PNGs annotate."""
    w1 = model.features[0][0].weight.detach().cpu().numpy()
    lines = [
        "AstroNet-v2 neuron report",
        f"checkpoint: {'TRAINED' if trained else 'UNTRAINED (random weights)'}",
        f"input: {name}",
        f"prediction: {CLASS_FOLDERS[pred]} "
        f"({probs[pred] * 100:.1f}% confidence)",
        "",
        f"Stage 1: 32 detectors, 3x3 weights, "
        f"range [{w1.min():+.3f}, {w1.max():+.3f}] "
        f"(random range is +/-{1 / np.sqrt(9):.2f})",
        f"Top-3 inputs of the first 10 Dense1 neurons: "
        f"{tops[:10]}",
        "",
        "Judge scores:",
        *[f"  {c:<8} {p * 100:5.1f}%" for c, p in zip(CLASS_FOLDERS, probs)],
    ]
    (out / "report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Generate all neuron-visualization artifacts for one crop."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=None,
                    help="crop .npy or PNG/JPG; default = demo star")
    ap.add_argument("--mode", choices=["fits", "png"], default="fits")
    ap.add_argument("--checkpoint", type=Path,
                    default=MODELS_DIR / "astro_net_v2_best.pt")
    ap.add_argument("--out", type=Path, default=Path("reports/neurons"))
    args = ap.parse_args()

    device = pick_device()
    model = build_model(device=device)
    trained = args.checkpoint.exists()
    if trained:
        state = torch.load(args.checkpoint, map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded trained checkpoint: {args.checkpoint}")
    else:
        print(f"NOTE: {args.checkpoint} not found — using the UNTRAINED "
              f"model (random weights). Re-run after training for the "
              f"'after' pictures.")

    img, name = load_input(args.input, args.mode)
    x = torch.from_numpy(img)[None, None].to(device)

    acts: dict[str, torch.Tensor] = {}
    hooks = [model.features[i].register_forward_hook(
        lambda m, inp, o, s=STAGES[i]: acts.__setitem__(s, o.detach()))
        for i in range(len(STAGES))]

    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())

    fig_filters(model, args.out)
    fig_activations(acts, name, args.out)
    fig_saliency(model, x, probs, pred, name, args.out)
    tops = fig_dense_map(model, args.out)
    write_report(model, tops, probs, pred, name, trained, args.out)
    for h in hooks:
        h.remove()

    print(f"Artifacts written to {args.out}/:\n"
          f"  1_filters_conv1.png       what each Stage-1 detector prefers\n"
          f"  2_activations.png         neurons firing for this crop\n"
          f"  3_saliency_and_judges.png pixels that drove the decision\n"
          f"  4_dense1_weight_map.png   combiner neurons x 256 votes\n"
          f"  report.txt                numbers + judge scores\n"
          f"Prediction on '{name}': {CLASS_FOLDERS[pred]} "
          f"{probs[pred] * 100:.1f}%"
          f"{'' if trained else '  (untrained model — scores are noise)'}")


if __name__ == "__main__":
    main()
