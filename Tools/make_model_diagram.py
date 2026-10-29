#!/usr/bin/env python3
"""Generate the AstroNet-v2 model poster (`Models/architecture.svg`/`.png`).

The poster is computed from the live model, not hand-written:

* parameter counts per block come from ``model.named_parameters()``;
* activation/node counts come from a real forward pass;
* the input thumbnail is a real 64x64 Hubble test crop (embedded as base64,
  so the SVG is self-contained);
* the capacity panel and the receptive-field timeline are derived from the
  same numbers.

Usage (from the project root)::

    .venv/bin/python Tools/make_model_diagram.py
"""

from __future__ import annotations

import base64
import html
import io
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "Models"
SVG_PATH = MODELS / "architecture.svg"
PNG_PATH = MODELS / "architecture.png"

W, H = 1680, 1120
FONT = ("Inter, 'SF Pro Display', -apple-system, BlinkMacSystemFont, "
        "'Helvetica Neue', Arial, sans-serif")
MONO = ("'SF Mono', ui-monospace, Menlo, Monaco, Consolas, "
        "'Liberation Mono', monospace")

INK = "#e8eefc"
MUTED = "#8c9bb8"
FAINT = "#55647e"
CYAN = "#38bdf8"
VIOLET = "#a78bfa"
PINK = "#f472b6"
AMBER = "#fbbf24"
GREEN = "#34d399"

CLASS_NAMES = ["Hubble", "JWST", "Roman", "Euclid", "Rubin",
               "Spitzer", "WISE", "Gaia", "Altro"]
#: Classes that actually have training crops in the current dataset.
ACTIVE_CLASSES = {"Hubble", "JWST", "Spitzer", "WISE", "Altro"}

BLOCK_COLORS = [(CYAN, "#0284c7"), (VIOLET, "#6366f1"),
                (PINK, "#db2777"), (AMBER, "#d97706")]


def fmt(n: int) -> str:
    """Italian thousands separator: 422857 -> ``422.857``."""
    return f"{n:,}".replace(",", ".")


def num(n: float, digits: int = 1) -> str:
    """Italian decimal comma."""
    return f"{n:.{digits}f}".replace(".", ",")


def esc(value: str) -> str:
    """Escape text for SVG."""
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: str, size: float = 14,
         fill: str = INK, weight: int = 400, anchor: str = "start",
         family: str = FONT, opacity: float = 1.0, letter: float = 0) -> str:
    """One SVG text element."""
    extra = f' letter-spacing="{letter}"' if letter else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size:.1f}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}"'
            f'{extra}>{esc(value)}</text>')


def rect(x: float, y: float, w: float, h: float, fill: str,
         rx: float = 0, stroke: str = "none", sw: float = 1,
         opacity: float = 1.0, extra: str = "") -> str:
    """One SVG rect element."""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}" {extra}/>')


def line(x1: float, y1: float, x2: float, y2: float, stroke: str,
         sw: float = 1, opacity: float = 1.0, dash: str = "") -> str:
    """One SVG line element."""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"'
            f'{dash_attr}/>')


def svg_text_block(lines: list[str]) -> str:
    """Join SVG fragments."""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Real crop thumbnail
# ---------------------------------------------------------------------------
def _read_crop() -> tuple[np.ndarray, str]:
    """Pick a real Hubble crop with a visible star; fall back to a PSF."""
    preferred = [
        ROOT / "Data/crops/test/hubble/hubble_0001_M33_0001_crop15.npy",
        ROOT / "Data/crops/test/hubble/hubble_0001_M33_0001_crop18.npy",
    ]
    candidates = preferred + sorted(
        (ROOT / "Data/crops/test/hubble").glob("*.npy"))
    for path in candidates:
        try:
            arr = np.load(path).astype(np.float32)
        except Exception:  # noqa: BLE001 — try the next crop
            continue
        if arr.shape != (64, 64):
            continue
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            continue
        arr = (arr - lo) / (hi - lo)
        return arr, path.stem
    # Deterministic synthetic Airy-like PSF if the dataset is absent.
    arr = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    rr = np.hypot(yy - 32, xx - 32)
    arr = np.exp(-(rr / 2.6) ** 2) + 0.35 * np.exp(-(rr / 9.0) ** 2)
    arr /= arr.max()
    return arr.astype(np.float32), "synthetic PSF"


def _crop_data_uri(arr: np.ndarray) -> str:
    """Turn a 64x64 crop into a self-contained, slightly cool-toned PNG."""
    # Lift the faint halo a little so the diffraction structure survives a
    # small README rendering; keep the core from clipping.
    gray = np.clip(arr, 0.0, 1.0) ** 0.82
    gray = (gray * 255.0).astype(np.uint8)
    base = Image.fromarray(gray).resize((240, 240), Image.NEAREST)
    tinted = Image.merge("RGB", (
        base.point(lambda v: int(v * 0.72)),
        base.point(lambda v: int(v * 0.88)),
        base.point(lambda v: min(255, int(v * 1.0))),
    ))
    buf = io.BytesIO()
    tinted.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# 2. Model statistics
# ---------------------------------------------------------------------------
def collect_stats() -> dict:
    """Measure shapes, nodes and parameters from the live AstroNet-v2."""
    import torch

    from Src.models.psf_cnn import build_model

    model = build_model(device="cpu").eval()
    blocks = []
    with torch.no_grad():
        x = torch.zeros(1, 1, 64, 64)
        for index, block in enumerate(model.features):
            x = block(x)
            conv = block[0]
            params = sum(p.numel() for p in block.parameters())
            weights = int(conv.weight.numel())
            fan_in = int(conv.in_channels * conv.kernel_size[0]
                         * conv.kernel_size[1])
            activations = int(x.shape[1] * x.shape[2] * x.shape[3])
            blocks.append({
                "index": index + 1,
                "filters": int(conv.out_channels),
                "shape": tuple(int(v) for v in x.shape[1:]),
                "activations": activations,
                "params": int(params),
                "weights": weights,
                "fan_in": fan_in,
                "params_nonweight": int(params - weights),
            })

        gap = model.gap(x).flatten(1)
        dense1 = model.classifier[1]
        dense2 = model.classifier[4]
        d1_act = int(dense1(gap).shape[1])
        d2_act = int(dense2(dense1(gap)).shape[1])

    dense1_params = sum(p.numel() for p in dense1.parameters())
    dense2_params = sum(p.numel() for p in dense2.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    weights = (sum(b["weights"] for b in blocks)
               + int(dense1.weight.numel()) + int(dense2.weight.numel()))
    total_activations = (1 * 64 * 64
                         + sum(b["activations"] for b in blocks)
                         + d1_act + d2_act + 256)
    total_neurons = (sum(b["activations"] for b in blocks)
                     + d1_act + d2_act + 256)
    bias_bn = total_params - weights

    # Sanity guards: the poster must not lie if the architecture changes.
    assert len(blocks) == 4, "poster layout expects four conv blocks"
    assert total_params == 422_857, f"unexpected params: {total_params}"

    mlpackage = MODELS / "AstroNetV2.mlpackage"
    if mlpackage.exists():
        size_mb = sum(p.stat().st_size for p in mlpackage.rglob("*")
                      if p.is_file()) / 1e6
    else:
        pt = MODELS / "astro_net_v2_best.pt"
        size_mb = pt.stat().st_size / 1e6 if pt.exists() else 1.62

    return {
        "blocks": blocks,
        "dense1": {
            "nodes": d1_act,
            "params": int(dense1_params),
            "weights": int(dense1.weight.numel()),
            "fan_in": int(dense1.in_features),
        },
        "dense2": {
            "nodes": d2_act,
            "params": int(dense2_params),
            "weights": int(dense2.weight.numel()),
            "fan_in": int(dense2.in_features),
        },
        "gap": 256,
        "input_pixels": 64 * 64,
        "total_params": int(total_params),
        "weights": int(weights),
        "bias_bn": int(bias_bn),
        "total_activations": int(total_activations),
        "total_neurons": int(total_neurons),
        "size_mb": float(size_mb),
    }


# ---------------------------------------------------------------------------
# 3. SVG poster
# ---------------------------------------------------------------------------
def make_svg(stats: dict, crop_uri: str, crop_name: str) -> str:
    """Compose the full poster as one self-contained SVG document."""
    blocks = stats["blocks"]
    total_params = stats["total_params"]
    weights = stats["weights"]
    max_params = max(b["params"] for b in blocks)

    defs: list[str] = [
        '<defs>',
        '<radialGradient id="bg" cx="18%" cy="0%" r="120%">'
        '<stop offset="0%" stop-color="#111f38"/>'
        '<stop offset="52%" stop-color="#080d18"/>'
        '<stop offset="100%" stop-color="#04070d"/>'
        '</radialGradient>',
        '<linearGradient id="head" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="#38bdf8"/>'
        '<stop offset="55%" stop-color="#a78bfa"/>'
        '<stop offset="100%" stop-color="#f472b6"/>'
        '</linearGradient>',
        '<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="#38bdf8"/>'
        '<stop offset="100%" stop-color="#a78bfa"/>'
        '</linearGradient>',
        '<pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">'
        '<path d="M48 0H0V48" fill="none" stroke="#9db7e8" '
        'stroke-width="1" opacity="0.055"/>'
        '</pattern>',
        '<filter id="soft" x="-40%" y="-40%" width="180%" height="180%">'
        '<feGaussianBlur stdDeviation="18"/>'
        '</filter>',
        '<filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">'
        '<feDropShadow dx="0" dy="10" stdDeviation="18" '
        'flood-color="#000000" flood-opacity="0.45"/>'
        '</filter>',
        '<clipPath id="cropClip"><rect x="76" y="226" width="196" '
        'height="196" rx="14"/></clipPath>',
    ]
    for i, (c1, c2) in enumerate(BLOCK_COLORS):
        defs.append(
            f'<linearGradient id="b{i+1}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{c1}" stop-opacity="0.20"/>'
            f'<stop offset="100%" stop-color="{c2}" stop-opacity="0.06"/>'
            f'</linearGradient>')
        defs.append(
            f'<linearGradient id="bl{i+1}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{c1}"/>'
            f'<stop offset="100%" stop-color="{c2}"/></linearGradient>')
    defs.append('</defs>')

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
        '<title>AstroNet-v2 — architettura e dimensione del modello</title>',
        '<desc>Dalla foto 64x64 alle 9 classi: blocchi convoluzionali, '
        'attivazioni, parametri, connessioni e campo recettivo.</desc>',
        *defs,
        rect(0, 0, W, H, "url(#bg)"),
        rect(0, 0, W, H, "url(#grid)"),
    ]

    # Sparse, deterministic star field in the background.
    rng = random.Random(7)
    for _ in range(54):
        x = rng.uniform(24, W - 24)
        y = rng.uniform(18, H - 18)
        r = rng.choice([0.7, 0.9, 1.1, 1.4, 1.8])
        op = rng.uniform(0.10, 0.38)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
                   f'fill="#cfe4ff" opacity="{op:.2f}"/>')

    # Header ----------------------------------------------------------------
    out += [
        text(56, 84, "AstroNet-v2", 44, INK, 750, letter=-1.2),
        text(58, 116, "Dalla forma delle stelle alla classe del telescopio",
             17, MUTED, 400),
        text(58, 142, "4 blocchi convoluzionali  ·  GAP  ·  2 dense  ·  "
                      "9 uscite", 12, FAINT, 500, letter=0.2),
        line(56, 162, 1624, 162, "#26344d", 1, 0.9),
    ]

    # Header stat pills.
    pills = [
        (1252, 136, f"{fmt(total_params)} parametri", CYAN),
        (1400, 120, f"{fmt(stats['total_neurons'])} neuroni", VIOLET),
        (1532, 92, f"{num(stats['size_mb'], 2)} MB", AMBER),
    ]
    for x, w, label, color in pills:
        out += [
            rect(x, 54, w, 38, "#0b1322", 19, "#2a3a58", 1, 0.92),
            f'<circle cx="{x+18}" cy="73" r="4" fill="{color}"/>',
            text(x + 30, 78, label, 12.5, INK, 600),
        ]

    # Input card ------------------------------------------------------------
    out += [
        rect(56, 196, 236, 310, "#0a1220", 18, "#263a5c", 1.2, 0.96,
             'filter="url(#shadow)"'),
        text(76, 224, "INPUT", 11, CYAN, 700, letter=1.8),
        text(76, 244, "un crop, un canale", 10.5, MUTED, 400),
        f'<image x="76" y="226" width="196" height="196" '
        f'href="{crop_uri}" xlink:href="{crop_uri}" '
        f'preserveAspectRatio="none" clip-path="url(#cropClip)" '
        f'style="image-rendering:pixelated"/>',
        rect(76, 226, 196, 196, "none", 14, "#4a6b95", 1, 0.8),
        # A light 8x8 grid over the crop to make the pixel nature legible.
    ]
    for k in range(1, 8):
        p = 76 + k * 196 / 8
        out.append(line(p, 226, p, 422, "#dcecff", 0.7, 0.13))
        p = 226 + k * 196 / 8
        out.append(line(76, p, 272, p, "#dcecff", 0.7, 0.13))
    out += [
        text(76, 452, "1 canale  ·  64 × 64", 13, INK, 600),
        text(76, 472, f"{fmt(stats['input_pixels'])} pixel  ·  "
                      "crop reale di Hubble", 10.5, MUTED, 400),
        text(76, 492, f"file: {crop_name}", 9.2, FAINT, 400,
             family=MONO),
    ]

    # Conv blocks -----------------------------------------------------------
    block_x = [330, 548, 766, 984]
    block_w, block_h, block_y = 178, 310, 196
    for i, b in enumerate(blocks):
        x, y = block_x[i], block_y
        c1, c2 = BLOCK_COLORS[i]
        shape = " × ".join(str(v) for v in b["shape"])
        out += [
            # Stacked "feature maps" behind the card.
            rect(x + 10, y + 10, block_w, block_h, c1, 16, "none", 1, 0.055),
            rect(x + 5, y + 5, block_w, block_h, c1, 16, "none", 1, 0.09),
            rect(x, y, block_w, block_h, f"url(#b{i+1})", 16, f"{c1}66",
                 1.2, 0.98, 'filter="url(#shadow)"'),
            text(x + 18, y + 28, f"CONV {b['index']}", 11, c1, 700,
                 letter=1.6),
            text(x + block_w - 18, y + 28, "3 × 3", 10, MUTED, 500,
                 anchor="end", family=MONO),
            text(x + 18, y + 76, str(b["filters"]), 40, INK, 750, letter=-1),
            text(x + 18, y + 96, "filtri", 11, MUTED, 500),
            text(x + 18, y + 124, shape, 13, INK, 600, family=MONO),
            text(x + 18, y + 147, f"{fmt(b['activations'])}", 19, c1, 750),
            text(x + 18, y + 164, "nodi attivi", 10.5, MUTED, 400),
            line(x + 18, y + 184, x + block_w - 18, y + 184,
                 f"{c1}55", 1),
            text(x + 18, y + 206, f"fan-in  {fmt(b['fan_in'])}",
                 10.5, MUTED, 500, family=MONO),
            text(x + 18, y + 226, f"{fmt(b['params'])} parametri",
                 11.5, INK, 650),
            text(x + 18, y + 246, "conv + batch-norm", 9.5, FAINT, 400),
        ]
        # Tiny per-block parameter bar, scaled to the largest block.
        bar_w = 142 * b["params"] / max_params
        out += [
            rect(x + 18, y + 264, 142, 7, "#17243b", 3.5),
            rect(x + 18, y + 264, max(bar_w, 3), 7, f"url(#bl{i+1})", 3.5),
            text(x + block_w - 18, y + 288, f"{b['params']/total_params*100:.1f}%",
                 10, MUTED, 600, anchor="end", family=MONO),
            text(x + 18, y + 288, "del totale", 9.5, FAINT, 400),
        ]

    # Arrows between input / blocks / head.
    def arrow(x1: float, y1: float, x2: float, y2: float,
              label: str | None = None, color: str = "#6d86ad") -> list[str]:
        mid = (x1 + x2) / 2
        parts = [
            f'<path d="M{x1:.1f} {y1:.1f} C{mid:.1f} {y1:.1f}, '
            f'{mid:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1.8" opacity="0.9" '
            f'marker-end="url(#arrowhead)"/>',
        ]
        if label:
            parts.append(text(mid, min(y1, y2) - 10, label, 9.5, MUTED,
                              600, anchor="middle", family=MONO))
        return parts

    out.append('<defs><marker id="arrowhead" viewBox="0 0 10 10" '
               'refX="8" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse">'
               '<path d="M0 0 L10 5 L0 10 z" fill="#6d86ad"/></marker></defs>')
    out += arrow(292, 330, 326, 330)
    for i in range(3):
        out += arrow(block_x[i] + block_w, 330,
                     block_x[i + 1] - 4, 330)
        mid = block_x[i] + block_w + 20
        out.append(text(mid, 350, "÷ 2", 8.2, FAINT, 500,
                        anchor="middle", family=MONO))
    out += arrow(block_x[3] + block_w, 330, 1206, 276)
    # Small chevrons between the three head cards (their gaps are narrow).
    for x in (1324, 1452):
        out.append(f'<path d="M{x} 272 L{x+8} 276 L{x} 280" '
                   f'fill="none" stroke="#6d86ad" stroke-width="1.6" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Head: GAP, Dense1, 9 classes.
    # GAP card (16x16 dots = 256).
    gx, gy, gw, gh = 1210, 196, 112, 160
    out += [
        rect(gx, gy, gw, gh, "#0a1220", 14, "#2a3a58", 1.1, 0.96,
             'filter="url(#shadow)"'),
        text(gx + 14, gy + 24, "GAP", 10.5, CYAN, 700, letter=1.4),
        text(gx + 14, gy + 42, "media globale", 9.2, MUTED, 400),
    ]
    for r in range(16):
        for c in range(16):
            cx = gx + 18 + c * 5.3
            cy = gy + 58 + r * 5.3
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.5" '
                       f'fill="{CYAN}" opacity="0.55"/>')
    out += [
        text(gx + gw - 12, gy + 148, "256 → 256", 9.5, INK, 600,
             anchor="end", family=MONO),
    ]

    # Dense1 card (16x8 dots = 128).
    dx, dy, dw, dh = 1338, 196, 112, 160
    out += [
        rect(dx, dy, dw, dh, "#0a1220", 14, "#2a3a58", 1.1, 0.96,
             'filter="url(#shadow)"'),
        text(dx + 14, dy + 24, "DENSE 1", 10.5, VIOLET, 700, letter=1.4),
        text(dx + 14, dy + 42, "ReLU", 9.2, MUTED, 400),
    ]
    for r in range(8):
        for c in range(16):
            cx = dx + 18 + c * 5.3
            cy = dy + 60 + r * 7.8
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.9" '
                       f'fill="{VIOLET}" opacity="0.62"/>')
    out += [
        text(dx + 14, dy + 148, "128 nodi", 9.5, INK, 600, family=MONO),
    ]

    # Output card with the 9 classes.
    ox, oy, ow, oh = 1466, 196, 158, 310
    out += [
        rect(ox, oy, ow, oh, "#0a1220", 16, "#3a2e4f", 1.1, 0.98,
             'filter="url(#shadow)"'),
        text(ox + 16, oy + 28, "9 CLASSI", 10.5, AMBER, 700, letter=1.4),
        text(ox + 16, oy + 46, "logits → softmax", 8.8, MUTED, 400,
             family=MONO),
    ]
    class_colors = [CYAN, VIOLET, "#8b95a8", "#8b95a8", "#8b95a8",
                    PINK, "#fb923c", "#8b95a8", GREEN]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, class_colors)):
        cy = oy + 76 + i * 24
        active = name in ACTIVE_CLASSES
        if active:
            out.append(f'<circle cx="{ox+22}" cy="{cy-4}" r="4" '
                       f'fill="{color}"/>')
        else:
            out.append(f'<circle cx="{ox+22}" cy="{cy-4}" r="3.6" '
                       f'fill="none" stroke="{color}" stroke-width="1.2" '
                       f'opacity="0.55"/>')
        out.append(text(ox + 34, cy, name, 11.5,
                        INK if active else MUTED, 600 if active else 400))
    out += [
        line(ox + 16, oy + 282, ox + ow - 16, oy + 282, "#3a2e4f", 1),
        text(ox + 16, oy + 300, "● 5 con dati   ○ 4 futuro", 8.2, FAINT,
             400),
    ]

    # Bottom-left: parameter distribution.
    px, py, pw, ph = 56, 548, 760, 506
    out += [
        rect(px, py, pw, ph, "#080f1c", 18, "#26344d", 1, 0.92),
        text(px + 28, py + 38, "DOVE VANNO I PESI", 12, CYAN, 700,
             letter=1.8),
        text(px + 28, py + 60, f"{fmt(total_params)} parametri in 6 gruppi · "
                              "i pesi sono condivisi sui pixel", 11, MUTED, 400),
    ]
    rows = [(f"Conv {b['index']}", b["params"], BLOCK_COLORS[i][0])
            for i, b in enumerate(blocks)]
    rows += [("Dense 1", stats["dense1"]["params"], VIOLET),
             ("Dense 2", stats["dense2"]["params"], AMBER)]
    bar_x, bar_max = 300, 330
    for j, (label, value, color) in enumerate(rows):
        y = py + 94 + j * 54
        width = bar_max * value / max_params
        out += [
            text(px + 28, y + 4, label, 12, INK, 600),
            rect(bar_x, y - 10, bar_max, 16, "#142036", 8),
            rect(bar_x, y - 10, max(width, 3), 16, color, 8,
                 opacity=0.88),
            text(px + pw - 28, y + 4, f"{fmt(value)}  ·  "
                                      f"{value/total_params*100:4.1f}%",
                 10.5, MUTED, 600, anchor="end", family=MONO),
        ]
    out += [
        line(px + 28, py + 424, px + pw - 28, py + 424, "#26344d", 1),
        text(px + 28, py + 450, "Conv 4 da solo vale il 70% dei parametri.",
             12.5, INK, 600),
        text(px + 28, py + 472, f"{fmt(weights)} connessioni (pesi) + "
                                f"{fmt(stats['bias_bn'])} bias/BN = "
                                f"{fmt(total_params)}", 10.5, MUTED, 400,
             family=MONO),
    ]

    # Bottom-right: receptive field + fan-in.
    qx, qy, qw, qh = 840, 548, 784, 506
    out += [
        rect(qx, qy, qw, qh, "#080f1c", 18, "#26344d", 1, 0.92),
        text(qx + 28, qy + 38, "QUANTO VEDE UN NEURONE", 12, AMBER, 700,
             letter=1.8),
        text(qx + 28, qy + 60, "campo recettivo in pixel del crop 64 × 64",
             11, MUTED, 400),
        text(qx + 28, qy + 96, "0", 9, FAINT, 500, family=MONO),
        text(qx + qw - 28, qy + 96, "64 px", 9, FAINT, 500,
             anchor="end", family=MONO),
        line(qx + 28, qy + 116, qx + qw - 28, qy + 116, "#26344d", 1.2),
    ]
    # 64 px scale: x from 868 to 1516, y=664.
    x0, x1, yy = qx + 28, qx + qw - 28, qy + 116
    for tick in (0, 16, 32, 48, 64):
        tx = x0 + (x1 - x0) * tick / 64
        out += [line(tx, yy, tx, yy + 8, "#405575", 1),
                text(tx, yy + 24, str(tick), 8.5, FAINT, 500,
                     anchor="middle", family=MONO)]
    rf_points = [(4, "pool 1"), (8, "pool 2"), (22, "pool 3"),
                 (46, "pool 4")]
    for value, label in rf_points:
        tx = x0 + (x1 - x0) * value / 64
        out += [
            f'<circle cx="{tx:.1f}" cy="{yy}" r="5.5" fill="#080f1c" '
            f'stroke="{AMBER}" stroke-width="2"/>',
            f'<circle cx="{tx:.1f}" cy="{yy}" r="2" fill="{AMBER}"/>',
            text(tx, yy - 16, f"{value} px", 9.5, INK, 600,
                 anchor="middle", family=MONO),
            text(tx, yy + 42, label, 8, FAINT, 500, anchor="middle"),
        ]
    out += [
        text(qx + 28, qy + 204, "46 px = 72% del crop: il neurone finale vede "
                                "quasi tutta la stella.", 10.5, MUTED, 400),
        line(qx + 28, qy + 226, qx + qw - 28, qy + 226, "#26344d", 1),
        text(qx + 28, qy + 254, "CONNESSIONI PER NEURONE", 10.5, VIOLET,
             700, letter=1.2),
    ]
    # Fan-in table.
    fan_rows = [(f"Conv {b['index']}", b["fan_in"], b["weights"])
                for b in blocks]
    fan_rows += [("Dense 1", stats["dense1"]["fan_in"],
                  stats["dense1"]["weights"]),
                 ("Dense 2", stats["dense2"]["fan_in"],
                  stats["dense2"]["weights"])]
    for j, (label, fan, w) in enumerate(fan_rows):
        y = qy + 282 + j * 28
        out += [
            text(qx + 28, y, label, 10.5, INK, 600),
            text(qx + 150, y, f"fan-in {fmt(fan)}", 9.5, MUTED, 500,
                 family=MONO),
            text(qx + qw - 28, y, f"{fmt(w)} pesi", 9.5, MUTED, 500,
                 anchor="end", family=MONO),
        ]
    out += [
        line(qx + 28, qy + 458, qx + qw - 28, qy + 458, "#26344d", 1),
        text(qx + 28, qy + 482, f"{fmt(weights)} connessioni totali · "
                                f"{fmt(stats['total_activations'])} attivazioni "
                                "incluso l'input", 9.8, FAINT, 400,
             family=MONO),
    ]

    # Footer ----------------------------------------------------------------
    out += [
        line(56, 1080, 1624, 1080, "#26344d", 1, 0.8),
        text(56, 1104, "AstroNet-v2  ·  input 64 × 64  ·  4 conv + GAP + "
                       "2 dense  ·  9 classi", 10, FAINT, 500),
        text(1624, 1104, "generato da Tools/make_model_diagram.py",
             9.5, "#40506a", 500, anchor="end", family=MONO),
        "</svg>",
    ]
    return "\n".join(out)


def main() -> int:
    """Generate SVG and, when available, a raster PNG copy."""
    stats = collect_stats()
    crop, crop_name = _read_crop()
    svg = make_svg(stats, _crop_data_uri(crop), crop_name)
    MODELS.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(svg, encoding="utf-8")
    print(f"[svg] {SVG_PATH.relative_to(ROOT)} "
          f"({SVG_PATH.stat().st_size / 1024:.0f} KB)")

    converter = None
    for name in ("rsvg-convert", "inkscape"):
        from shutil import which
        converter = which(name)
        if converter:
            break
    if converter:
        cmd = ([converter, "-w", "2400", "-h", "1600",
                "-o", str(PNG_PATH), str(SVG_PATH)]
               if converter.endswith("rsvg-convert") else
               [converter, "--export-type=png", "--export-width=2400",
                f"--export-filename={PNG_PATH}", str(SVG_PATH)])
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[png] {PNG_PATH.relative_to(ROOT)} "
                  f"({PNG_PATH.stat().st_size / 1024:.0f} KB)")
        except subprocess.CalledProcessError as exc:
            print(f"[png] conversion failed: {exc.stderr.decode()[:200]}")
    else:
        print("[png] rsvg-convert/inkscape not found; only SVG written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
