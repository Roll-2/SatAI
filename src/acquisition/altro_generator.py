"""Synthetic generator for the "Altro" (out-of-distribution) class (Task 5.2).

The "Altro" class is NOT optional (spec 5.2). Composition implemented here:

* **40% amateur-style PSFs** — synthesized telescope PSFs (Celestron/Meade
  style: 4-vane spider spikes, defocused donuts, chromatic fringes) plus
  any real images the user drops in ``data/fits_raw/altro/``.
* **20% satellite-like textures** — grayscale, satellite-imagery-like
  synthetic textures (streets/fields/city blocks), no point sources.
* **20% pure Gaussian noise**.
* **20% galaxy cutouts** — real Legacy Survey cutouts downloaded by
  :mod:`src.acquisition.galaxy_cutouts` (no point-like stars).

Everything is saved as ``.npy`` crops directly into the staging area.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.constants import CROP_SIZE, CROPS_STAGING_DIR, RAND_SEED

__all__ = ["generate_altro_crops"]


# ---------------------------------------------------------------------------
# 40% — amateur-style PSF crops
# ---------------------------------------------------------------------------
def _amateur_psf(rng: random.Random) -> np.ndarray:
    """Synthesize an amateur-telescope PSF crop (4-vane spider + defocus)."""
    size = CROP_SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0 - 0.5
    r = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)

    img = np.zeros((size, size), dtype=np.float32)

    # Gaussian core with random defocus (donut).
    sigma_core = rng.uniform(1.2, 3.5)
    img += np.exp(-0.5 * (r / sigma_core) ** 2)
    if rng.random() < 0.35:                       # defocused donut
        donut_r = rng.uniform(4.0, 9.0)
        img += 0.6 * np.exp(-0.5 * ((r - donut_r) / 1.2) ** 2)

    # 4 diffraction spikes from the spider vanes, random angle.
    vane_angle = rng.uniform(0.0, np.pi / 4)
    for k in range(4):
        d = np.abs(np.sin(theta - vane_angle - k * np.pi / 2))
        spike_len = rng.uniform(12.0, 30.0)
        spike = np.exp(-0.5 * (d / 0.02) ** 2) * (r < spike_len) \
            * np.exp(-r / spike_len)
        img += 0.8 * spike

    # Chromatic fringe (slight ring modulation) + noise.
    img += 0.05 * np.sin(r * 2.5 + rng.uniform(0, 6))
    img += rng.gauss(0.0, 0.02) * np.ones_like(img)
    img += np.random.default_rng(rng.randint(0, 2**32)).normal(0, 0.01, img.shape)

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 20% — satellite-like textures (streets / fields / blocks)
# ---------------------------------------------------------------------------
def _satellite_texture(rng: random.Random) -> np.ndarray:
    """Synthesize a grayscale satellite-imagery-like texture crop."""
    size = CROP_SIZE
    rng_np = np.random.default_rng(rng.randint(0, 2**32))
    img = rng_np.uniform(0.25, 0.5, (size, size)).astype(np.float32)

    # Rectangular "fields" with random brightness.
    for _ in range(rng.randint(4, 10)):
        x0, y0 = rng.randint(0, size - 8), rng.randint(0, size - 8)
        w, h = rng.randint(6, 24), rng.randint(6, 24)
        img[y0:y0 + h, x0:x0 + w] += rng_np.uniform(-0.15, 0.15)

    # "Streets": bright straight lines.
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.5:                    # horizontal / vertical
            y0 = rng.randint(4, size - 4)
            width = rng.randint(1, 3)
            img[y0:y0 + width, :] += 0.3
        else:
            x0 = rng.randint(4, size - 4)
            width = rng.randint(1, 3)
            img[:, x0:x0 + width] += 0.3

    img += rng_np.normal(0, 0.03, img.shape).astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 20% — pure Gaussian noise
# ---------------------------------------------------------------------------
def _gaussian_noise(rng: random.Random) -> np.ndarray:
    """Pure Gaussian noise crop."""
    rng_np = np.random.default_rng(rng.randint(0, 2**32))
    img = rng_np.normal(0.5, rng.uniform(0.05, 0.25), (CROP_SIZE, CROP_SIZE))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 20% — real galaxy cutouts (Legacy Survey)
# ---------------------------------------------------------------------------
def _galaxy_crops(galaxy_dir: Path, n_needed: int, rng: random.Random,
                  find_blobs) -> list[np.ndarray]:
    """Extract 64x64 crops from downloaded Legacy Survey galaxy FITS."""
    from src.preprocessing.loader import load_fits
    from src.preprocessing.crop_generator import extract_crops

    crops: list[np.ndarray] = []
    if not galaxy_dir.exists():
        return crops
    files = sorted(galaxy_dir.glob("*.fits"))
    rng.shuffle(files)

    for fpath in files:
        if len(crops) >= n_needed:
            break
        try:
            img = load_fits(fpath)
            positions = find_blobs(img)
            got = extract_crops(img, positions, max_crops=4)
            crops.extend(got)
        except Exception:  # noqa: BLE001 — skip corrupt cutouts
            continue
    return crops


def generate_altro_crops(n_crops: int,
                         out_dir: Path | None = None,
                         seed: int = RAND_SEED) -> dict[str, int]:
    """Generate the full "Altro" crop set per spec 5.2 proportions.

    Args:
        n_crops: total number of crops to produce.
        out_dir: destination staging directory.
        seed: RNG seed for reproducibility.

    Returns:
        Counts per sub-source:
        ``{"amateur": ..., "satellite": ..., "noise": ..., "galaxy": ...}``.
    """
    out_path = out_dir or (CROPS_STAGING_DIR / "altro")
    out_path.mkdir(parents=True, exist_ok=True)

    from src.preprocessing.star_finder import find_stars_png

    rng = random.Random(seed)
    n_galaxy = int(n_crops * 0.20)
    n_noise = int(n_crops * 0.20)
    n_sat = int(n_crops * 0.20)
    n_amateur = n_crops - n_galaxy - n_noise - n_sat   # remainder (40%)

    counts = {"amateur": 0, "satellite": 0, "noise": 0, "galaxy": 0}

    # Real galaxy cutouts first (they may be limited by downloads).
    galaxy_crops = _galaxy_crops(FITS_RAW_DIR / "altro" / "galaxies",
                                 n_galaxy, rng, find_stars_png)
    for i, crop in enumerate(galaxy_crops):
        np.save(out_path / f"alt_gal_{i:04d}.npy", crop)
    counts["galaxy"] = len(galaxy_crops)

    # Fill any galaxy shortfall with amateur-style PSFs (spec 40% share).
    n_amateur += max(0, n_galaxy - len(galaxy_crops))

    generators = {
        "alt_ama": (_amateur_psf, n_amateur),
        "alt_sat": (_satellite_texture, n_sat),
        "alt_noi": (_gaussian_noise, n_noise),
    }
    for prefix, (fn, count) in generators.items():
        for i in tqdm(range(count), desc=f"Altro/{prefix}", unit="crop"):
            np.save(out_path / f"{prefix}_{i:04d}.npy", fn(rng))
            counts[{"alt_ama": "amateur", "alt_sat": "satellite",
                    "alt_noi": "noise"}[prefix]] += 1

    return counts
