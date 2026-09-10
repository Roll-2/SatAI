"""Data augmentation with rigorous per-class rules (Task 6).

Each class has a fixed policy dict (rotations, flips, shift amplitude,
noise sigma, contrast-stretch probability) taken verbatim from the spec
table. Rotations are only ever multiples of 90 degrees (``np.rot90``,
lossless) or multiples of 60 degrees (``scipy.ndimage.rotate``, required
by the hexagonal JWST/Roman/Euclid geometry) — Appendix A forbids
anything else.

Physical motivations encoded below:
* **Roman**  — curved asymmetric arms: flips destroy the signature.
* **Spitzer** — undersampled, pixel-locked sensor: no rotations.
* **Gaia**   — unidirectional TDI scanning: only 0/180 degrees.
* **WISE**   — circular ghosting halo: rotations irrelevant, shift yes.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from src.utils.constants import CLASS_NAMES, CROP_SIZE

__all__ = ["POLICIES", "augment", "Augmenter"]

#: class_id -> policy (Task 6 table, applied "senza eccezioni").
POLICIES: dict[int, dict] = {
    0: dict(name=CLASS_NAMES[0], rotations=[0, 90, 180, 270], flip=True,
            shift=2, noise_sigma=0.01, contrast_prob=0.50),
    1: dict(name=CLASS_NAMES[1], rotations=[0, 60, 120, 180, 240, 300],
            flip=True, shift=2, noise_sigma=0.01, contrast_prob=0.50),
    2: dict(name=CLASS_NAMES[2], rotations=[0, 60, 120, 180, 240, 300],
            flip=False, shift=2, noise_sigma=0.01, contrast_prob=0.50),
    3: dict(name=CLASS_NAMES[3], rotations=[0, 60, 120, 180, 240, 300],
            flip=True, shift=2, noise_sigma=0.01, contrast_prob=0.50),
    4: dict(name=CLASS_NAMES[4], rotations=[0, 90, 180, 270], flip=True,
            shift=2, noise_sigma=0.02, contrast_prob=0.50),
    5: dict(name=CLASS_NAMES[5], rotations=[0], flip=False,
            shift=1, noise_sigma=0.01, contrast_prob=0.30),
    6: dict(name=CLASS_NAMES[6], rotations=[0], flip=False,
            shift=1, noise_sigma=0.01, contrast_prob=0.30),
    7: dict(name=CLASS_NAMES[7], rotations=[0, 180], flip=False,
            shift=1, noise_sigma=0.01, contrast_prob=0.30),
    8: dict(name=CLASS_NAMES[8], rotations=[0, 90, 180, 270], flip=True,
            shift=2, noise_sigma=0.02, contrast_prob=0.50),
}

_LOSSLESS_ANGLES = {0, 90, 180, 270}   # handled by np.rot90


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate by ``angle`` degrees, lossless for multiples of 90."""
    if angle % 90 == 0 and angle in _LOSSLESS_ANGLES:
        return np.rot90(img, k=angle // 90).copy()
    # 60-degree multiples (JWST / Roman / Euclid hexagonal symmetry).
    return scipy_rotate(img, angle=angle, reshape=False, order=1,
                        mode="nearest").astype(np.float32)


def augment(crop: np.ndarray, class_id: int,
            rng: random.Random | None = None) -> np.ndarray:
    """Apply the class-``class_id`` augmentation policy to one crop.

    Args:
        crop: float32 64x64 image in [0, 1].
        class_id: class index (0-8), selects the policy.
        rng: optional ``random.Random`` (determinism in tests).

    Returns:
        New augmented float32 crop in [0, 1].
    """
    rng = rng or random.Random()
    policy = POLICIES[class_id]
    img = np.asarray(crop, dtype=np.float32).copy()

    # 1. Rotation (allowed set only).
    angle = rng.choice(policy["rotations"])
    if angle != 0:
        img = _rotate(img, angle)

    # 2. Horizontal / vertical flip (only where physically allowed).
    if policy["flip"]:
        if rng.random() < 0.5:
            img = np.fliplr(img).copy()
        if rng.random() < 0.5:
            img = np.flipud(img).copy()

    # 3. Random integer shift (np.roll, no interpolation).
    max_shift = policy["shift"]
    dx, dy = rng.randint(-max_shift, max_shift), rng.randint(-max_shift, max_shift)
    if dx or dy:
        img = np.roll(img, shift=(dy, dx), axis=(0, 1))

    # 4. Contrast stretch with the class probability.
    if rng.random() < policy["contrast_prob"]:
        lo, hi = np.percentile(img, [1.0, 99.0])
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

    # 5. Gaussian noise with the class sigma.
    sigma = policy["noise_sigma"]
    noise = np.random.default_rng(rng.randint(0, 2**32)) \
        .normal(0.0, sigma, img.shape).astype(np.float32)
    img = img + noise

    return np.clip(img, 0.0, 1.0).astype(np.float32)


class Augmenter:
    """Callable wrapper around :func:`augment` for use in Datasets."""

    def __init__(self, enabled: bool = True, seed: int | None = None):
        """Create an augmenter.

        Args:
            enabled: if False, ``__call__`` returns the crop untouched.
            seed: optional base seed for deterministic runs.
        """
        self.enabled = enabled
        self._seed = seed
        self._rng = random.Random(seed)

    def __call__(self, crop: np.ndarray, class_id: int) -> np.ndarray:
        """Augment (or pass through) one crop."""
        if not self.enabled:
            return np.asarray(crop, dtype=np.float32)
        return augment(crop, class_id, rng=self._rng)
