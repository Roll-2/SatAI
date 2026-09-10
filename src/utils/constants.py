"""Project-wide constants: paths, classes, and pipeline parameters.

Every module imports from here so there is a single source of truth
(and no circular dependencies).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
FITS_RAW_DIR: Path = DATA_DIR / "fits_raw"
CROPS_DIR: Path = DATA_DIR / "crops"
TRAIN_DIR: Path = CROPS_DIR / "train"
VAL_DIR: Path = CROPS_DIR / "val"
TEST_DIR: Path = CROPS_DIR / "test"
PNG_SAMPLES_DIR: Path = DATA_DIR / "png_samples"
MODELS_DIR: Path = PROJECT_ROOT / "models_saved"

#: Temporary staging area for crops before the stratified split.
CROPS_STAGING_DIR: Path = CROPS_DIR / "staging"

# --------------------------------------------------------------------------
# The 9 classes (Appendice B of the spec)
# --------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Hubble",   # 0
    "JWST",     # 1
    "Roman",    # 2
    "Euclid",   # 3
    "Rubin",    # 4
    "Spitzer",  # 5
    "WISE",     # 6
    "Gaia",     # 7
    "Altro",    # 8
]

#: Folder name (inside data/fits_raw/ and data/crops/<split>/) for each class.
CLASS_FOLDERS: list[str] = [name.lower() for name in CLASS_NAMES]

#: Pretty names used by the CLI output (Task 11).
DISPLAY_NAMES: list[str] = [
    "Hubble Space Telescope",
    "James Webb Space Telescope",
    "Nancy Grace Roman Space Telescope",
    "Euclid Space Telescope",
    "Vera C. Rubin Observatory",
    "Spitzer Space Telescope",
    "Wide-field Infrared Survey Explorer",
    "Gaia Space Telescope",
    "Telescopio sconosciuto",
]

NUM_CLASSES: int = len(CLASS_NAMES)          # 9
ALTRO_CLASS_ID: int = 8                      # "Altro" / out-of-distribution

# --------------------------------------------------------------------------
# Crop / image geometry (Task 4.3)
# --------------------------------------------------------------------------
CROP_SIZE: int = 64                          # crops are always 64x64 pixels
CROP_CENTER: int = 32                        # peak pixel must land at (32, 32)
MAX_CROPS_PER_IMAGE: int = 20                # max crops extracted per source FITS
TOP_CROPS_FOR_INFERENCE: int = 10            # multi-crop voting (Task 10)
MIN_CROPS_FOR_INFERENCE: int = 3             # below this -> "Immagine insufficiente"
EDGE_MARGIN: int = CROP_CENTER               # peak must be >= 32 px from any border

# --------------------------------------------------------------------------
# Preprocessing (Task 3)
# --------------------------------------------------------------------------
SKY_PERCENTILE_CLIP: float = 99.5            # values above the 99.5th percentile -> 1.0
SATURATION_THRESHOLD: float = 0.99           # pixels >= 0.99 are considered saturated
MEDIAN_FILTER_SIZE: int = 3                  # anti-saturation kernel (3x3)

# --------------------------------------------------------------------------
# Star finding (Task 4)
# --------------------------------------------------------------------------
DAO_FWHM: float = 2.0                        # DAOStarFinder fwhm
DAO_THRESHOLD_SIGMA: float = 5.0             # threshold = 5.0 * sky_std
DAO_DROP_BRIGHTEST: int = 5                  # drop the 5 brightest (too saturated)
DAO_DROP_FAINT: int = 10                     # drop the 10 faintest (too weak)
DAO_KEEP: int = 20                           # keep the next 20 (intermediate mag)

BLOB_MIN_SIGMA: int = 1                      # skimage.feature.blob_log params
BLOB_MAX_SIGMA: int = 5
BLOB_NUM_SIGMA: int = 10
BLOB_THRESHOLD: float = 0.1
BLOB_TOP_K: int = 10                         # top 10 blobs for PNG inference

# --------------------------------------------------------------------------
# Training (Task 8)
# --------------------------------------------------------------------------
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
ADAM_EPS: float = 1e-8
WEIGHT_DECAY: float = 1e-5
MAX_EPOCHS: int = 40
EARLY_STOPPING_PATIENCE: int = 7
LR_SCHEDULER_FACTOR: float = 0.1
LR_SCHEDULER_PATIENCE: int = 5
NUM_WORKERS: int = 8                         # M4 Pro performance cores
PIN_MEMORY: bool = False                     # irrelevant on MPS, useful on CUDA

BEST_MODEL_PATH: Path = MODELS_DIR / "astro_net_v2_best.pt"
LAST_MODEL_PATH: Path = MODELS_DIR / "astro_net_v2_last.pt"

# --------------------------------------------------------------------------
# Inference decision thresholds (Tasks 9.3 / 10)
# --------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.55           # below -> "Non determinabile"
RAND_SEED: int = 42
