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
DATA_DIR: Path = PROJECT_ROOT / "Data"
FITS_RAW_DIR: Path = DATA_DIR / "fits_raw"
CROPS_DIR: Path = DATA_DIR / "crops"
TRAIN_DIR: Path = CROPS_DIR / "train"
VAL_DIR: Path = CROPS_DIR / "val"
TEST_DIR: Path = CROPS_DIR / "test"
PNG_SAMPLES_DIR: Path = DATA_DIR / "png_samples"
MODELS_DIR: Path = PROJECT_ROOT / "Models"

#: Temporary staging area for crops before the stratified split.
CROPS_STAGING_DIR: Path = CROPS_DIR / "staging"

#: Uploads from the desktop UI (only the current photo is kept).
UI_UPLOADS_DIR: Path = DATA_DIR / "ui_uploads"

# --------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------
#: PyTorch checkpoint (fallback backend, used on non-Apple hardware).
BEST_MODEL_PATH: Path = MODELS_DIR / "astro_net_v2_best.pt"
LAST_MODEL_PATH: Path = MODELS_DIR / "astro_net_v2_last.pt"

#: CoreML copy of the SAME model, exported for the Apple Neural Engine.
CORE_ML_MODEL_PATH: Path = MODELS_DIR / "AstroNetV2.mlpackage"
CORE_ML_INPUT_NAME: str = "image"
CORE_ML_OUTPUT_NAME: str = "logits"

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

#: Folder name (inside Data/fits_raw/ and Data/crops/<split>/) for each class.
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
INFERENCE_FWHM_MAX: float = 30.0             # relaxed ceiling: extended objects
                                             # (galassie) raggiungono il modello
                                             # e vengono classificate "Altro";
                                             # il training resta severo (14 px)
INFERENCE_MIN_SNR: float = 5.0               # SNR ridotto all'inferenza: su una
                                             # galassia la std del crop E' la
                                             # sorgente, il gate 10-sigma da
                                             # stella puntiforme la eliminerebbe
#: Gate OOD sull'IMMAGINE intera: mediana delle SNR dei crop. Misurato
#: (14/09): junk da rumore/textura 9.5-13, p50 reali >= 23.7 su TUTTE le
#: 5 classi. 16 sta nel mezzo: scarta le foto senza sorgenti reali invece
#: di etichettarle con confidenza 0.77+ (bug: foto satellite -> JWST 99.7%).
OOD_MEDIAN_SNR_MIN: float = 16.0

# --------------------------------------------------------------------------
# Preprocessing (Task 3)
# --------------------------------------------------------------------------
SKY_PERCENTILE_CLIP: float = 99.5            # values above the 99.5th percentile -> 1.0
SATURATION_THRESHOLD: float = 0.99           # pixels >= 0.99 are considered saturated
MEDIAN_FILTER_SIZE: int = 3                  # anti-saturation kernel (3x3)

#: IRSA bandpass names accepted per mission (SIA column
#: ``energy_bandpassname``): Spitzer class = IRAC channel 1 (3.6 um) only,
#: WISE class = W1 (3.4 um) only. All other channels change the PSF
#: shape and would poison the class.
IRSA_BANDPASS: dict[str, set[str]] = {
    "spitzer": {"IRAC1"},
    "wise": {"W1"},
}

# --------------------------------------------------------------------------
# Star finding (Task 4)
# --------------------------------------------------------------------------
DAO_FWHM: float = 2.0                        # DAOStarFinder fwhm (default)
DAO_THRESHOLD_SIGMA: float = 5.0             # threshold = 5.0 * sky_std
#: Fix 14/09 notte: non buttare più via le stelle brillanti. La regola
#: "drop 5 brightest" nasce PRIMA dell'anti-saturatore: il desaturate
#: ripara i nuclei saturi, e le stelle brillanti sono l'unica parte del
#: dataset in cui i bracci di diffrazione (4 vs 6 linee) sono ben visibili
#: (misurato: contrasto bracci 2.3-2.5 sigma vs 1.6-2.0 delle deboli).
#: Senza di esse il modello non puo' imparare la firma fisica HST-vs-JWST.
DAO_DROP_BRIGHTEST: int = 0                  # keep the brightest: spikes live there
DAO_DROP_FAINT: int = 10                     # drop the 10 faintest (too weak)
DAO_KEEP: int = 20                           # keep the next 20 (intermediate mag)

#: Per-telescope finder parameters. The default (fwhm=2.0, 5 sigma) is
#: wrong for most archives: WISE undersamples a ~6 px PSF, JWST needs a
#: 10-sigma floor to survive crowded NIRCam fields. ``find_stars_fits``
#: looks the telescope up here (fallback: DAO_FWHM / DAO_THRESHOLD_SIGMA).
FINDER_PARAMS: dict[str, dict[str, float]] = {
    "hubble":    {"fwhm": 2.0, "threshold_sigma": 5.0},
    "jwst":      {"fwhm": 1.5, "threshold_sigma": 10.0},   # NIRCam
    "jwst_miri": {"fwhm": 2.5, "threshold_sigma": 8.0},    # MIRI (undersampled)
    "wise":      {"fwhm": 6.0, "threshold_sigma": 3.0},
    "spitzer":   {"fwhm": 2.0, "threshold_sigma": 5.0},    # IRAC ch. 1
    "euclid":    {"fwhm": 2.0, "threshold_sigma": 5.0},
    "gaia":      {"fwhm": 3.0, "threshold_sigma": 5.0},
    "roman":     {"fwhm": 2.0, "threshold_sigma": 5.0},
    "rubin":     {"fwhm": 3.0, "threshold_sigma": 5.0},
}

# --------------------------------------------------------------------------
# Crop quality gates (mandatory before a crop is saved)
# --------------------------------------------------------------------------
#: Minimum significance of the central peak vs the sigma-clipped sky noise
#: of the crop itself. Below ~10 sigma no diffraction structure survives.
MIN_CROP_SNR: float = 10.0

#: Accepted FWHM range of the source re-centered in the crop (seconds-moment
#: estimate at half maximum). Below ~1.5 px the "source" is a hot pixel or
#: a cosmic-ray glitch; above ~14 px it is an extended object (galaxy,
#: cluster core, nebula — the staged junk measured 29-39 px) — never a PSF.
#: Real PSFs span 1.5-8 px (WISE W1 is the widest at ~6 px); the upper
#: bound keeps margin for saturated-core broadening.
CROP_FWHM_MIN: float = 1.5
CROP_FWHM_MAX: float = 14.0

#: A crop whose brightest pixel exceeds this value is dominated by
#: saturation remnants (the anti-saturator already repairs pixels that
#: were flagged in the full frame). Kept for reporting/inference tools.
CROP_SATURATION_MAX: float = 0.95

BLOB_MIN_SIGMA: int = 1                      # skimage.feature.blob_log params
BLOB_MAX_SIGMA: int = 12                     # include galassie (sigma ~7-13 px)
BLOB_NUM_SIGMA: int = 22
BLOB_THRESHOLD: float = 0.1
#: SNR minimo di un blob contro il cielo sigma-clipped DELL'IMMAGINE: senza
# questo gate il rumore/texture di una foto qualunque produce blob che il
# modello etichetta con confidenza 0.76+ (bug 14/09: foto satellite -> JWST
# al 99.7%). I crop reali partono da ~10 sigma sul proprio cielo.
BLOB_SNR_MIN: float = 8.0
#: Raggio di deduplica dei blob (px): un solo rilevamento per plateau — i
#: voti multi-crop devono arrivare da SORGENTI DIVERSE, non dallo stesso
#: blob ri-trovato a scale sigma diverse.
BLOB_DEDUP_RADIUS: float = 20.0
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
NUM_WORKERS: int = 4                         # M4 Pro: 4 workers bilanciano memoria/Dataloader throughput
PIN_MEMORY: bool = False                     # irrelevant on MPS, useful on CUDA
#: Libera la cache MPS dopo ogni epoca di training per evitare accumulo
#: di memoria GPU (MPS) durante training prolungato — il modello è piccolo
#: (1.6 MB) ma i DataLoader con worker possono accumulare buffer GPU.

# --------------------------------------------------------------------------
# Inference decision thresholds (Tasks 9.3 / 10)
# --------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.55           # below -> "Non determinabile"
RAND_SEED: int = 42
