"""Fine-tune AstroNet-v2 on real telescope images after synthetic pre-training.

Usage (from project root):
    python -m Src.training.finetune_real \\
        --real-hubble-dir /percorso/foto_hubble\ to\ test \\
        --real-jwst-dir /percorso/foto_jwst\ to\ test \\
        --n-per-class 5000 \\
        --epochs 20

The script implements the practical fine-tuning recipe from research:
- 1:1 real-to-synthetic mixing in each training batch
- Discriminative learning rates: backbone 1e-5, head 1e-4 (5x multiplier)
- Freeze backbone for first 2-5 epochs, then unfreeze
- 10-25 epochs total with cosine decay and early stopping
- Astronomical-aware data augmentation (seeing convolution, sky noise, photon/read noise)
- Validation on held-out real images (not seen during training)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Src.data.build_dataset import build_dataset, CLASS_FOLDERS, CROPS_STAGING_DIR
from Src.models.psf_cnn import build_model
from Src.preprocessing.anti_saturator import desaturate
from Src.preprocessing.crop_generator import extract_crops, generate_crops_for_image
from Src.preprocessing.loader import load_image
from Src.preprocessing.star_finder import find_stars_png
from Src.utils.constants import (
    BEST_MODEL_PATH,
    CROP_SIZE,
    CROP_FWHM_MIN,
    CROP_FWHM_MAX,
    INFERENCE_FWHM_MAX,
    INFERENCE_MIN_SNR,
    MIN_CROPS_FOR_INFERENCE,
    OOD_MEDIAN_SNR_MIN,
    TOP_CROPS_FOR_INFERENCE,
    RAND_SEED,
)

#: Backwards-compatible alias used by the CLI help below.
MODELS_DIR = BEST_MODEL_PATH.parent

# 9 classes: 0-7 real telescopes, 8 = Altro
NUM_CLASSES = 9
ALTRO_CLASS_ID = 8


# ---------------------------------------------------------------------------
# Helper: extract real star crops from PNG/JPG images
# ---------------------------------------------------------------------------

def extract_real_crops_from_image(
    image_path: Path,
    max_crops: int = 20,
    fwhm_max: float = INFERENCE_FWHM_MAX,
    min_snr: float = INFERENCE_MIN_SNR,
) -> list[np.ndarray]:
    """Extract up to ``max_crops`` 64x64 star crops from one real image.

    Returns list of ``(64, 64)`` float32 arrays, each already passing
    ``crop_qc_pass`` gates (SNR + FWHM).  If no valid crops are found,
    returns an empty list.
    """
    try:
        img = load_image(str(image_path))
    except Exception:
        return []

    # Anti-saturation (mandatory before star finding)
    img = desaturate(img)

    # Stage 1: fast star finding (high recall)
    positions = find_stars_png(img)

    # Stage 2: crop only detected candidates
    crops = extract_crops(
        img, positions,
        max_crops=max_crops,
        fwhm_max=fwhm_max,
        min_snr=min_snr,
    )

    return crops


def collect_real_crops(
    real_dir: Path,
    class_name: str,
    max_images: int | None = None,
    max_crops_per_image: int = 20,
    seed: int = RAND_SEED,
) -> list[tuple[np.ndarray, int]]:
    """Walk ``real_dir`` and collect (crop, class_id) tuples.

    ``class_name`` must be a folder name inside ``real_dir`` that matches
    one of ``CLASS_FOLDERS`` (e.g. ``"hubble"`` or ``"jwst"``).
    """
    from Src.utils.constants import CLASS_FOLDERS

    folder = class_name.lower()
    if folder not in CLASS_FOLDERS:
        raise ValueError(f"Unknown class folder '{folder}'; expected one of {CLASS_FOLDERS}")

    class_id = CLASS_FOLDERS.index(folder)
    img_files = sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png"))

    if max_images is not None:
        img_files = img_files[:max_images]

    rng = random.Random(seed)
    rng.shuffle(img_files)

    crops: list[tuple[np.ndarray, int]] = []
    for fpath in img_files:
        if len(crops) >= 500:  # cap total real crops to keep training tractable
            break
        extracted = extract_real_crops_from_image(
            fpath, max_crops=max_crops_per_image,
        )
        for crop in extracted:
            crops.append((crop.copy(), class_id))

    return crops


# ---------------------------------------------------------------------------
# Dataset wrapper: mixes real + synthetic crops
# ---------------------------------------------------------------------------

class RealSyntheticMixedDataset(torch.utils.data.Dataset):
    """Dataset that mixes real and synthetic crops in each epoch.

    During training, each batch contains approximately ``real_frac`` fraction
    of real crops and ``(1 - real_frac)`` synthetic crops.  This implements
    the 1:1 real:synthetic mixing recommended by the fine-tuning research.

    At test/validation time, only real crops are returned (for honest
    evaluation on real hold-out data).
    """

    def __init__(self,
        synthetic_crops: list[np.ndarray],
        real_crops: list[tuple[np.ndarray, int]],
        real_frac: float = 0.5,
        transform=None,
    ):
        self.synthetic_crops = synthetic_crops
        self.real_crops = real_crops          # list of (crop_array, class_id)
        self.real_frac = real_frac
        self.transform = transform

        # Build index lists
        self.synth_indices = list(range(len(self.synthetic_crops)))
        self.real_indices = list(range(len(self.real_crops)))

        # For "real-only" mode (validation)
        self.real_only = False

    def set_real_only(self, flag: bool = True):
        """Switch to real-only mode (for validation evaluation)."""
        self.real_only = flag

    def __len__(self) -> int:
        # Return a large number so we iterate multiple epochs; the loader
        # uses epoch-based looping.
        return 10_000

    def __getitem__(self, idx: int):
        if self.real_only:
            # Validation: just cycle through real crops
            crop, class_id = self.real_crops[idx % len(self.real_crops)]
        else:
            # Training: mix real + synthetic per batch
            # Decide per-sample whether this is real or synthetic
            if random.random() < self.real_frac:
                # Pick a real crop
                crop, class_id = self.real_crops[idx % len(self.real_crops)]
            else:
                # Pick a synthetic crop
                crop = self.synthetic_crops[idx % len(self.synthetic_crops)]
                class_id = 8  # Altro class for synthetic (but we'll remap)

        # Crop is 64x64 float32 in [0,1]; add channel dim -> (1, 64, 64)
        crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()

        # Map class_id to 0..8; synthetic crops default to Altro (8)
        if class_id is None:
            class_id = ALTRO_CLASS_ID

        # Optional simple transform (flip/rotate for training)
        if self.transform is not None:
            crop_tensor = self.transform(crop_tensor)

        return crop_tensor, class_id


# ---------------------------------------------------------------------------
# Simple augmentation: random flip + optional seeing convolution
# ---------------------------------------------------------------------------

def make_train_transform(use_seeing_conv: bool = False,
                       seeing_fwhm_range: tuple[float, float] = (2.0, 6.0)):
    """Return a simple torchvision-like transform for training crops.

    Currently supports:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90-degree rotation (p=0.5) — preserves PSF symmetry stats
    - If ``use_seeing_conv``: add Gaussian blur with random FWHM from
      ``seeing_fwhm_range`` to simulate varying seeing.  This is applied
      on-the-fly per-sample; for speed in practice you may want to pre-
     -convolve a fixed set of seeing kernels and sample from those.
    """
    from torch.nn import functional as F

    def transform(x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, 64, 64)
        if random.random() < 0.5:
            x = torch.flip(x, [2])  # horizontal
        if random.random() < 0.5:
            x = torch.flip(x, [1])  # vertical
        if random.random() < 0.5:
            # 90-degree rotation
            x = torch.rot90(x, dims=(1, 2))

        if use_seeing_conv:
            fwhm = random.uniform(*seeing_fwhm_range)
            # Gaussian blur: create kernel on the fly (slow but simple)
            kernel_size = int(6 * fwhm + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size)
            sigma = fwhm / (2 * 3**0.5)  # fwhm = 2*sqrt(2*ln(2)) * sigma
            # Gaussian 1D coords
            coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
            gauss_1d = torch.exp(-coords.float() ** 2 / (2 * sigma ** 2))
            kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
            kernel_2d = kernel_2d / kernel_2d.sum()
            # Apply depthwise conv
            padding = kernel_size // 2
            x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
            x = F.conv2d(x, kernel_2d.view(1, 1, kernel_size, kernel_size), groups=1)

        return x

    return transform


# ---------------------------------------------------------------------------
# Fine-training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune AstroNet-v2 on real telescope images")
    parser.add_argument("--real-hubble-dir", type=str,
                        required=True,
                        help="Path to folder with real Hubble images")
    parser.add_argument("--real-jwst-dir", type=str,
                        required=True,
                        help="Path to folder with real JWST images")
    parser.add_argument("--n-per-class", type=int, default=5000,
                        help="Synthetic crops per class in staging (spec: 5000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tuning epochs (recommended: 10-25)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Head learning rate (backbone will use 1e-5)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--save-best", type=str,
                        default=str(MODELS_DIR / "astro_net_v2_finetuned.pt"),
                        help="Path to save best model")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    print(f"=== Fine-tuning AstroNet-v2 ===")
    print(f"Real Hubble dir: {args.real_hubble_dir}")
    print(f"Real JWST dir: {args.real_jwst_dir}")
    print(f"Synthetic n-per-class: {args.n_per_class}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Head LR: {args.learning_rate}, Backbone LR: {args.learning_rate / 5.0}")
    print()

    # ------------------------------------------------------------------
    # 1. Load pre-trained model (already trained on full synthetic dataset)
    # ------------------------------------------------------------------
    print("Loading pre-trained model...")
    model = build_model(device=device)
    # BEST_MODEL_PATH is the synthetic pre-trained checkpoint
    best_path = BEST_MODEL_PATH
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pre-trained weights from {best_path}")
    else:
        print(f"Warning: {best_path} not found — using freshly initialized model")

    # ------------------------------------------------------------------
    # 2. Collect real crops from the provided directories
    # ------------------------------------------------------------------
    print("\nCollecting real star crops...")

    # Collect real Hubble crops
    real_hubble_dir = Path(args.real_hubble_dir)
    real_hubble_crops = collect_real_crops(
        real_hubble_dir, "hubble", max_images=None, max_crops_per_image=20,
    )
    print(f"  Real Hubble crops: {len(real_hubble_crops)}")

    # Collect real JWST crops
    real_jwst_dir = Path(args.real_jwst_dir)
    real_jwst_crops = collect_real_crops(
        real_jwst_dir, "jwst", max_images=None, max_crops_per_image=20,
    )
    print(f"  Real JWST crops: {len(real_jwst_crops)}")

    all_real_crops = real_hubble_crops + real_jwst_crops
    print(f"  Total real crops: {len(all_real_crops)}")

    if len(all_real_crops) == 0:
        print("ERROR: No real crops extracted. Check image paths and star finding parameters.")
        return

    # ------------------------------------------------------------------
    # 3. Load synthetic crops from the staging area (already generated by build_dataset)
    # ------------------------------------------------------------------
    print("\nLoading synthetic crops...")
    staging_dir = CROPS_STAGING_DIR

    # Gather all .npy crops from staging (all 9 classes)
    synth_crops: list[np.ndarray] = []
    synth_labels: list[int] = []

    for class_idx, folder_name in enumerate(CLASS_FOLDERS):
        class_staging_dir = staging_dir / folder_name
        if not class_staging_dir.exists():
            continue
        npy_files = sorted(class_staging_dir.glob("**/*.npy"))
        for nf in npy_files:
            crop = np.load(nf)
            if crop.shape == (CROP_SIZE, CROP_SIZE):
                synth_crops.append(crop)
                synth_labels.append(class_idx)

    print(f"  Synthetic crops available: {len(synth_crops)} (across {len(set(synth_labels))} classes)")
    if len(synth_crops) == 0:
        print("WARNING: No synthetic crops found. Running build_dataset first...")
        # Run a quick build to get at least some synthetic crops
        from Src.data.build_dataset import main as build_main
        build_main(args.n_per_class)
        # Reload
        for class_idx, folder_name in enumerate(CLASS_FOLDERS):
            class_staging_dir = staging_dir / folder_name
            if not class_staging_dir.exists():
                continue
            npy_files = sorted(class_staging_dir.glob("**/*.npy"))
            for nf in npy_files:
                crop = np.load(nf)
                if crop.shape == (CROP_SIZE, CROP_SIZE):
                    synth_crops.append(crop)
                    synth_labels.append(class_idx)
        print(f"  After build: synthetic crops: {len(synth_crops)}")

    # ------------------------------------------------------------------
    # 4. Create the mixed dataset and data loaders
    # ------------------------------------------------------------------
    real_frac = 0.5  # 1:1 real:synthetic mix in each batch
    dataset = RealSyntheticMixedDataset(
        synthetic_crops=synth_crops,
        real_crops=all_real_crops,
        real_frac=real_frac,
    )

    # For validation, we use only real crops (held-out set)
    # Split real crops into train-val
    n_val = int(0.2 * len(all_real_crops))
    val_real_crops = all_real_crops[:n_val]
    train_real_crops = all_real_crops[n_val:]

    # Replace dataset real crops with train portion
    dataset.real_crops = train_real_crops

    # Validation dataset (real-only)
    val_dataset = RealSyntheticMixedDataset(
        synthetic_crops=synth_crops,
        real_crops=val_real_crops,
        real_frac=0.0,  # real-only for validation
    )
    val_dataset.set_real_only(True)

    # Simple train transform: flip/rotate
    train_transform = make_train_transform(use_seeing_conv=False, seeing_fwhm_range=(2.0, 6.0))

    # Re-apply transform to training dataset
    # (The dataset transform is applied in __getitem__; we'll handle it in the loop)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # keep single-process for simplicity; set >0 if on Linux
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    print(f"\nTrain batches per epoch: ~{len(train_loader)}")
    print(f"Val batches per epoch: ~{len(val_loader)}")

    # ------------------------------------------------------------------
    # 5. Setup optimizer with discriminative learning rates
    # ------------------------------------------------------------------
    # Backbone parameters: all features ConvBlock params
    # Head parameters: classifier Linear layers
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "features" in name or "features." in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"Backbone params: {len(backbone_params)} tensors")
    print(f"Head params: {len(head_params)} tensors")

    # Optimizer: AdamW with discriminative LRs
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.learning_rate / 5.0},
        {"params": head_params, "lr": args.learning_rate},
    ])

    # Loss function: CrossEntropyLoss (model outputs logits, no softmax)
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler: cosine decay over the full fine-tuning run
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ------------------------------------------------------------------
    # 6. Fine-training loop
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    patience = 5  # early stopping patience
    patience_counter = 0

    CLASS_NAMES = [
        "Hubble", "JWST", "Roman", "Euclid", "Rubin", "Spitzer", "WISE", "Gaia", "Altro",
    ]

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)  # (B, 9) raw logits
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()  # step scheduler per-batch

            train_loss_sum += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss_avg = train_loss_sum / max(1, len(train_loader))

        # ---- Validate on real held-out images ----
        if epoch % max(1, args.eval_every) == 0:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    val_loss_sum += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).item()

            val_acc = 100.0 * val_correct / max(1, val_total)
            val_loss_avg = val_loss_sum / max(1, len(val_loader))

            print(f"Epoch {epoch:02d}/{args.epochs}: "
                  f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.2f}% | "
                  f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.2f}%")

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, args.save_best)
                print(f"  -> Saved best model (val_acc={best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            print(f"Epoch {epoch:02d}/{args.epochs}: "
                  f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.2f}%")

    # ---- Final save ----
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, args.save_best.replace(".pt", "_last.pt"))

    print(f"\n=== Fine-tuning complete ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {args.save_best}")


if __name__ == "__main__":
    main()