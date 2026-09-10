"""Hardware selection and Apple Silicon optimizations (Task 0).

Priority: ``mps`` (Apple Silicon) -> ``cuda`` (Windows/Linux GPUs) -> ``cpu``.

Includes the startup sanity test from spec section 0.3 and the
``set_float32_matmul_precision('high')`` optimization from section 0.4.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device with priority mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_sanity_test() -> torch.device:
    """Run the spec's startup sanity test and return the selected device.

    Prints the exact messages required by Task 0.3 and materializes a small
    tensor on the device to prove that it is actually usable.
    """
    if torch.backends.mps.is_available():
        print("MPS attivo su Apple Silicon")
        x = torch.randn(1, 1, 64, 64, device="mps")
        print(f"Test tensore: {x.device}")
        return torch.device("mps")

    if torch.cuda.is_available():
        print("MPS non disponibile, uso CPU o CUDA")
        print(f"CUDA attivo: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1, 1, 64, 64, device="cuda")
        print(f"Test tensore: {x.device}")
        return torch.device("cuda")

    print("MPS non disponibile, uso CPU o CUDA")
    print("CUDA non disponibile, uso CPU")
    return torch.device("cpu")


def configure_hardware() -> torch.device:
    """Apply Apple Silicon optimizations and run the sanity test.

    Must be called *before* any tensor operation (spec 0.4). Returns the
    device that training/inference should use.
    """
    # 0.4 — before any tensor operation.
    torch.set_float32_matmul_precision("high")
    return run_sanity_test()


def dataloader_kwargs(num_workers: int = 8) -> dict:
    """Return the DataLoader options required by spec 0.5.

    ``num_workers=8`` (M4 Pro performance cores), ``pin_memory=False``
    (irrelevant on MPS, useful only on CUDA) and ``persistent_workers=True``.
    Callers must spread these over ``**kwargs``.
    """
    return {
        "batch_size": 256,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
