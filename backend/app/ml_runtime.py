"""Process-wide ML runtime settings (must load before torch/faiss on macOS)."""

from __future__ import annotations

import os
import sys


def configure_ml_runtime() -> None:
    """Allow PyTorch and Faiss to coexist when both link libomp (common on macOS)."""
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # Avoid fork + MPS crashes in DataLoader workers on macOS.
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


def resolve_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_amp(device) -> bool:
    import torch

    return torch.device(device).type in ("cuda", "mps")


def resolve_num_workers(device: str | None, num_workers: int) -> int:
    """MPS and macOS multiprocessing are unstable; default to in-process loading."""
    if num_workers <= 0:
        return 0
    resolved = device or resolve_device()
    if sys.platform == "darwin" or resolved == "mps":
        return 0
    return num_workers


configure_ml_runtime()
