"""Utility functions for device detection and other helpers."""
import torch


def get_best_device() -> torch.device:
    """
    Detect the best available device in order: MPS (Apple Silicon) → CUDA → CPU.
    
    Returns:
        torch.device: The best available device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Get a torch device, with automatic fallback if None.
    
    Args:
        device: Device specification. Can be:
            - None: auto-detect best device (MPS → CUDA → CPU)
            - str: device name like "mps", "cuda", "cuda:0", "cpu"
            - torch.device: passed through
    
    Returns:
        torch.device: The requested or auto-detected device.
    """
    if device is None:
        return get_best_device()
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device
