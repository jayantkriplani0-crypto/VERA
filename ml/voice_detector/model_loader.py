"""Model loader for Spectra-AASIST3."""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


MODEL_HUB_REPO: str = "lab260/Spectra-AASIST3"


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_spectra_aasist3(
    repo_or_path: str = MODEL_HUB_REPO,
    device: str = "auto"
) -> tuple[nn.Module, str]:
    """Load the pretrained SpectraAASIST3 model.

    Args:
        repo_or_path: Hugging Face model repository ID or local checkpoint path.
        device: Device to place the model on ('auto', 'cpu', 'cuda', etc.).

    Returns:
        tuple of (model, resolved_device_name)

    Raises:
        RuntimeError: If model instantiation or weight loading fails.
    """
    from models.spectra_aasist3_net import SpectraAASIST3

    target_device = resolve_device(device)
    device_name = str(target_device)

    try:
        model = SpectraAASIST3.from_pretrained(repo_or_path)
        model = model.to(target_device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Spectra-AASIST3 from '{repo_or_path}' onto {device_name}: {exc}"
        ) from exc

    return model, device_name
