"""Model-specific preprocessing for Spectra-AASIST3.

Follows the official Hugging Face model card:
  1. Preemphasis (0.97) applied to the full waveform: y[n] = x[n] - 0.97 * x[n-1]
  2. Deterministic first-64,600-sample window (~4.04s): tile-repeat if shorter,
     truncate to first 64,600 samples if longer. No random crops or padding.
"""
from __future__ import annotations

import numpy as np


# Official Spectra-AASIST3 constants
PREEMPHASIS_COEFFICIENT: float = 0.97
WINDOW_LENGTH_SAMPLES: int = 64_600  # ~4.0375 seconds at 16 kHz


def apply_preemphasis(waveform: np.ndarray, coef: float = PREEMPHASIS_COEFFICIENT) -> np.ndarray:
    """Apply first-order high-pass preemphasis filter to the 1D waveform.

    Formula:
        y[0] = x[0]
        y[n] = x[n] - coef * x[n-1] for n >= 1

    Args:
        waveform: 1D numpy array of float32 samples.
        coef: Preemphasis filter coefficient (default: 0.97).

    Returns:
        1D numpy array with preemphasis applied, dtype float32.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")
    if len(waveform) == 0:
        raise ValueError("Cannot apply preemphasis on an empty waveform.")

    filtered = np.empty_like(waveform, dtype=np.float32)
    filtered[0] = waveform[0]
    filtered[1:] = waveform[1:] - coef * waveform[:-1]
    return filtered


def apply_windowing(waveform: np.ndarray, target_length: int = WINDOW_LENGTH_SAMPLES) -> np.ndarray:
    """Window waveform to exactly target_length samples using official deterministic policy.

    Policy:
      - If shorter than target_length: Tile-repeat until length >= target_length,
        then take the first target_length samples.
      - If longer than target_length: Take the first target_length samples.
      - If exactly target_length: Return unchanged.

    Args:
        waveform: 1D numpy array of float32 samples.
        target_length: Required sample length (default: 64,600).

    Returns:
        1D numpy array of length target_length, dtype float32.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")
    if len(waveform) == 0:
        raise ValueError("Cannot apply windowing on an empty waveform.")

    if len(waveform) < target_length:
        repeats = (target_length // len(waveform)) + 1
        tiled = np.tile(waveform, repeats)
        return tiled[:target_length].astype(np.float32)

    return waveform[:target_length].astype(np.float32)


def preprocess_waveform(
    waveform: np.ndarray,
    preemphasis_coef: float = PREEMPHASIS_COEFFICIENT,
    target_window_length: int = WINDOW_LENGTH_SAMPLES
) -> np.ndarray:
    """Full preprocessing pipeline for Spectra-AASIST3.

    Args:
        waveform: 1D mono float32 waveform (at 16 kHz).
        preemphasis_coef: Preemphasis coefficient (0.97).
        target_window_length: Target sample count (64,600).

    Returns:
        1D numpy array of shape (64600,) ready for tensor conversion and model forward pass.
    """
    # 1. Apply preemphasis to the full waveform
    filtered = apply_preemphasis(waveform, coef=preemphasis_coef)

    # 2. Window deterministically to 64,600 samples
    windowed = apply_windowing(filtered, target_length=target_window_length)

    return windowed
