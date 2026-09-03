"""Audio loading and validation module for Layer 1 Voice Authenticity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math


TARGET_SAMPLE_RATE: int = 16_000


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata for loaded audio."""
    file_path: str
    original_sample_rate: int
    target_sample_rate: int
    duration_seconds: float
    original_channels: int
    num_samples_raw: int
    num_samples_processed: int
    was_resampled: bool


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample a 1D audio array to the target sample rate using polyphase filtering.

    Args:
        audio: 1D numpy array of float32 samples.
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16,000 Hz).

    Returns:
        1D numpy array resampled to target_sr with float32 dtype.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = resample_poly(audio, up, down).astype(np.float32)
    return resampled


def load_audio(
    file_path: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    allow_resample: bool = True
) -> tuple[np.ndarray, AudioMetadata]:
    """Load an audio file, convert to mono float32, and ensure target sample rate.

    Args:
        file_path: Path to the audio file (WAV format recommended).
        target_sr: Expected sample rate (default: 16,000 Hz).
        allow_resample: If True, resamples audio not matching target_sr;
                        if False, raises ValueError on mismatch.

    Returns:
        tuple of (waveform_1d_float32, AudioMetadata)

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If audio is empty, corrupted, or unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception as exc:
        raise ValueError(f"Failed to read audio file '{path}': {exc}") from exc

    if data.size == 0:
        raise ValueError(f"Audio file is empty: '{path}'")

    raw_samples = len(data)
    orig_channels = 1 if data.ndim == 1 else data.shape[1]
    duration_sec = raw_samples / float(sr)

    # Convert multi-channel (e.g. stereo) to mono by averaging across channels
    if data.ndim > 1:
        waveform = np.mean(data, axis=1, dtype=np.float32)
    else:
        waveform = data.astype(np.float32)

    # Validate finite values
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Audio file contains NaN or infinite values: '{path}'")

    # Handle sample rate
    was_resampled = False
    if sr != target_sr:
        if not allow_resample:
            raise ValueError(
                f"Sample rate mismatch: expected {target_sr} Hz, got {sr} Hz "
                f"for '{path}'. Resampling is disabled."
            )
        waveform = resample_audio(waveform, orig_sr=sr, target_sr=target_sr)
        was_resampled = True

    metadata = AudioMetadata(
        file_path=str(path.resolve()),
        original_sample_rate=sr,
        target_sample_rate=target_sr,
        duration_seconds=duration_sec,
        original_channels=orig_channels,
        num_samples_raw=raw_samples,
        num_samples_processed=len(waveform),
        was_resampled=was_resampled,
    )

    return waveform.astype(np.float32), metadata
