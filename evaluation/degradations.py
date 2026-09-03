"""Controlled acoustic degradation and corruption transforms for robustness testing.

Transforms:
  1. Clean Audio (Identity)
  2. Background Noise (Additive noise at controlled SNR, e.g. 15 dB)
  3. Telephone / VoIP Channel (300-3400 Hz bandpass filter + mu-law quantization)
  4. Replay / Re-recording (Room impulse reverberation + acoustic channel convolution)
  5. Audio Level Variation (Gain attenuation / dynamic range scaling)
  6. Unseen Spoof Source (Novel synthetic vocoder & diffusion artifacts)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from scipy import signal
import soundfile as sf


def add_background_noise(audio: np.ndarray, snr_db: float = 15.0, seed: int = 42) -> np.ndarray:
    """Add stationary environmental background noise at a specific SNR (in dB)."""
    rng = np.random.RandomState(seed)
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-9:
        return audio

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=len(audio)).astype(np.float32)

    corrupted = audio + noise
    # Prevent hard clipping
    max_val = np.max(np.abs(corrupted))
    if max_val > 1.0:
        corrupted = corrupted / max_val
    return corrupted.astype(np.float32)


def apply_telephony_compression(audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """Simulate telephone PSTN / VoIP channel (G.711 / 300Hz-3400Hz bandpass + mu-law quantization)."""
    # 1. Bandpass filter 300 Hz - 3400 Hz
    nyquist = sr / 2.0
    low = 300.0 / nyquist
    high = 3400.0 / nyquist
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos, audio).astype(np.float32)

    # 2. Mu-law 8-bit companding simulation
    mu = 255.0
    sign = np.sign(filtered)
    abs_x = np.abs(filtered)
    # Clip to [-1, 1]
    abs_x = np.clip(abs_x, 0.0, 1.0)
    companded = sign * (np.log(1.0 + mu * abs_x) / np.log(1.0 + mu))

    # 8-bit quantization
    quantized = np.round((companded + 1.0) * 127.5)
    dequantized = (quantized / 127.5) - 1.0

    # Expand
    expanded = np.sign(dequantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(dequantized) - 1.0)
    return expanded.astype(np.float32)


def apply_replay_acoustics(audio: np.ndarray, sr: int = 16_000, seed: int = 42) -> np.ndarray:
    """Simulate physical loudspeaker replay and microphone re-recording with room reverberation."""
    rng = np.random.RandomState(seed)
    # Synthetic room impulse response (RIR) with exponential decay
    rir_len = int(sr * 0.15)  # 150 ms reverberation tail
    t_rir = np.linspace(0, 0.15, rir_len)
    decay = np.exp(-t_rir / 0.04)  # 40ms decay constant
    rir = rng.normal(0, 1, rir_len) * decay
    # Direct path peak
    rir[0] = 1.0
    rir = rir / np.sum(np.abs(rir))

    # Convolve with audio
    replayed = signal.convolve(audio, rir, mode="same")

    # Slight high-frequency roll-off typical of consumer mic/speaker
    b, a = signal.butter(2, 6000.0 / (sr / 2.0), btype="low")
    replayed = signal.lfilter(b, a, replayed)

    max_val = np.max(np.abs(replayed))
    if max_val > 1e-6:
        replayed = replayed / max_val * np.max(np.abs(audio))
    return replayed.astype(np.float32)


def apply_level_variation(audio: np.ndarray, gain_db: float = -14.0) -> np.ndarray:
    """Simulate low-volume / distant microphone audio level variation."""
    gain_linear = 10.0 ** (gain_db / 20.0)
    scaled = audio * gain_linear
    return scaled.astype(np.float32)


@dataclass(frozen=True)
class DegradationCondition:
    """Specification of a test condition."""
    name: str
    code: str
    description: str
    transform_fn: Callable[[np.ndarray], np.ndarray]


DEGRADATION_CONDITIONS: Dict[str, DegradationCondition] = {
    "clean": DegradationCondition(
        name="Clean Baseline",
        code="C0_CLEAN",
        description="Original uncorrupted studio/clean benchmark audio",
        transform_fn=lambda x: x,
    ),
    "background_noise": DegradationCondition(
        name="Background Noise (15dB SNR)",
        code="C1_NOISE_15DB",
        description="Additive environmental background noise at 15 dB SNR",
        transform_fn=lambda x: add_background_noise(x, snr_db=15.0),
    ),
    "telephony": DegradationCondition(
        name="Telephone / VoIP Compression",
        code="C2_TELEPHONY",
        description="300-3400Hz PSTN bandpass filter + 8-bit mu-law codec companding",
        transform_fn=lambda x: apply_telephony_compression(x),
    ),
    "replay": DegradationCondition(
        name="Replay / Re-recording",
        code="C3_REPLAY",
        description="Room impulse reverberation (150ms tail) + speaker/mic acoustic transfer",
        transform_fn=lambda x: apply_replay_acoustics(x),
    ),
    "level_variation": DegradationCondition(
        name="Audio Level Variation (-14dB)",
        code="C4_LEVEL_VAR",
        description="Gain attenuation to simulate distant speaker or weak microphone input",
        transform_fn=lambda x: apply_level_variation(x, gain_db=-14.0),
    ),
}
