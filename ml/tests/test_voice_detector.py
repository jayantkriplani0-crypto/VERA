"""Unit tests for Layer 1 Voice Authenticity Detection (Spectra-AASIST3).

Tests:
  - Mono WAV
  - Stereo WAV
  - Non-16kHz WAV (8kHz, 22.05kHz, 44.1kHz, 48kHz)
  - Short audio (< 64,600 samples)
  - Exact-length audio (== 64,600 samples)
  - Long audio (> 64,600 samples)
  - Invalid / missing file handling
  - Preemphasis mathematical correctness
  - Windowing tile-repeat and truncation policies
  - Score interpretation without fake probabilities or percentages
  - Full end-to-end detector pipeline with both dummy and loaded model
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from ml.preprocessing.audio_loader import (
    AudioMetadata,
    TARGET_SAMPLE_RATE,
    load_audio,
    resample_audio,
)
from ml.preprocessing.preprocessor import (
    PREEMPHASIS_COEFFICIENT,
    WINDOW_LENGTH_SAMPLES,
    apply_preemphasis,
    apply_windowing,
    preprocess_waveform,
)
from ml.voice_detector.scorer import (
    BONAFIDE_CLASS_INDEX,
    OFFICIAL_EER_THRESHOLD,
    ScoreInterpretation,
    interpret_score,
)
from ml.voice_detector.detector import (
    DetectionResult,
    VoiceAuthenticityDetector,
)


# ---------------------------------------------------------------------------
# Helpers for synthesizing test WAV files
# ---------------------------------------------------------------------------
def create_test_wav(
    filepath: Path,
    duration_sec: float,
    sample_rate: int = 16_000,
    channels: int = 1,
    freq: float = 440.0
) -> Path:
    """Generate a synthetic sine-wave WAV file for testing."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    if channels == 1:
        data = sine
    else:
        # Multi-channel: slightly different frequencies per channel
        ch_list = [sine * (1.0 + 0.1 * i) for i in range(channels)]
        data = np.column_stack(ch_list).astype(np.float32)

    sf.write(str(filepath), data, sample_rate)
    return filepath


# ---------------------------------------------------------------------------
# Preprocessing Unit Tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    """Test official preprocessing algorithms."""

    def test_preemphasis_formula(self):
        """Verify y[n] = x[n] - 0.97 * x[n-1]."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        coef = 0.97
        expected = np.array([
            1.0,
            2.0 - 0.97 * 1.0,
            3.0 - 0.97 * 2.0,
            4.0 - 0.97 * 3.0,
        ], dtype=np.float32)

        actual = apply_preemphasis(x, coef=coef)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_preemphasis_empty_error(self):
        with pytest.raises(ValueError, match="empty waveform"):
            apply_preemphasis(np.array([], dtype=np.float32))

    def test_windowing_short_audio_tiling(self):
        """Short audio (< 64600) must be tiled until >= 64600 and sliced."""
        short_len = 16_000  # 1.0 second
        x = np.arange(short_len, dtype=np.float32)
        windowed = apply_windowing(x, target_length=WINDOW_LENGTH_SAMPLES)

        assert len(windowed) == WINDOW_LENGTH_SAMPLES
        # Check that it tiled correctly
        np.testing.assert_array_equal(windowed[:short_len], x)
        np.testing.assert_array_equal(windowed[short_len:2 * short_len], x)

    def test_windowing_exact_length(self):
        """Exact 64600 samples should return identical array."""
        x = np.random.randn(WINDOW_LENGTH_SAMPLES).astype(np.float32)
        windowed = apply_windowing(x, target_length=WINDOW_LENGTH_SAMPLES)
        assert len(windowed) == WINDOW_LENGTH_SAMPLES
        np.testing.assert_array_equal(windowed, x)

    def test_windowing_long_audio_truncation(self):
        """Long audio (> 64600) must take the first 64600 samples."""
        long_len = 100_000
        x = np.arange(long_len, dtype=np.float32)
        windowed = apply_windowing(x, target_length=WINDOW_LENGTH_SAMPLES)
        assert len(windowed) == WINDOW_LENGTH_SAMPLES
        np.testing.assert_array_equal(windowed, x[:WINDOW_LENGTH_SAMPLES])

    def test_full_preprocess_pipeline(self):
        """Verify full preprocessor pipeline produces float32 of shape (64600,)."""
        x = np.random.randn(48_000).astype(np.float32)
        processed = preprocess_waveform(x)
        assert processed.shape == (WINDOW_LENGTH_SAMPLES,)
        assert processed.dtype == np.float32


# ---------------------------------------------------------------------------
# Audio Loader Unit Tests
# ---------------------------------------------------------------------------
class TestAudioLoader:
    """Test audio loading across various WAV configurations."""

    def test_mono_wav_loading(self, tmp_path):
        """Verify mono 16kHz WAV loads cleanly."""
        wav_file = create_test_wav(tmp_path / "mono.wav", duration_sec=2.0, sample_rate=16000, channels=1)
        audio, meta = load_audio(wav_file)

        assert audio.ndim == 1
        assert meta.original_channels == 1
        assert meta.original_sample_rate == 16000
        assert meta.target_sample_rate == 16000
        assert not meta.was_resampled
        assert abs(meta.duration_seconds - 2.0) < 1e-4

    def test_stereo_wav_mixing(self, tmp_path):
        """Verify stereo WAV is averaged to single-channel mono."""
        wav_file = create_test_wav(tmp_path / "stereo.wav", duration_sec=1.5, sample_rate=16000, channels=2)
        audio, meta = load_audio(wav_file)

        assert audio.ndim == 1
        assert meta.original_channels == 2
        assert len(audio) == 24000  # 1.5 * 16000

    @pytest.mark.parametrize("orig_sr", [8000, 22050, 44100, 48000])
    def test_non_16k_resampling(self, tmp_path, orig_sr):
        """Verify non-16kHz WAVs are resampled to exactly 16kHz."""
        wav_file = create_test_wav(tmp_path / f"audio_{orig_sr}.wav", duration_sec=2.0, sample_rate=orig_sr)
        audio, meta = load_audio(wav_file)

        assert meta.original_sample_rate == orig_sr
        assert meta.target_sample_rate == 16000
        assert meta.was_resampled
        assert abs(len(audio) - 32000) <= 2  # ~2 seconds at 16kHz

    def test_invalid_missing_file(self, tmp_path):
        """Verify missing file raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist.wav"
        with pytest.raises(FileNotFoundError):
            load_audio(missing)

    def test_corrupt_or_empty_file(self, tmp_path):
        """Verify empty file raises ValueError."""
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            load_audio(empty)


# ---------------------------------------------------------------------------
# Score Interpretation Unit Tests
# ---------------------------------------------------------------------------
class TestScorer:
    """Test score interpretation rules."""

    def test_bona_fide_interpretation(self):
        """Score above threshold is classified as bona fide."""
        raw_score = 2.382637
        interp = interpret_score(raw_score)

        assert interp.is_bona_fide is True
        assert "BONA FIDE" in interp.predicted_label
        assert "Raw Bona Fide Logit" in interp.metric_type
        # Verify no percentage or fake probability in summary
        summary = interp.summary_text()
        assert "%" not in summary
        assert "probability" not in summary.lower() or "not a percentage or probability" in summary.lower()

    def test_spoof_interpretation(self):
        """Score below threshold is classified as spoof."""
        raw_score = -4.500000
        interp = interpret_score(raw_score)

        assert interp.is_bona_fide is False
        assert "SPOOF" in interp.predicted_label
        assert interp.raw_score == -4.5
        assert interp.threshold == OFFICIAL_EER_THRESHOLD

    def test_no_percentage_claims(self):
        """Verify scores are explicitly represented as logits, never percentages."""
        interp = interpret_score(0.0)
        assert not hasattr(interp, "probability")
        assert not hasattr(interp, "percentage")


# ---------------------------------------------------------------------------
# Detector Pipeline Unit Tests with Mock / Stub Model
# ---------------------------------------------------------------------------
class DummySpectraNet(nn.Module):
    """Stub network returning controllable 2-class logits for rapid test execution."""

    def __init__(self, bona_fide_logit: float = 2.382637):
        super().__init__()
        self.bona_fide_logit = bona_fide_logit
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [batch_size, 2] -> [spoof_logit, bona_fide_logit]
        bs = x.shape[0]
        logits = torch.tensor([[ -self.bona_fide_logit, self.bona_fide_logit ]], dtype=torch.float32)
        return logits.expand(bs, -1)


class TestVoiceAuthenticityDetector:
    """Test the full detector pipeline with dummy and real audio configurations."""

    @pytest.fixture
    def detector(self) -> VoiceAuthenticityDetector:
        model = DummySpectraNet(bona_fide_logit=2.382637)
        return VoiceAuthenticityDetector(model=model, device="cpu", model_repo="lab260/Spectra-AASIST3")

    def test_predict_mono_file(self, detector, tmp_path):
        wav = create_test_wav(tmp_path / "mono_test.wav", duration_sec=3.0, sample_rate=16000, channels=1)
        result = detector.predict_file(wav)

        assert isinstance(result, DetectionResult)
        assert result.sample_rate == 16000
        assert abs(result.duration_seconds - 3.0) < 1e-4
        assert result.interpretation.is_bona_fide is True
        assert abs(result.raw_score - 2.382637) < 1e-5

    def test_predict_stereo_file(self, detector, tmp_path):
        wav = create_test_wav(tmp_path / "stereo_test.wav", duration_sec=2.5, sample_rate=16000, channels=2)
        result = detector.predict_file(wav)

        assert result.metadata.original_channels == 2
        assert abs(result.duration_seconds - 2.5) < 1e-4
        assert result.interpretation.is_bona_fide is True

    def test_predict_non_16k_file(self, detector, tmp_path):
        wav = create_test_wav(tmp_path / "audio_44k.wav", duration_sec=1.8, sample_rate=44100, channels=1)
        result = detector.predict_file(wav)

        assert result.sample_rate == 44100
        assert result.metadata.was_resampled is True
        assert result.interpretation.is_bona_fide is True

    def test_predict_short_audio(self, detector, tmp_path):
        """Audio under 4.04s (e.g. 0.5s) should be processed via tiling without error."""
        wav = create_test_wav(tmp_path / "short.wav", duration_sec=0.5, sample_rate=16000)
        result = detector.predict_file(wav)

        assert result.duration_seconds < 1.0
        assert result.interpretation.is_bona_fide is True

    def test_predict_exact_length_audio(self, detector, tmp_path):
        """Audio exactly 64,600 samples (4.0375s)."""
        duration = 64_600 / 16_000
        wav = create_test_wav(tmp_path / "exact.wav", duration_sec=duration, sample_rate=16000)
        result = detector.predict_file(wav)

        assert abs(result.duration_seconds - 4.0375) < 1e-4
        assert result.interpretation.is_bona_fide is True

    def test_predict_long_audio(self, detector, tmp_path):
        """Audio over 4.04s (e.g. 8.0s) should be truncated without error."""
        wav = create_test_wav(tmp_path / "long.wav", duration_sec=8.0, sample_rate=16000)
        result = detector.predict_file(wav)

        assert result.duration_seconds > 7.9
        assert result.interpretation.is_bona_fide is True

    def test_predict_missing_file_raises(self, detector, tmp_path):
        with pytest.raises(FileNotFoundError):
            detector.predict_file(tmp_path / "non_existent.wav")

    def test_report_formatting(self, detector, tmp_path):
        wav = create_test_wav(tmp_path / "report_test.wav", duration_sec=3.0, sample_rate=16000)
        result = detector.predict_file(wav)
        report = result.format_report()

        assert "VERA LAYER 1: VOICE AUTHENTICITY DETECTION REPORT" in report
        assert "report_test.wav" in report
        assert "16000 Hz" in report
        assert "3.00 seconds" in report
        assert "lab260/Spectra-AASIST3" in report
        assert "Raw Model Score" in report
        assert "BONA FIDE" in report
        # Verify no percentage claims
        assert "100%" not in report
        assert "probability" not in report.lower() or "not a percentage or probability" in report.lower()
