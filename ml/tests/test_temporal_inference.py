"""Unit tests for VERA Layer 1 Temporal / Rolling Real-Time Inference module."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from ml.temporal.engine import RollingInferenceEngine, TemporalInferenceEvent
from ml.temporal.rolling_buffer import (
    AudioWindow,
    DEFAULT_HOP_SIZE_SAMPLES,
    RollingAudioBuffer,
    WINDOW_SIZE_SAMPLES,
)
from ml.temporal.smoother import (
    ExponentialMovingAverageSmoother,
    MovingAverageSmoother,
    create_smoother,
)
from ml.temporal.state_machine import DetectionState, HysteresisStateMachine
from ml.voice_detector.detector import VoiceAuthenticityDetector


# =====================================================================
# 1. Buffer Tests
# =====================================================================

class TestRollingAudioBuffer:
    """Test buffer initialization, window extraction, stride, and flushing."""

    def test_buffer_construction(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000, sample_rate=16000)
        assert buf.window_size_samples == 64600
        assert buf.hop_size_samples == 16000
        assert buf.sample_rate == 16000
        assert buf.buffered_samples_count == 0
        assert buf.total_samples_received == 0

        with pytest.raises(ValueError):
            RollingAudioBuffer(window_size_samples=0)
        with pytest.raises(ValueError):
            RollingAudioBuffer(hop_size_samples=-1)

    def test_window_extraction_exact(self):
        buf = RollingAudioBuffer()
        # Feed exactly 1 window
        chunk = np.zeros(WINDOW_SIZE_SAMPLES, dtype=np.float32)
        windows = buf.append(chunk)

        assert len(windows) == 1
        w = windows[0]
        assert len(w.samples) == WINDOW_SIZE_SAMPLES
        assert w.window_index == 0
        assert w.start_sample_idx == 0
        assert w.end_sample_idx == 64600
        assert w.start_sec == 0.0
        assert w.end_sec == 4.0375
        assert not w.is_flushed
        # Buffer retains (64600 - 16000 = 48600) for next overlap
        assert buf.buffered_samples_count == 64600 - DEFAULT_HOP_SIZE_SAMPLES

    def test_overlap_stride(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        # Feed 64600 + 16000 = 80600 samples -> should yield exactly 2 windows
        chunk = np.ones(64600 + 16000, dtype=np.float32)
        windows = buf.append(chunk)

        assert len(windows) == 2
        assert windows[0].window_index == 0
        assert windows[0].start_sec == 0.0
        assert windows[0].end_sec == 4.0375

        assert windows[1].window_index == 1
        assert windows[1].start_sample_idx == 16000
        assert windows[1].start_sec == 1.0000
        assert windows[1].end_sec == 5.0375

    def test_chunked_incremental_arrival(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        chunk_size = 1600  # 100 ms chunks
        total_windows = []

        # Stream 5 seconds of audio in 100ms increments (50 chunks)
        for i in range(50):
            chunk = np.sin(np.linspace(0, 10, chunk_size, dtype=np.float32))
            ready = buf.append(chunk)
            total_windows.extend(ready)

        # 5.0s = 80,000 samples -> 1st at 64600, 2nd at 80600 (not quite reached, so 1 window)
        assert len(total_windows) == 1
        assert total_windows[0].window_index == 0

        # Feed 1 more chunk to cross 80,600 boundary
        ready2 = buf.append(np.zeros(1600, dtype=np.float32))
        total_windows.extend(ready2)
        assert len(total_windows) == 2
        assert total_windows[1].window_index == 1

    def test_short_audio_flush(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        # Ingest 32,000 samples (2.0s, shorter than window_size)
        short_chunk = np.linspace(-0.5, 0.5, 32000, dtype=np.float32)
        w_early = buf.append(short_chunk)
        assert len(w_early) == 0

        # Flush should tile audio to produce exactly 64,600 samples
        flushed_w = buf.flush(pad_mode="tile")
        assert flushed_w is not None
        assert len(flushed_w.samples) == 64600
        assert flushed_w.is_flushed is True
        assert flushed_w.window_index == 0
        assert buf.buffered_samples_count == 0

        # Second flush on empty buffer returns None
        assert buf.flush() is None

    def test_zero_pad_flush(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        short_chunk = np.ones(1000, dtype=np.float32)
        buf.append(short_chunk)
        flushed_w = buf.flush(pad_mode="zero")
        assert flushed_w is not None
        assert len(flushed_w.samples) == 64600
        assert np.sum(flushed_w.samples[:1000]) == 1000.0
        assert np.sum(flushed_w.samples[1000:]) == 0.0

    def test_stereo_and_bytes_handling(self):
        buf = RollingAudioBuffer(window_size_samples=1000, hop_size_samples=500)
        stereo = np.zeros((1000, 2), dtype=np.float32)
        windows = buf.append(stereo)
        assert len(windows) == 1
        assert windows[0].samples.ndim == 1


# =====================================================================
# 2. Smoother Tests
# =====================================================================

class TestTemporalSmoothers:
    """Test Moving Average and Exponential Moving Average smoothers."""

    def test_sma_smoother(self):
        sma = MovingAverageSmoother(window_len=3)
        assert sma.current_value is None

        v1 = sma.update(10.0)
        assert v1 == 10.0

        v2 = sma.update(20.0)
        assert v2 == 15.0

        v3 = sma.update(30.0)
        assert v3 == 20.0

        # Sliding window drops 10.0: (20 + 30 + 40) / 3 = 30.0
        v4 = sma.update(40.0)
        assert v4 == 30.0
        assert sma.history == [20.0, 30.0, 40.0]

    def test_ema_smoother(self):
        ema = ExponentialMovingAverageSmoother(alpha=0.5)
        # S0 = 100.0 (unbiased first sample)
        assert ema.update(100.0) == 100.0
        # S1 = 0.5 * 50 + 0.5 * 100 = 75.0
        assert ema.update(50.0) == 75.0
        # S2 = 0.5 * 0 + 0.5 * 75 = 37.5
        assert ema.update(0.0) == 37.5

    def test_smoother_reset(self):
        ema = ExponentialMovingAverageSmoother(alpha=0.3)
        ema.update(50.0)
        ema.reset()
        assert ema.current_value is None
        assert len(ema.history) == 0

    def test_create_smoother_factory(self):
        s_ema = create_smoother("ema", alpha=0.4)
        assert isinstance(s_ema, ExponentialMovingAverageSmoother)
        assert s_ema.alpha == 0.4

        s_sma = create_smoother("sma", window_len=4)
        assert isinstance(s_sma, MovingAverageSmoother)
        assert s_sma.window_len == 4

        with pytest.raises(ValueError):
            create_smoother("unsupported_type")


# =====================================================================
# 3. State Machine & Hysteresis Tests
# =====================================================================

class TestHysteresisStateMachine:
    """Test multi-state hysteresis transitions, dwell debounce, and jitter rejection."""

    def test_zone_mapping(self):
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0)
        assert sm.determine_instant_zone(10.0) == DetectionState.BONAFIDE
        assert sm.determine_instant_zone(34.9) == DetectionState.BONAFIDE
        assert sm.determine_instant_zone(35.0) == DetectionState.SUSPICIOUS
        assert sm.determine_instant_zone(50.0) == DetectionState.SUSPICIOUS
        assert sm.determine_instant_zone(64.9) == DetectionState.SUSPICIOUS
        assert sm.determine_instant_zone(65.0) == DetectionState.SPOOF
        assert sm.determine_instant_zone(95.0) == DetectionState.SPOOF

    def test_debounce_dwell_logic(self):
        # dwell_count = 2 means target state requires 2 consecutive windows to commit
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)
        assert sm.current_state == DetectionState.BONAFIDE

        # First spoof score (80.0) -> pending, not yet committed
        state, changed = sm.update(80.0)
        assert state == DetectionState.BONAFIDE
        assert changed is False

        # Second consecutive spoof score (85.0) -> committed!
        state, changed = sm.update(85.0)
        assert state == DetectionState.SPOOF
        assert changed is True
        assert sm.total_transitions == 1

    def test_transient_glitch_rejection(self):
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)
        # Normal bona fide
        sm.update(10.0)
        assert sm.current_state == DetectionState.BONAFIDE

        # Single transient spike into spoof zone (glitch / pop)
        state, changed = sm.update(90.0)
        assert state == DetectionState.BONAFIDE
        assert changed is False

        # Next window returns to bona fide -> transition cancelled
        state, changed = sm.update(15.0)
        assert state == DetectionState.BONAFIDE
        assert changed is False
        assert sm.total_transitions == 0

    def test_suspicious_entry(self):
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0)
        state, changed = sm.update(50.0)
        assert state == DetectionState.SUSPICIOUS
        assert changed is True


# =====================================================================
# 4. Engine & Streaming Integration Tests
# =====================================================================

class MockDetector:
    """Mock detector returning predefined logits without neural network execution."""
    def __init__(self, fixed_score: float = 12.0):
        self.fixed_score = fixed_score

    def _run_forward(self, preprocessed_waveform: np.ndarray) -> float:
        assert len(preprocessed_waveform) == WINDOW_SIZE_SAMPLES
        return self.fixed_score


class TestRollingInferenceEngine:
    """Test unified engine execution, event streams, and edge cases."""

    def test_engine_streaming_with_mock(self):
        mock_det = MockDetector(fixed_score=12.5)  # Genuine voice logit
        engine = RollingInferenceEngine(detector=mock_det, smoothing_method="ema", smoothing_param=0.5)

        # Ingest 5 seconds of audio in 0.5s chunks
        chunk_len = 8000
        all_events = []
        for _ in range(10):
            events = engine.push_chunk(np.zeros(chunk_len, dtype=np.float32))
            all_events.extend(events)

        # Window 0 emitted at 64,600 samples (~4.04s)
        assert len(all_events) == 1
        ev = all_events[0]
        assert isinstance(ev, TemporalInferenceEvent)
        assert ev.window_index == 0
        assert ev.raw_score == 12.5
        assert ev.calibrated_prob_bonafide > 0.99
        assert ev.calibrated_spoof_signal < 1.0
        assert ev.voice_integrity_score > 99.0
        assert ev.state == DetectionState.BONAFIDE
        assert ev.is_alert is False
        assert ev.latency_ms >= 0.0

        # Flush remaining 80,000 - 64,600 = 15,400 samples
        flush_events = engine.flush()
        assert len(flush_events) == 1
        assert flush_events[0].is_flushed is True
        assert flush_events[0].window_index == 1

    def test_silence_edge_case(self):
        mock_det = MockDetector(fixed_score=0.0)  # Ambiguous logit
        engine = RollingInferenceEngine(detector=mock_det)
        silence = np.zeros(WINDOW_SIZE_SAMPLES, dtype=np.float32)
        events = engine.push_chunk(silence)
        assert len(events) == 1
        assert events[0].raw_score == 0.0

    def test_noise_edge_case(self):
        mock_det = MockDetector(fixed_score=-2.5)  # Spoof logit
        engine = RollingInferenceEngine(detector=mock_det, dwell_count=1)
        noise = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32) * 0.1
        events = engine.push_chunk(noise)
        assert len(events) == 1
        assert events[0].calibrated_spoof_signal > 90.0
        assert events[0].state == DetectionState.SPOOF
        assert events[0].is_alert is True

    @pytest.mark.skipif(
        not Path("data/asvspoof2019/eval/flac/LA_E_1025210.flac").is_file(),
        reason="Requires downloaded evaluation FLAC file",
    )
    def test_real_audio_streaming_simulation(self):
        """Simulate real-time streaming using an actual ASVspoof evaluation FLAC audio file."""
        engine = RollingInferenceEngine(smoothing_method="ema", smoothing_param=0.4)
        audio_file = Path("data/asvspoof2019/eval/flac/LA_E_1025210.flac")

        events = list(engine.process_file_stream(audio_file, chunk_size_sec=0.5))
        assert len(events) >= 1
        for ev in events:
            assert isinstance(ev, TemporalInferenceEvent)
            assert ev.raw_score is not None
            assert 0.0 <= ev.calibrated_spoof_signal <= 100.0
            assert 0.0 <= ev.voice_integrity_score <= 100.0
            assert ev.state in (DetectionState.BONAFIDE, DetectionState.SUSPICIOUS, DetectionState.SPOOF)
            assert ev.latency_ms > 0.0
