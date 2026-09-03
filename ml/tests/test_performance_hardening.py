"""Regression and edge-case tests for VERA Layer 1 performance optimization and hardening."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.preprocessing.preprocessor import apply_preemphasis
from ml.temporal.engine import RollingInferenceEngine, TemporalInferenceEvent
from ml.temporal.rolling_buffer import RollingAudioBuffer, WINDOW_SIZE_SAMPLES, DEFAULT_HOP_SIZE_SAMPLES
from ml.temporal.state_machine import DetectionState
from ml.voice_detector.detector import VoiceAuthenticityDetector


# =====================================================================
# 1. Numerical Parity Tests for Optimizations
# =====================================================================

class TestPreemphasisOptimizationParity:
    """Verify optimized apply_preemphasis matches exact mathematical definition."""

    def test_exact_formula_parity(self):
        # Generate arbitrary test waveform
        np.random.seed(42)
        waveform = np.random.uniform(-1.0, 1.0, 64600).astype(np.float32)
        coef = 0.97

        # Reference naive implementation
        expected = np.empty_like(waveform)
        expected[0] = waveform[0]
        for i in range(1, len(waveform)):
            expected[i] = waveform[i] - coef * waveform[i - 1]

        # Optimized vectorized implementation
        actual = apply_preemphasis(waveform, coef=coef)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_dc_and_silence_preemphasis(self):
        # Pure DC should result in (1 - 0.97) = 0.03 * DC after sample 0
        dc = np.ones(1000, dtype=np.float32) * 2.0
        filtered = apply_preemphasis(dc, coef=0.97)
        assert filtered[0] == 2.0
        np.testing.assert_allclose(filtered[1:], 2.0 * (1.0 - 0.97), atol=1e-6)

        # Pure silence should remain exact zeros
        silence = np.zeros(1000, dtype=np.float32)
        assert np.all(apply_preemphasis(silence) == 0.0)


class TestBatchInferenceParity:
    """Verify batched forward pass matches sequential forward pass."""

    @pytest.fixture(scope="class")
    def detector(self):
        return VoiceAuthenticityDetector(device="cpu")

    def test_batch_vs_sequential_parity(self, detector):
        np.random.seed(123)
        w1 = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32)
        w2 = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32)
        w3 = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32)

        # Sequential
        s1 = detector._run_forward(w1)
        s2 = detector._run_forward(w2)
        s3 = detector._run_forward(w3)
        sequential_scores = [s1, s2, s3]

        # Batched
        batched_scores = detector._run_forward_batch([w1, w2, w3])

        assert len(batched_scores) == 3
        for seq, batch in zip(sequential_scores, batched_scores):
            assert abs(seq - batch) < 1e-4, f"Discrepancy: seq={seq}, batch={batch}"


# =====================================================================
# 2. Buffer Stability & Long Stream Tests
# =====================================================================

class TestRingBufferParityAndStability:
    """Verify buffer correctly extracts exact windows over hundreds of chunks."""

    def test_buffer_numerical_parity(self):
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        np.random.seed(99)
        full_audio = np.random.uniform(-0.9, 0.9, 160000).astype(np.float32)

        # Ingest in small 2000-sample chunks
        extracted_windows = []
        chunk_size = 2000
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i : i + chunk_size]
            extracted_windows.extend(buf.append(chunk))

        # Expected number of windows: (160000 - 64600) // 16000 + 1 = 6 windows
        assert len(extracted_windows) == 6

        # Verify exact samples against direct slice from full_audio
        for w in extracted_windows:
            expected_slice = full_audio[w.start_sample_idx : w.end_sample_idx]
            np.testing.assert_array_equal(w.samples, expected_slice)
            assert w.end_sample_idx - w.start_sample_idx == 64600

    def test_long_continuous_stream_stability(self):
        """Simulate 200 chunks (100 seconds of audio) to verify constant memory and no drift."""
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        chunk = np.random.randn(8000).astype(np.float32)  # 500ms chunks

        windows_count = 0
        last_end_sample = -1

        for i in range(200):
            windows = buf.append(chunk)
            for w in windows:
                assert w.window_index == windows_count
                assert w.end_sample_idx > last_end_sample
                last_end_sample = w.end_sample_idx
                windows_count += 1

        # Total samples = 200 * 8000 = 1,600,000 samples = 100 seconds
        assert buf.total_samples_received == 1_600_000
        # Expected windows: (1,600,000 - 64,600) // 16,000 + 1 = 96 windows
        assert windows_count == 96
        # Retained buffer should never grow unbounded
        assert buf.buffered_samples_count < 64600

    def test_irregular_bursty_chunk_arrivals(self):
        """Verify irregular chunk sizes (100ms, 1200ms, 50ms, 3000ms) process correctly."""
        buf = RollingAudioBuffer(window_size_samples=64600, hop_size_samples=16000)
        chunk_sizes = [1600, 19200, 800, 48000, 12800, 3200, 64600]

        total_windows = []
        for sz in chunk_sizes:
            chunk = np.sin(np.linspace(0, 10, sz, dtype=np.float32))
            total_windows.extend(buf.append(chunk))

        total_samples = sum(chunk_sizes)
        expected_windows = (total_samples - 64600) // 16000 + 1
        assert len(total_windows) == expected_windows

        # Verify sequential indices
        for i, w in enumerate(total_windows):
            assert w.window_index == i


# =====================================================================
# 3. Edge Cases & Error Handling Tests
# =====================================================================

class MockFastDetector:
    """Mock detector for rapid edge-case testing."""
    def __init__(self, score: float = 5.0):
        self.score = score

    def _run_forward(self, preprocessed_waveform: np.ndarray) -> float:
        return self.score

    def _run_forward_batch(self, preprocessed_waveforms: list[np.ndarray]) -> list[float]:
        return [self.score] * len(preprocessed_waveforms)


class TestEngineEdgeCasesAndLifecycle:
    """Verify silence, clipping, reset, and malformed inputs in RollingInferenceEngine."""

    def test_pure_silence_stream(self):
        engine = RollingInferenceEngine(detector=MockFastDetector(score=3.0))
        silence_chunk = np.zeros(80000, dtype=np.float32)
        events = engine.push_chunk(silence_chunk)
        assert len(events) >= 1
        for ev in events:
            assert ev.raw_score == 3.0
            assert 0.0 <= ev.calibrated_spoof_signal <= 100.0

    def test_extreme_amplitude_clipping(self):
        engine = RollingInferenceEngine(detector=MockFastDetector(score=-2.0))
        # Audio exceeding [-1.0, 1.0] should be normalized safely
        clipped_audio = np.ones(64600, dtype=np.float32) * 10000.0
        events = engine.push_chunk(clipped_audio)
        assert len(events) == 1

    def test_stream_reset_and_reuse(self):
        engine = RollingInferenceEngine(detector=MockFastDetector(score=10.0))
        # Stream 1
        events1 = engine.push_chunk(np.zeros(64600, dtype=np.float32))
        assert len(events1) == 1
        assert events1[0].window_index == 0

        # Reset
        engine.reset()
        assert len(engine.events) == 0
        assert engine.buffer.buffered_samples_count == 0
        assert engine.buffer.total_samples_received == 0

        # Stream 2
        events2 = engine.push_chunk(np.zeros(64600, dtype=np.float32))
        assert len(events2) == 1
        assert events2[0].window_index == 0  # Re-starts from 0

    def test_malformed_inputs(self):
        engine = RollingInferenceEngine(detector=MockFastDetector(score=1.0))
        # Empty array returns empty list
        assert engine.push_chunk(np.empty(0, dtype=np.float32)) == []

        # Multi-dimensional audio (> 2D) should raise ValueError
        with pytest.raises(ValueError):
            engine.push_chunk(np.zeros((10, 10, 10), dtype=np.float32))

        # Invalid type should raise TypeError
        with pytest.raises(TypeError):
            engine.push_chunk({"audio": [1, 2, 3]})
