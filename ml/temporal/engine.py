"""Unified Rolling Inference Engine for near-real-time audio streams.

Coordinates:
  1. RollingAudioBuffer (preallocated sample accumulation & 64,600-sample windowing)
  2. Frozen Spectra-AASIST3 Voice Detector (unnormalized bona fide logit extraction)
  3. Frozen Platt Calibrator (Voice Integrity & Spoof Signal on 0-100 scale)
  4. Configurable Temporal Smoother (SMA / EMA mitigation of transient noise)
  5. Hysteresis State Machine (BONAFIDE, SUSPICIOUS, SPOOF with debounce)

Optimized for:
  - Zero-redundancy pre-emphasis without temporary array concatenation
  - Batched forward pass for multi-window arrivals
  - Pre-allocated ring buffer memory reuse
  - Configurable PyTorch CPU threading
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import sys
import time
from typing import Generator, List, Optional
import numpy as np
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.apply_calibration import CalibratedVoiceScorer, DEFAULT_CONFIG_PATH
from ml.preprocessing.audio_loader import load_audio, TARGET_SAMPLE_RATE
from ml.preprocessing.preprocessor import apply_preemphasis
from ml.temporal.rolling_buffer import AudioWindow, RollingAudioBuffer, WINDOW_SIZE_SAMPLES, DEFAULT_HOP_SIZE_SAMPLES
from ml.temporal.smoother import TemporalSmoother, create_smoother
from ml.temporal.state_machine import DetectionState, HysteresisStateMachine
from ml.voice_detector.detector import VoiceAuthenticityDetector


@dataclass(frozen=True)
class TemporalInferenceEvent:
    """Structured event emitted for each processed temporal audio window."""
    window_index: int
    timestamp_sec: float              # Stream-time at window boundary
    window_start_sec: float
    window_end_sec: float
    raw_score: float                  # Unnormalized bona-fide logit from Spectra-AASIST3
    calibrated_prob_bonafide: float   # in [0.0, 1.0]
    calibrated_prob_spoof: float      # in [0.0, 1.0]
    voice_integrity_score: float      # 0–100 scale (100 = confirmed genuine)
    calibrated_spoof_signal: float    # 0–100 scale (100 = confirmed synthetic/spoof)
    decision_confidence: float        # in [0.0, 1.0]
    smoothed_spoof_signal: float      # Temporal smoothed spoof signal
    smoothed_integrity_score: float   # 100.0 - smoothed_spoof_signal
    state: DetectionState             # BONAFIDE / SUSPICIOUS / SPOOF
    state_changed: bool               # True if this event caused a state transition
    is_alert: bool                    # True if state is SPOOF
    latency_ms: float                 # Forward pass + scoring compute latency in ms
    is_flushed: bool = False          # True if generated via flush()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class RollingInferenceEngine:
    """Production-grade streaming inference pipeline for VERA Layer 1."""

    def __init__(
        self,
        detector: Optional[VoiceAuthenticityDetector] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
        window_size_samples: int = WINDOW_SIZE_SAMPLES,
        hop_size_samples: int = DEFAULT_HOP_SIZE_SAMPLES,
        sample_rate: int = TARGET_SAMPLE_RATE,
        smoothing_method: str = "ema",
        smoothing_param: float = 0.35,
        bonafide_threshold: float = 35.0,
        spoof_threshold: float = 65.0,
        dwell_count: int = 2,
        num_threads: Optional[int] = None,
        device: str = "auto",
    ) -> None:
        # Optional thread tuning for CPU inference
        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(num_threads)

        # 1. Detector (frozen Spectra-AASIST3)
        if detector is not None:
            self.detector = detector
        else:
            self.detector = VoiceAuthenticityDetector(device=device)

        # 2. Calibrator (frozen parameters from config.json)
        self.scorer = CalibratedVoiceScorer(config_path=config_path)

        # 3. Rolling Buffer (preallocated contiguous memory)
        self.buffer = RollingAudioBuffer(
            window_size_samples=window_size_samples,
            hop_size_samples=hop_size_samples,
            sample_rate=sample_rate,
        )

        # 4. Smoother
        if smoothing_method.lower() in ("ema", "exponential"):
            self.smoother: TemporalSmoother = create_smoother("ema", alpha=smoothing_param)
        else:
            self.smoother = create_smoother("sma", window_len=int(smoothing_param))

        # 5. Hysteresis State Machine
        self.state_machine = HysteresisStateMachine(
            bonafide_threshold=bonafide_threshold,
            spoof_threshold=spoof_threshold,
            dwell_count=dwell_count,
            initial_state=DetectionState.BONAFIDE,
        )

        self._events_emitted: List[TemporalInferenceEvent] = []

    @property
    def events(self) -> List[TemporalInferenceEvent]:
        """Sequence of all events emitted since initialization or reset."""
        return list(self._events_emitted)

    @property
    def current_state(self) -> DetectionState:
        """Current committed operational detection state."""
        return self.state_machine.current_state

    def _score_and_emit(
        self,
        window: AudioWindow,
        raw_score: float,
        latency_ms: float,
    ) -> TemporalInferenceEvent:
        """Apply calibration, temporal smoothing, and hysteresis state transition."""
        # Platt calibration
        cal_res = self.scorer.calibrate_raw_score(raw_score)

        # Temporal smoothing
        smoothed_spoof = self.smoother.update(cal_res.calibrated_spoof_signal)
        smoothed_integrity = max(0.0, min(100.0, 100.0 - smoothed_spoof))

        # Hysteresis state transition
        committed_state, state_changed = self.state_machine.update(smoothed_spoof)

        event = TemporalInferenceEvent(
            window_index=window.window_index,
            timestamp_sec=window.end_sec,
            window_start_sec=window.start_sec,
            window_end_sec=window.end_sec,
            raw_score=round(raw_score, 6),
            calibrated_prob_bonafide=round(cal_res.calibrated_probability_bonafide, 6),
            calibrated_prob_spoof=round(cal_res.calibrated_probability_spoof, 6),
            voice_integrity_score=round(cal_res.voice_integrity_score, 2),
            calibrated_spoof_signal=round(cal_res.calibrated_spoof_signal, 2),
            decision_confidence=round(cal_res.decision_confidence, 4),
            smoothed_spoof_signal=round(smoothed_spoof, 2),
            smoothed_integrity_score=round(smoothed_integrity, 2),
            state=committed_state,
            state_changed=state_changed,
            is_alert=bool(committed_state == DetectionState.SPOOF),
            latency_ms=round(latency_ms, 2),
            is_flushed=window.is_flushed,
        )

        self._events_emitted.append(event)
        return event

    def _process_single_window(self, window: AudioWindow) -> TemporalInferenceEvent:
        """Process a single 64,600-sample window."""
        t0 = time.perf_counter()
        preprocessed = apply_preemphasis(window.samples, coef=0.97)
        raw_score = self.detector._run_forward(preprocessed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._score_and_emit(window, raw_score, latency_ms)

    def push_chunk(self, chunk: np.ndarray | bytes | list[float]) -> List[TemporalInferenceEvent]:
        """Ingest an incoming audio chunk and return any completed window events."""
        ready_windows = self.buffer.append(chunk)
        if not ready_windows:
            return []

        if len(ready_windows) == 1:
            return [self._process_single_window(ready_windows[0])]

        # Multi-window batched forward pass optimization
        t0 = time.perf_counter()
        preprocessed_batch = [apply_preemphasis(w.samples, coef=0.97) for w in ready_windows]
        raw_scores = self.detector._run_forward_batch(preprocessed_batch)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_window_latency_ms = total_latency_ms / float(len(ready_windows))

        events: List[TemporalInferenceEvent] = []
        for w, score in zip(ready_windows, raw_scores):
            events.append(self._score_and_emit(w, score, per_window_latency_ms))

        return events

    def flush(self) -> List[TemporalInferenceEvent]:
        """Process any remaining buffered audio into a final window and emit its event."""
        trailing_window = self.buffer.flush()
        if trailing_window is not None:
            return [self._process_single_window(trailing_window)]
        return []

    def reset(self) -> None:
        """Reset the buffer, smoother, state machine, and event history."""
        self.buffer.reset()
        self.smoother.reset()
        self.state_machine.reset()
        self._events_emitted.clear()

    def process_file_stream(
        self,
        file_path: str | Path,
        chunk_size_sec: float = 0.5,
        simulate_realtime: bool = False,
    ) -> Generator[TemporalInferenceEvent, None, None]:
        """Stream an audio file in simulated chunks, yielding events as windows become ready."""
        waveform, meta = load_audio(file_path, target_sr=self.buffer.sample_rate)
        chunk_samples = int(chunk_size_sec * self.buffer.sample_rate)

        num_chunks = int(np.ceil(len(waveform) / float(chunk_samples)))

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(len(waveform), start + chunk_samples)
            chunk = waveform[start:end]

            if simulate_realtime and i > 0:
                time.sleep(chunk_size_sec)

            events = self.push_chunk(chunk)
            for ev in events:
                yield ev

        for ev in self.flush():
            yield ev
