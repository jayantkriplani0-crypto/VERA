"""Service layer for VERA Layer 1 Voice Authenticity.

Orchestrates:
  - Single startup initialization of frozen Spectra-AASIST3 model and Platt calibrator.
  - Safe in-memory single-file audio analysis.
  - Isolated per-client streaming sessions with individual RollingInferenceEngines.
  - Thread-safe, serialized inference execution across concurrent requests on CPU.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import io
from pathlib import Path
import time
from typing import List, Optional
import uuid
import numpy as np
import soundfile as sf
import torch

from calibration.apply_calibration import CalibratedVoiceScorer, DEFAULT_CONFIG_PATH
from ml.preprocessing.audio_loader import TARGET_SAMPLE_RATE, resample_audio
from ml.preprocessing.preprocessor import apply_preemphasis, apply_windowing
from ml.temporal.engine import RollingInferenceEngine, TemporalInferenceEvent
from ml.voice_detector.detector import VoiceAuthenticityDetector
from api.schemas import StreamingVoiceEvent, VoiceAnalysisResponse

# Operational limits
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 Megabytes
MAX_STREAM_DURATION_SEC = 300.0       # 5 minutes maximum continuous stream session


class StreamingSession:
    """Encapsulates a dedicated streaming session for a single WebSocket client."""

    def __init__(
        self,
        service: "VoiceService",
        session_id: str,
        max_duration_sec: float = MAX_STREAM_DURATION_SEC,
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.max_duration_sec = max_duration_sec

        # Dedicated RollingInferenceEngine using shared detector & scorer
        self.engine = RollingInferenceEngine(
            detector=self.service.detector,
            config_path=self.service.config_path,
            smoothing_method="ema",
            smoothing_param=0.35,
            dwell_count=2,
        )
        self.created_at = time.time()
        self.total_audio_received_sec: float = 0.0

    async def push_chunk(self, chunk_data: bytes | np.ndarray) -> List[StreamingVoiceEvent]:
        """Process incoming audio chunk through engine with serialized forward pass."""
        if (time.time() - self.created_at) > self.max_duration_sec:
            raise TimeoutError(f"Session exceeded maximum duration limit of {self.max_duration_sec}s.")

        # Serialize inference forward pass through service lock
        async with self.service.inference_lock:
            events: List[TemporalInferenceEvent] = await asyncio.to_thread(
                self.engine.push_chunk, chunk_data
            )

        api_events: List[StreamingVoiceEvent] = []
        for ev in events:
            api_events.append(self._to_streaming_event(ev))
        return api_events

    async def flush(self) -> List[StreamingVoiceEvent]:
        """Flush trailing buffer samples into a final evaluation window."""
        async with self.service.inference_lock:
            events: List[TemporalInferenceEvent] = await asyncio.to_thread(self.engine.flush)

        return [self._to_streaming_event(ev) for ev in events]

    def reset(self) -> None:
        """Reset internal buffer, smoother, and state machine."""
        self.engine.reset()

    def release(self) -> None:
        """Cleanly release resources upon client disconnect."""
        self.engine.reset()

    def _to_streaming_event(self, ev: TemporalInferenceEvent) -> StreamingVoiceEvent:
        return StreamingVoiceEvent(
            event="voice_authenticity",
            request_id=self.session_id,
            window_id=ev.window_index,
            timestamp_start=ev.window_start_sec,
            timestamp_end=ev.window_end_sec,
            raw_logit=ev.raw_score,
            p_bonafide=ev.calibrated_prob_bonafide,
            p_spoof=ev.calibrated_prob_spoof,
            voice_integrity_score=ev.voice_integrity_score,
            spoof_signal=ev.calibrated_spoof_signal,
            smoothed_spoof_signal=ev.smoothed_spoof_signal,
            confidence=ev.decision_confidence,
            state=ev.state.value,
            is_alert=ev.is_alert,
            latency_ms=ev.latency_ms,
        )


class VoiceService:
    """Singleton service managing the frozen detector, calibrator, and streaming sessions."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: str = "cpu",
        num_threads: int = 6,
    ) -> None:
        self.config_path = Path(config_path)
        self.device = device
        self.num_threads = num_threads

        # Optional thread tuning for CPU
        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[VoiceService] Initializing frozen Spectra-AASIST3 on device='{device}' (threads={num_threads})...")
        t0 = time.perf_counter()
        self.detector = VoiceAuthenticityDetector(device=device)
        t_model = time.perf_counter() - t0
        print(f"[VoiceService] Model loaded in {t_model:.2f}s.")

        print(f"[VoiceService] Loading frozen Platt calibration from '{self.config_path}'...")
        self.scorer = CalibratedVoiceScorer(config_path=self.config_path)
        print(f"[VoiceService] Calibration active: w={self.scorer.w:+.6f}, b={self.scorer.b:+.6f}, threshold={self.scorer.threshold} (boundary s*={-self.scorer.b/self.scorer.w:+.6f})")

        # Concurrency synchronization lock for CPU tensor forward passes
        self.inference_lock = asyncio.Lock()

    @property
    def is_healthy(self) -> bool:
        return self.detector is not None and self.scorer is not None

    @property
    def model_name(self) -> str:
        return self.detector.model_repo

    @property
    def calibrated_boundary(self) -> float:
        return round(-self.scorer.b / self.scorer.w, 6)

    @property
    def operational_threshold(self) -> float:
        return self.scorer.threshold

    async def analyze_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> VoiceAnalysisResponse:
        """Perform single-audio voice authenticity analysis on in-memory bytes."""
        if len(audio_bytes) == 0:
            raise ValueError("Provided audio file is empty (0 bytes).")
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Audio file size ({len(audio_bytes) / (1024*1024):.1f} MB) exceeds limit of 25 MB.")

        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # 1. Read audio in-memory
        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sample_rate = sf.read(bio, dtype="float32")
        except Exception as e:
            raise ValueError(f"Could not decode audio data: {e}") from e

        # Ensure mono
        if data.ndim > 1:
            mono = np.mean(data, axis=1, dtype=np.float32)
        else:
            mono = data.astype(np.float32)

        duration_sec = len(mono) / float(sample_rate)

        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            mono = resample_audio(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

        # Preprocess (0.97 pre-emphasis + 64,600 window length)
        preprocessed = apply_preemphasis(mono, coef=0.97)
        preprocessed = apply_windowing(preprocessed, target_length=64600)

        # 2. Synchronized forward pass
        async with self.inference_lock:
            raw_score = await asyncio.to_thread(self.detector._run_forward, preprocessed)

        # 3. Platt calibration
        cal_res = self.scorer.calibrate_raw_score(raw_score)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        classification = "BONAFIDE" if cal_res.calibrated_spoof_signal < self.scorer.threshold else "SPOOF"

        return VoiceAnalysisResponse(
            request_id=request_id,
            model=self.model_name,
            raw_bona_fide_logit=round(raw_score, 6),
            calibrated_bona_fide_probability=round(cal_res.calibrated_probability_bonafide, 6),
            calibrated_spoof_probability=round(cal_res.calibrated_probability_spoof, 6),
            voice_integrity_score=round(cal_res.voice_integrity_score, 2),
            spoof_signal=round(cal_res.calibrated_spoof_signal, 2),
            decision_confidence=round(cal_res.decision_confidence, 4),
            classification=classification,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration_sec, 2),
            processing_latency_ms=round(latency_ms, 2),
        )

    def create_streaming_session(self, session_id: Optional[str] = None) -> StreamingSession:
        """Create a new isolated streaming session for a connected WebSocket client."""
        sid = session_id or str(uuid.uuid4())
        return StreamingSession(service=self, session_id=sid)
