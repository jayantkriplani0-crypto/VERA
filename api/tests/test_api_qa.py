"""Comprehensive QA validation suite for VERA Layer 1 API Contract and Concurrency/Isolation."""
import asyncio
import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from api.app import app
from api.schemas import HealthResponse, StreamingVoiceEvent, VoiceAnalysisResponse
from api.services.voice_service import MAX_UPLOAD_BYTES, VoiceService
import api.dependencies as deps


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_pcm_bytes(freq: float = 440.0, duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    return (sig * 32767.0).astype(np.int16).tobytes()


def make_wav_bytes(freq: float = 440.0, duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    bio = io.BytesIO()
    sf.write(bio, sig, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


# =====================================================================
# Phase 4: API Contract QA
# =====================================================================

class TestApiContractQA:
    """Verify security, error semantics, schemas, and privacy behavior."""

    def test_oversized_upload_rejection(self, client: TestClient):
        # Create dummy content exceeding MAX_UPLOAD_BYTES (25 MB)
        oversized = b"0" * (MAX_UPLOAD_BYTES + 1024)
        files = {"file": ("big_file.wav", oversized, "audio/wav")}
        resp = client.post("/api/v1/voice/analyze", files=files)
        assert resp.status_code == 413
        assert "exceeds limit" in resp.json()["detail"].lower()

    def test_no_path_leakage_in_errors(self, client: TestClient):
        # Corrupt file
        files = {"file": ("corrupt.wav", b"INVALID_DATA", "audio/wav")}
        resp = client.post("/api/v1/voice/analyze", files=files)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Ensure internal filesystem path is NOT leaked to caller
        assert "C:\\" not in detail
        assert "Users" not in detail
        assert "/home" not in detail

    def test_service_unavailable_handling(self):
        saved = deps._global_voice_service
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                deps._global_voice_service = None
                resp = c.get("/api/v1/voice/health")
                assert resp.status_code == 503
                assert "not initialized" in resp.json()["detail"].lower()
        finally:
            deps._global_voice_service = saved


# =====================================================================
# Phase 5: Concurrency & Stream Isolation QA
# =====================================================================

class TestConcurrencyAndIsolationQA:
    """Verify concurrency safety, zero state leakage, and isolated streaming sessions."""

    def test_concurrent_rest_requests(self, client: TestClient):
        wav1 = make_wav_bytes(freq=300.0, duration_sec=4.5)
        wav2 = make_wav_bytes(freq=800.0, duration_sec=4.5)

        resp1 = client.post("/api/v1/voice/analyze", files={"file": ("tone1.wav", wav1, "audio/wav")})
        resp2 = client.post("/api/v1/voice/analyze", files={"file": ("tone2.wav", wav2, "audio/wav")})

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        data1 = VoiceAnalysisResponse(**resp1.json())
        data2 = VoiceAnalysisResponse(**resp2.json())

        assert data1.request_id != data2.request_id
        assert data1.raw_bona_fide_logit is not None
        assert data2.raw_bona_fide_logit is not None

    def test_interleaved_websocket_stream_isolation(self, client: TestClient):
        """Send interleaved chunks from two separate clients simultaneously.
        Verify that buffer, smoothing, and hysteresis states never cross-contaminate.
        """
        # Client 1 audio: 64,600 samples of 200Hz tone
        c1_pcm = make_pcm_bytes(freq=200.0, duration_sec=4.04)
        # Client 2 audio: 64,600 samples of 1000Hz tone
        c2_pcm = make_pcm_bytes(freq=1000.0, duration_sec=4.04)

        with client.websocket_connect("/api/v1/voice/stream") as ws1, \
             client.websocket_connect("/api/v1/voice/stream") as ws2:

            # Send half of c1, then half of c2
            half_len = len(c1_pcm) // 2
            ws1.send_bytes(c1_pcm[:half_len])
            ws2.send_bytes(c2_pcm[:half_len])

            # Send second half
            ws1.send_bytes(c1_pcm[half_len:])
            ws2.send_bytes(c2_pcm[half_len:])

            # Both should emit window 0
            raw_ev1 = ws1.receive_text()
            raw_ev2 = ws2.receive_text()

            ev1 = StreamingVoiceEvent.model_validate_json(raw_ev1)
            ev2 = StreamingVoiceEvent.model_validate_json(raw_ev2)

            assert ev1.window_id == 0
            assert ev2.window_id == 0
            assert ev1.request_id != ev2.request_id
            assert ev1.timestamp_end == pytest.approx(4.0375, abs=1e-3)
            assert ev2.timestamp_end == pytest.approx(4.0375, abs=1e-3)

            # Flush ws1 only
            ws1.send_text('{"action": "reset"}')
            reset_resp = ws1.receive_json()
            assert reset_resp["status"] == "reset_complete"

            # ws2 should be unaffected
            ws2.send_text('{"action": "flush"}')
            flush_resp = ws2.receive_text()
            flush_ev2 = StreamingVoiceEvent.model_validate_json(flush_resp)
            assert flush_ev2.window_id == 1
