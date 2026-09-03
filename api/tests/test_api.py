"""Comprehensive automated tests for VERA Layer 1 Voice Authenticity API."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from api.app import app
from api.schemas import HealthResponse, StreamingVoiceEvent, VoiceAnalysisResponse


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient that triggers FastAPI startup and shutdown lifespans."""
    with TestClient(app) as test_client:
        yield test_client


def create_in_memory_wav(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    """Generate in-memory mono WAV bytes containing a multi-tone synthetic audio signal."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), dtype=np.float32)
    audio = 0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


# =====================================================================
# 1. Health Endpoint Tests
# =====================================================================

class TestHealthEndpoint:
    """Verify GET /api/v1/voice/health."""

    def test_health_check_success(self, client: TestClient):
        response = client.get("/api/v1/voice/health")
        assert response.status_code == 200

        data = response.json()
        validated = HealthResponse(**data)
        assert validated.status == "healthy"
        assert validated.model_loaded is True
        assert validated.calibration_loaded is True
        assert "Spectra-AASIST3" in validated.model_name
        assert validated.api_version == "1.0.0"
        assert validated.calibrated_boundary == pytest.approx(2.78162, abs=1e-4)
        assert validated.operational_spoof_threshold == 50.0


# =====================================================================
# 2. Single-Audio Analysis Endpoint Tests
# =====================================================================

class TestAnalyzeEndpoint:
    """Verify POST /api/v1/voice/analyze."""

    def test_analyze_valid_wav(self, client: TestClient):
        wav_bytes = create_in_memory_wav(duration_sec=4.5)
        files = {"file": ("test_audio.wav", wav_bytes, "audio/wav")}

        response = client.post("/api/v1/voice/analyze", files=files)
        assert response.status_code == 200

        data = response.json()
        validated = VoiceAnalysisResponse(**data)
        assert validated.request_id is not None
        assert "Spectra-AASIST3" in validated.model
        assert validated.raw_bona_fide_logit is not None
        assert 0.0 <= validated.calibrated_bona_fide_probability <= 1.0
        assert 0.0 <= validated.calibrated_spoof_probability <= 1.0
        assert 0.0 <= validated.voice_integrity_score <= 100.0
        assert 0.0 <= validated.spoof_signal <= 100.0
        assert 0.0 <= validated.decision_confidence <= 1.0
        assert validated.classification in ("BONAFIDE", "SPOOF")
        assert validated.duration_seconds == pytest.approx(4.5, abs=0.1)
        assert validated.processing_latency_ms > 0.0

    @pytest.mark.skipif(
        not Path("data/asvspoof2019/eval/flac/LA_E_1025210.flac").is_file(),
        reason="Requires downloaded evaluation FLAC file",
    )
    def test_analyze_real_flac_file(self, client: TestClient):
        with open("data/asvspoof2019/eval/flac/LA_E_1025210.flac", "rb") as f:
            flac_bytes = f.read()

        files = {"file": ("LA_E_1025210.flac", flac_bytes, "audio/flac")}
        response = client.post("/api/v1/voice/analyze", files=files)
        assert response.status_code == 200

        data = response.json()
        validated = VoiceAnalysisResponse(**data)
        assert validated.classification == "SPOOF"
        assert validated.spoof_signal > 90.0

    def test_analyze_empty_file_error(self, client: TestClient):
        files = {"file": ("empty.wav", b"", "audio/wav")}
        response = client.post("/api/v1/voice/analyze", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_analyze_corrupt_audio_error(self, client: TestClient):
        files = {"file": ("corrupt.wav", b"NOT_A_WAV_HEADER_DATA_12345", "audio/wav")}
        response = client.post("/api/v1/voice/analyze", files=files)
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_analyze_unsupported_format_error(self, client: TestClient):
        files = {"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")}
        response = client.post("/api/v1/voice/analyze", files=files)
        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"].lower()


# =====================================================================
# 3. WebSocket Streaming Tests
# =====================================================================

class TestWebSocketStreaming:
    """Verify WebSocket /api/v1/voice/stream."""

    def test_streaming_chunk_ingestion(self, client: TestClient):
        with client.websocket_connect("/api/v1/voice/stream") as ws:
            # Stream 5 seconds of audio in 100ms packets (1600 samples = 3200 bytes int16)
            chunk_samples = 1600
            t = np.linspace(0, 0.1, chunk_samples, dtype=np.float32)
            chunk_audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()

            events_received = []

            # Ingest 50 chunks (5.0 seconds = 80,000 samples -> crosses 64,600 window threshold)
            for _ in range(50):
                ws.send_bytes(chunk_audio)

            # Window 0 should have been emitted
            data = ws.receive_text()
            event = StreamingVoiceEvent.model_validate_json(data)
            assert event.event == "voice_authenticity"
            assert event.window_id == 0
            assert event.timestamp_start == 0.0
            assert event.timestamp_end == pytest.approx(4.0375, abs=1e-3)
            assert 0.0 <= event.voice_integrity_score <= 100.0
            assert 0.0 <= event.spoof_signal <= 100.0
            assert event.state in ("BONAFIDE", "SUSPICIOUS", "SPOOF")
            assert event.latency_ms > 0.0

            # Test Flush control frame
            ws.send_text('{"action": "flush"}')
            flush_data = ws.receive_text()
            flush_event = StreamingVoiceEvent.model_validate_json(flush_data)
            assert flush_event.window_id == 1

    def test_multi_client_stream_isolation(self, client: TestClient):
        """Verify two simultaneous WebSocket clients maintain completely isolated states."""
        with client.websocket_connect("/api/v1/voice/stream") as ws1, \
             client.websocket_connect("/api/v1/voice/stream") as ws2:

            # Send 64,600 samples (4.04s) to WS 1
            audio1 = (np.ones(64600, dtype=np.float32) * 1000).astype(np.int16).tobytes()
            ws1.send_bytes(audio1)

            ev1_raw = ws1.receive_text()
            ev1 = StreamingVoiceEvent.model_validate_json(ev1_raw)
            assert ev1.window_id == 0

            # WS 2 sends a flush on empty buffer
            ws2.send_text('{"action": "reset"}')
            reset_msg = ws2.receive_json()
            assert reset_msg["status"] == "reset_complete"

            # WS 1 session ID is distinct from WS 2
            assert ev1.request_id is not None

    def test_websocket_control_messages(self, client: TestClient):
        with client.websocket_connect("/api/v1/voice/stream") as ws:
            # Test reset action
            ws.send_text('{"action": "reset"}')
            resp = ws.receive_json()
            assert resp["status"] == "reset_complete"

            # Test malformed control message
            ws.send_text('NOT_JSON_STRING')
            err_resp = ws.receive_json()
            assert "error" in err_resp
            assert err_resp["error"] == "malformed_control_message"
