import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

from unittest.mock import patch

def test_verify_voice_success():
    with patch("app.services.voice_integrity_service.analyze_voice") as mock_analyze:
        mock_analyze.return_value = {
            "voice_integrity_score": 0.85,
            "label": "synthetic",
            "model_name": "MelodyMachine/Deepfake-Audio-Detection-V2",
            "confidence": 0.85
        }
        
        resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]
        
        wav_bytes = create_dummy_wav()
        files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
        
        resp2 = client.post(f"/api/v1/sessions/{session_id}/verify", files=files)
        assert resp2.status_code == 200
        body = resp2.json()
        
        assert body["session_id"] == session_id
        assert body["status"] == "success"
        data = body["data"]
        assert data["filename"] == "test_voice.wav"
        assert "voice_integrity" in data
        
        vi = data["voice_integrity"]
        assert "voice_integrity_score" in vi
        assert 0.0 <= vi["voice_integrity_score"] <= 1.0
        assert vi["label"] in ["genuine", "synthetic"]
        assert vi["model_name"] == "MelodyMachine/Deepfake-Audio-Detection-V2"
        assert vi["confidence"] == 0.85

def test_verify_voice_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/verify", files=files)
    assert resp.status_code == 404
