import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

@patch("app.services.asr_service.transcribe_audio")
def test_transcribe_audio_success(mock_transcribe):
    mock_transcribe.return_value = {
        "transcript": "hello world",
        "language": "en",
        "model_name": "openai/whisper-tiny",
        "confidence": 0.95
    }
    
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/transcript", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    data = body["data"]
    assert data["filename"] == "test_voice.wav"
    assert "transcription" in data
    
    tx = data["transcription"]
    assert tx["transcript"] == "hello world"
    assert tx["language"] == "en"
    assert tx["model_name"] == "openai/whisper-tiny"

def test_transcribe_audio_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/transcript", files=files)
    assert resp.status_code == 404
