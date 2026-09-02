import pytest
from fastapi.testclient import TestClient
from app.main import app
import wave
import io
import os

client = TestClient(app)

def create_dummy_wav():
    # Create a minimal valid wav file in memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 16000 frames (1 second of silence)
        wf.writeframes(b"\x00\x00" * 16000)
    buf.seek(0)
    return buf.read()

def test_upload_audio_success():
    # First, create a session
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    
    # Now upload audio
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/audio", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    data = body["data"]
    assert data["filename"] == "test.wav"
    assert data["sample_rate"] == 16000
    assert data["channels"] == 1
    assert data["status"] == "processed"
    assert 0.9 <= data["duration"] <= 1.1

def test_upload_audio_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session-id/audio", files=files)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"
