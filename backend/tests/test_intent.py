import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_intent_service_logic():
    from app.services.intent_service import analyze_intent
    
    # Benign
    res1 = analyze_intent("Hello, how is the weather today?")
    assert res1["social_engineering_score"] == 0.0
    assert res1["intent_category"] == "benign"
    assert len(res1["signals"]) == 0
    
    # Suspicious
    res2 = analyze_intent("I need you to wire transfer money right now to this bank fraud department.")
    assert res2["social_engineering_score"] >= 0.7
    assert "request_payment" in res2["signals"]
    assert "urgency" in res2["signals"]
    assert "impersonation_claim" in res2["signals"]
    assert res2["intent_category"] == "high_risk_scam"

@patch("app.services.asr_service.transcribe_audio")
def test_analyze_session_intent_endpoint(mock_transcribe):
    mock_transcribe.return_value = {
        "transcript": "urgent please share your otp and pin immediately",
        "language": "en",
        "model_name": "openai/whisper-tiny",
        "confidence": 0.99
    }
    
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/intent", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    data = body["data"]
    assert data["transcript"] == "urgent please share your otp and pin immediately"
    
    intent = data["intent_analysis"]
    assert intent["social_engineering_score"] >= 0.7
    assert intent["intent_category"] == "high_risk_scam"
    assert "urgency" in intent["signals"]
    assert "request_auth_code" in intent["signals"]

def test_analyze_session_intent_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/intent", files=files)
    assert resp.status_code == 404
