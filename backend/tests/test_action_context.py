import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_action_context_service_logic():
    from app.services.action_context_service import analyze_action_context
    
    # Benign
    res1 = analyze_action_context("how is the weather?", {"signals": []})
    assert res1["action_risk_score"] == 0.0
    assert res1["context_risk_score"] == 0.0
    assert res1["action_category"] == "low_risk_action"
    
    # High risk
    intent_analysis = {"signals": ["urgency", "request_auth_code"]}
    res2 = analyze_action_context("urgent give me your otp", intent_analysis)
    
    assert res2["action_risk_score"] >= 0.7
    assert res2["context_risk_score"] >= 0.7
    assert res2["action_category"] == "high_risk_action"
    assert "auth_credential_request" in res2["signals"]
    assert "high_pressure_context" in res2["signals"]

@patch("app.services.asr_service.transcribe_audio")
def test_analyze_session_action_context_endpoint(mock_transcribe):
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
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/action-context", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    data = body["data"]
    
    ac = data["action_context_analysis"]
    assert ac["action_risk_score"] >= 0.7
    assert ac["action_category"] == "high_risk_action"
    assert "auth_credential_request" in ac["signals"]

def test_analyze_session_action_context_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/action-context", files=files)
    assert resp.status_code == 404
