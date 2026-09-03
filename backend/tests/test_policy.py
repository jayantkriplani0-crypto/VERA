import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_policy_logic_no_escalation():
    from app.services.policy_service import evaluate_policy
    
    risk_analysis = {"risk_level": "medium"}
    action_context = {"signals": ["harmless_signal"]}
    
    result = evaluate_policy(risk_analysis, action_context)
    
    assert result["decision"] == "warn"
    assert result["escalated"] is False

def test_policy_logic_escalation():
    from app.services.policy_service import evaluate_policy
    
    risk_analysis = {"risk_level": "high"}
    action_context = {"signals": ["auth_credential_request"]} # Critical signal
    
    result = evaluate_policy(risk_analysis, action_context)
    
    assert result["decision"] == "block"
    assert result["escalated"] is True

def test_policy_logic_already_critical():
    from app.services.policy_service import evaluate_policy
    
    risk_analysis = {"risk_level": "critical"}
    action_context = {"signals": ["auth_credential_request"]}
    
    result = evaluate_policy(risk_analysis, action_context)
    
    # Can't escalate beyond block
    assert result["decision"] == "block"
    assert result["escalated"] is False

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.asr_service.transcribe_audio")
def test_analyze_session_decision_endpoint(mock_transcribe, mock_analyze):
    mock_analyze.return_value = {"voice_integrity_score": 0.8, "confidence": 0.9} # High voice risk
    mock_transcribe.return_value = {"transcript": "urgent please share your otp and pin immediately", "language": "en", "confidence": 0.99} # Triggers critical action
    
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    session_id = resp.json()["session_id"]
    
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/decision", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    
    policy = body["data"]["policy"]
    assert "decision" in policy
    assert policy["decision"] in ["allow", "warn", "verify", "block"]
    assert policy["decision"] == "block" # Highly risky transcript + high risk deepfake

def test_analyze_session_decision_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/decision", files=files)
    assert resp.status_code == 404
