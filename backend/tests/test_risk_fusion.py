import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_risk_fusion_logic_all_signals():
    from app.services.risk_fusion_service import calculate_risk
    
    voice = {"voice_integrity_score": 0.9, "confidence": 0.95}  # Highly suspicious (deepfake) -> 0.9 * 0.4 = 0.36
    speaker = {"speaker_similarity_score": 0.1, "match": False, "confidence": 0.9} # Mismatch -> risk=0.9 * 0.2 = 0.18
    intent = {"social_engineering_score": 0.8, "signals": ["urgency"], "confidence": 0.8} # High intent -> 0.8 * 0.15 = 0.12
    action = {"action_risk_score": 0.7, "context_risk_score": 0.7, "signals": ["auth_credential_request"], "confidence": 0.8} # High action -> 0.7 * 0.25 = 0.175
    
    res = calculate_risk(voice, speaker, intent, action)
    
    assert res["overall_risk_score"] > 0.8
    assert res["risk_level"] == "critical"
    assert "speaker_mismatch" in res["contributing_signals"]
    assert "high_voice_manipulation_probability" in res["contributing_signals"]
    assert "urgency" in res["contributing_signals"]

def test_risk_fusion_logic_missing_speaker():
    from app.services.risk_fusion_service import calculate_risk
    
    voice = {"voice_integrity_score": 0.1, "confidence": 0.95}  
    intent = {"social_engineering_score": 0.1, "signals": [], "confidence": 0.8} 
    action = {"action_risk_score": 0.1, "context_risk_score": 0.1, "signals": [], "confidence": 0.8} 
    
    res = calculate_risk(voice, None, intent, action)
    
    # Weights sum to 0.8 (0.4 + 0.15 + 0.25)
    # Score = (0.1*0.4 + 0.1*0.15 + 0.1*0.25) / 0.8 = 0.1
    assert abs(res["overall_risk_score"] - 0.1) < 0.01
    assert res["risk_level"] == "low"

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.asr_service.transcribe_audio")
def test_analyze_session_risk_endpoint(mock_transcribe, mock_analyze):
    mock_analyze.return_value = {"voice_integrity_score": 0.1, "confidence": 0.9, "label": "genuine", "model_name": "test"}
    mock_transcribe.return_value = {"transcript": "hello world", "language": "en", "model_name": "test", "confidence": 0.9}
    
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    session_id = resp.json()["session_id"]
    
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/risk", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    
    risk = body["data"]["risk_analysis"]
    assert risk["risk_level"] == "low"
    assert risk["overall_risk_score"] < 0.25

def test_analyze_session_risk_invalid_session():
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    resp = client.post("/api/v1/sessions/invalid-session/risk", files=files)
    assert resp.status_code == 404
