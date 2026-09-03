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
    action = {"action_risk_score": 0.8, "context_risk_score": 0.8, "signals": ["auth_credential_request"], "confidence": 0.8} # High action -> 0.8 * 0.25 = 0.20
    
    res = calculate_risk(voice, speaker, intent, action)
    
    assert res["overall_risk_score"] >= 0.85
    assert res["risk_level"] == "critical"
    assert "speaker_mismatch" in res["contributing_signals"]
    assert "high_voice_manipulation_probability" in res["contributing_signals"]
    assert "urgency" in res["contributing_signals"]

def test_risk_fusion_test_a():
    # Test A: voice = 0.50, speaker = unavailable, intent = 0, action = 0, context = 0, signals = []
    # Expected: risk <= 0.10, risk_level = LOW
    from app.services.risk_fusion_service import calculate_risk
    voice = {"voice_integrity_score": 0.50, "confidence": 1.0}
    intent = {"social_engineering_score": 0.0, "signals": [], "confidence": 1.0}
    action = {"action_risk_score": 0.0, "context_risk_score": 0.0, "signals": [], "confidence": 1.0}
    res = calculate_risk(voice, None, intent, action)
    assert res["overall_risk_score"] <= 0.10
    assert res["risk_level"] == "low"

def test_risk_fusion_test_b():
    # Test B: voice = 0.50, speaker = unavailable, intent contains suspicious OTP request, action contains credential disclosure
    # Expected: NOT LOW. Must escalate appropriately.
    from app.services.risk_fusion_service import calculate_risk
    voice = {"voice_integrity_score": 0.50, "confidence": 1.0}
    intent = {"social_engineering_score": 0.8, "signals": ["request_auth_code"], "confidence": 1.0}
    action = {"action_risk_score": 0.8, "context_risk_score": 0.8, "signals": ["auth_credential_request"], "confidence": 1.0}
    res = calculate_risk(voice, None, intent, action)
    assert res["risk_level"] != "low"
    assert res["overall_risk_score"] > 0.10

def test_risk_fusion_test_c():
    # Test C: voice = 0.90, no suspicious transcript
    # Originally expected HIGH/CRITICAL, but due to deepfake model hallucinations on real
    # browser recordings, we now MUST clamp this to LOW if there are no other signals.
    from app.services.risk_fusion_service import calculate_risk
    voice = {"voice_integrity_score": 0.90, "confidence": 1.0}
    intent = {"social_engineering_score": 0.0, "signals": [], "confidence": 1.0}
    action = {"action_risk_score": 0.0, "context_risk_score": 0.0, "signals": [], "confidence": 1.0}
    res = calculate_risk(voice, None, intent, action)
    assert res["risk_level"] == "low"
    assert res["overall_risk_score"] <= 0.10

def test_risk_fusion_test_d():
    # Test D: voice = 0.20, intent = 0, action = 0, speaker unavailable
    # Expected: LOW
    from app.services.risk_fusion_service import calculate_risk
    voice = {"voice_integrity_score": 0.20, "confidence": 1.0}
    intent = {"social_engineering_score": 0.0, "signals": [], "confidence": 1.0}
    action = {"action_risk_score": 0.0, "context_risk_score": 0.0, "signals": [], "confidence": 1.0}
    res = calculate_risk(voice, None, intent, action)
    assert res["risk_level"] == "low"
    assert res["overall_risk_score"] <= 0.30

def test_risk_fusion_test_e():
    # Test E: voice = 0.50, intent = 0, action = 0, speaker has a strong mismatch
    # Expected: NOT LOW, because speaker mismatch is a genuine suspicious signal.
    from app.services.risk_fusion_service import calculate_risk
    voice = {"voice_integrity_score": 0.50, "confidence": 1.0}
    speaker = {"speaker_similarity_score": 0.1, "match": False, "confidence": 1.0}
    intent = {"social_engineering_score": 0.0, "signals": [], "confidence": 1.0}
    action = {"action_risk_score": 0.0, "context_risk_score": 0.0, "signals": [], "confidence": 1.0}
    res = calculate_risk(voice, speaker, intent, action)
    assert res["risk_level"] != "low"
    assert res["overall_risk_score"] > 0.10

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
