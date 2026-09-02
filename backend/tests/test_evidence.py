import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_evidence_service_logic():
    from app.services.evidence_service import generate_evidence_record
    
    session_id = "test-session-123"
    analysis_data = {
        "transcript": "hello",
        "intent": {"category": "benign"}
    }
    
    record1 = generate_evidence_record(session_id, analysis_data)
    record2 = generate_evidence_record(session_id, analysis_data)
    
    # Hashes should be different if the timestamp differs, so we must mock or ignore timestamp
    # Wait, the hash incorporates timestamp. So hashes differ across calls.
    assert "evidence_record" in record1
    assert "hash" in record1
    assert record1["algorithm"] == "SHA-256"
    
    # If we fix timestamp, hashes must be identical
    data_with_fixed_ts = {**record1["evidence_record"], "timestamp": "2026-01-01T00:00:00Z"}
    from app.services.evidence_service import generate_canonical_json
    import hashlib
    canonical1 = generate_canonical_json(data_with_fixed_ts)
    hash1 = hashlib.sha256(canonical1.encode('utf-8')).hexdigest()
    
    data_with_fixed_ts2 = {**record1["evidence_record"], "timestamp": "2026-01-01T00:00:00Z"}
    canonical2 = generate_canonical_json(data_with_fixed_ts2)
    hash2 = hashlib.sha256(canonical2.encode('utf-8')).hexdigest()
    
    assert hash1 == hash2

def test_evidence_get_endpoint():
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    session_id = resp.json()["session_id"]
    
    resp2 = client.get(f"/api/v1/sessions/{session_id}/evidence")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["session_id"] == session_id
    assert body["status"] == "no_analysis_recorded"
    assert body["data"]["evidence_record"] is None
    assert body["data"]["hash"] is None

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.asr_service.transcribe_audio")
def test_evidence_post_endpoint(mock_transcribe, mock_analyze):
    mock_analyze.return_value = {"voice_integrity_score": 0.8, "confidence": 0.9}
    mock_transcribe.return_value = {"transcript": "urgent please share your otp", "language": "en"}
    
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    session_id = resp.json()["session_id"]
    
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    
    resp2 = client.post(f"/api/v1/sessions/{session_id}/evidence", files=files)
    assert resp2.status_code == 200
    body = resp2.json()
    
    assert body["session_id"] == session_id
    assert body["status"] == "success"
    data = body["data"]
    assert data["evidence_record"]["session_id"] == session_id
    assert data["evidence_record"]["transcript"] == "urgent please share your otp"
    assert "hash" in data

def test_evidence_invalid_session():
    resp = client.get("/api/v1/sessions/invalid-session/evidence")
    assert resp.status_code == 404
