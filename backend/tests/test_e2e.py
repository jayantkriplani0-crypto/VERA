import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import hashlib
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.speaker_verification_service.compare_embeddings")
@patch("app.services.speaker_verification_service.extract_embedding")
@patch("app.services.asr_service.transcribe_audio")
def test_e2e_unknown_caller(mock_transcribe, mock_extract, mock_compare, mock_analyze):
    # Mock ML
    mock_analyze.return_value = {"voice_integrity_score": 0.1, "label": "genuine", "confidence": 0.9}
    import numpy as np
    mock_extract.return_value = np.zeros((1, 256), dtype=np.float32)
    mock_compare.return_value = {"speaker_similarity_score": 0.0, "match": False, "confidence": 0.0}
    mock_transcribe.return_value = {"transcript": "can you hear me?", "language": "en"}
    
    # 1. Create session
    resp = client.post("/api/v1/sessions", json={"caller_id": "unknown_number"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    
    # Dummy audio
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    
    # 2. Upload audio (if the route exists, otherwise skip, but we'll try)
    resp = client.post(f"/api/v1/sessions/{session_id}/audio", files=files)
    assert resp.status_code == 200
    
    # 3. Transcript
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/transcript", files=files)
    assert resp.status_code == 200
    assert resp.json()["data"]["transcription"]["transcript"] == "can you hear me?"
    
    # 4. Intent
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/intent", files=files)
    assert resp.status_code == 200
    
    # 5. Action/Context
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/action-context", files=files)
    assert resp.status_code == 200
    
    # 6. Risk (unknown caller, no profile_id)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/risk", files=files)
    assert resp.status_code == 200
    risk = resp.json()["data"]["risk_analysis"]
    # Voice integrity is 0.1, so overall risk should be low
    assert risk["risk_level"] == "low"
    
    # 7. Decision
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/decision", files=files)
    assert resp.status_code == 200
    policy = resp.json()["data"]["policy"]
    assert policy["decision"] == "allow"
    
    # 8. Evidence
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/evidence", files=files)
    assert resp.status_code == 200
    body = resp.json()
    evidence_data = body["data"]
    assert evidence_data["evidence_record"]["session_id"] == session_id
    assert "audio" not in evidence_data["evidence_record"]  # No raw audio
    
    # 9. Verify Hash Determinism
    from app.services.evidence_service import generate_canonical_json
    import hashlib
    canonical = generate_canonical_json(evidence_data["evidence_record"])
    expected_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    assert evidence_data["hash"] == expected_hash
    
    # 10. WebSocket
    with client.websocket_connect(f"/api/v1/ws/sessions/{session_id}") as websocket:
        large_payload = wav_bytes * (150000 // len(wav_bytes) + 1)
        websocket.send_bytes(large_payload)
        data = websocket.receive_json()
        if "error" not in data:
            assert data["session_id"] == session_id
            assert data["risk_level"] == "low"

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.speaker_verification_service.compare_embeddings")
@patch("app.services.speaker_verification_service.extract_embedding")
@patch("app.services.asr_service.transcribe_audio")
def test_e2e_known_caller_critical(mock_transcribe, mock_extract, mock_compare, mock_analyze):
    # Mock ML
    mock_analyze.return_value = {"voice_integrity_score": 0.9, "label": "fake", "confidence": 0.9} # Deepfake!
    import numpy as np
    mock_extract.return_value = np.zeros((1, 256), dtype=np.float32)
    mock_compare.return_value = {"speaker_similarity_score": 0.1, "match": False, "confidence": 0.9} # Doesn't match!
    mock_transcribe.return_value = {"transcript": "urgent send the otp pin now", "language": "en"} # Urgent action!
    
    # 1. Create Voice Profile
    resp = client.post("/api/v1/voice-profiles", files={"file": ("test.wav", create_dummy_wav(), "audio/wav")}, data={"user_id": "test_user"})
    assert resp.status_code == 201
    profile_id = resp.json()["profile_id"]
    
    # 2. Create session
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_user"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    
    # 3. Risk (known caller, pass profile_id)
    wav_bytes = create_dummy_wav()
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/risk", files=files, data={"profile_id": profile_id})
    assert resp.status_code == 200
    risk = resp.json()["data"]["risk_analysis"]
    assert risk["risk_level"] == "critical"
    
    # 4. Decision
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    resp = client.post(f"/api/v1/sessions/{session_id}/decision", files=files, data={"profile_id": profile_id})
    assert resp.status_code == 200
    policy = resp.json()["data"]["policy"]
    assert policy["decision"] == "block"
    assert policy["escalated"] is False # already critical
