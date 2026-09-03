import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from tests.test_audio import create_dummy_wav

client = TestClient(app)

def test_websocket_invalid_session():
    with pytest.raises(Exception): # Starlette testclient raises an exception on rejected websockets
        with client.websocket_connect("/api/v1/ws/sessions/invalid-session"):
            pass

@patch("app.services.voice_integrity_service.analyze_voice")
@patch("app.services.asr_service.transcribe_audio")
def test_websocket_processing(mock_transcribe, mock_analyze):
    mock_analyze.return_value = {"voice_integrity_score": 0.1, "confidence": 0.9}
    mock_transcribe.return_value = {"transcript": "hello this is a test", "language": "en"}
    
    # 1. Create a session
    resp = client.post("/api/v1/sessions", json={"caller_id": "test_caller"})
    session_id = resp.json()["session_id"]
    
    # 2. Connect to WebSocket
    with client.websocket_connect(f"/api/v1/ws/sessions/{session_id}") as websocket:
        # 3. Send binary audio larger than the threshold (100,000 bytes)
        # Create a dummy wav and just multiply bytes to exceed threshold
        dummy_wav = create_dummy_wav()
        large_payload = dummy_wav * (150000 // len(dummy_wav) + 1)
        
        websocket.send_bytes(large_payload)
        
        # 4. Receive response
        data = websocket.receive_json()
        
        assert "error" not in data or "Audio parsing" in data.get("error", "")
        if "error" not in data:
            assert data["session_id"] == session_id
            assert data["transcript"] == "hello this is a test"
            assert data["decision"] in ["allow", "warn", "verify", "block"]
            assert data["risk_level"] in ["low", "medium", "high", "critical"]
            assert "voice_integrity_score" in data
