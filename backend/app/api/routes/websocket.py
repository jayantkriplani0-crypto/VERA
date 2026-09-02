import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import Optional
import tempfile
import os
import librosa
from app.db.database import get_db
from app.services import session_service
from app.services import voice_integrity_service
from app.services import asr_service
from app.services import intent_service
from app.services import action_context_service
from app.services import risk_fusion_service
from app.services import voice_profile_service
from app.services import policy_service

logger = logging.getLogger("vera.websocket")

router = APIRouter()

@router.websocket("/api/v1/ws/sessions/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    profile_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    VERA Real-Time Audio Analysis WebSocket.

    Audio Protocol
    --------------
    Clients MUST stream audio as complete, self-contained WAV chunks:
      - Format:      WAV (RIFF), 16-bit PCM
      - Sample rate: 16 000 Hz
      - Channels:    Mono (1 channel)
      - Chunk size:  Aim for 2–4 seconds per chunk (~64 KB–128 KB uncompressed).

    The server accumulates binary data until the internal buffer exceeds 100 000 bytes
    (~3 seconds of 16 kHz 16-bit mono), then decodes, analyses, and pushes a JSON
    result back through the same connection.

    Each JSON response contains:
      session_id, transcript, voice_integrity_score, speaker_similarity_score,
      overall_risk_score, risk_level, decision, signals

    On error (e.g., malformed chunk), the server sends {"error": "<reason>"} and
    continues listening — the connection is NOT closed.

    Query Parameters
    ----------------
    profile_id (optional): UUID of a trusted voice profile for speaker verification.
                           Omit for unknown/first-time callers.

    Close Codes
    -----------
    1008 – Session not found (connection rejected before accept).
    """

    # Verify session
    db_session = session_service.get_session(db=db, session_id=session_id)
    if db_session is None:
        await websocket.close(code=1008, reason="Session not found")
        return
        
    await websocket.accept()
    
    buffer = bytearray()
    
    # 2-4 seconds at 16kHz 16-bit mono is ~64KB - 128KB
    # We'll use a ~100KB threshold to trigger processing
    PROCESS_THRESHOLD = 100000 
    
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)
            
            if len(buffer) >= PROCESS_THRESHOLD:
                # Process buffer
                temp_audio_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio.write(buffer)
                        temp_audio_path = temp_audio.name
                    
                    # Clear buffer after writing so memory is freed
                    buffer.clear()
                    
                    # Catch librosa parsing issues on malformed binary
                    y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                    
                    # Run Pipeline
                    voice_result = voice_integrity_service.analyze_voice(y, sr)
                    
                    speaker_result = None
                    if profile_id:
                        try:
                            speaker_result = voice_profile_service.compare_audio_to_profile(db, profile_id, y, sr)
                        except Exception:
                            pass
                            
                    asr_result = asr_service.transcribe_audio(y, sr)
                    transcript = asr_result.get("transcript", "")
                    
                    intent_result = intent_service.analyze_intent(transcript)
                    action_context_result = action_context_service.analyze_action_context(transcript, intent_result)
                    
                    risk_result = risk_fusion_service.calculate_risk(
                        voice_analysis=voice_result,
                        speaker_analysis=speaker_result,
                        intent_analysis=intent_result,
                        action_context_analysis=action_context_result
                    )
                    
                    policy_result = policy_service.evaluate_policy(risk_result, action_context_result)
                    
                    await websocket.send_json({
                        "session_id": session_id,
                        "transcript": transcript,
                        "voice_integrity_score": voice_result.get("voice_integrity_score"),
                        "speaker_similarity_score": speaker_result.get("speaker_similarity_score") if speaker_result else None,
                        "overall_risk_score": risk_result.get("overall_risk_score"),
                        "risk_level": risk_result.get("risk_level"),
                        "decision": policy_result.get("decision"),
                        "signals": risk_result.get("contributing_signals", [])
                    })
                    
                except Exception as e:
                    # Ignore unparseable chunks and wait for more valid data
                    buffer.clear()
                    await websocket.send_json({"error": f"Audio parsing or processing failed: {str(e)}"})
                finally:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
