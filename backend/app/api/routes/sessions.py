from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
import os

from app.db.database import get_db
from app.schemas.session import SessionCreate, SessionResponse
from app.services import session_service, evidence_service

router = APIRouter()

# ---------------------------------------------------------------------------
# Helper – standard response envelope
# ---------------------------------------------------------------------------
def ok(session_id: str, data: dict) -> dict:
    return {"session_id": session_id, "status": "success", "data": data, "error": None}


# ---------------------------------------------------------------------------
# Session management (keep original shape – used by response_model)
# ---------------------------------------------------------------------------

@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(session_in: SessionCreate, db: Session = Depends(get_db)):
    return session_service.create_session(db=db, session_in=session_in)



@router.get("", response_model=list[SessionResponse])
async def list_sessions_endpoint(limit: int = 50, db: Session = Depends(get_db)):
    return session_service.list_sessions(db, limit=limit)

@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    db_session = session_service.get_session(db=db, session_id=session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return db_session


# ---------------------------------------------------------------------------
# Shared audio validation helper
# ---------------------------------------------------------------------------
ALLOWED_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a", ".aac"}

def _validate_audio(file: UploadFile, session_id: str, db: Session):
    """Raises HTTPException if session or audio format is invalid.

    Accepts WAV, MP3, OGG, FLAC (soundfile-native) and also WebM/M4A/AAC
    that browsers may produce via MediaRecorder — audio_service will attempt
    to decode them via librosa/audioread as a fallback.
    """
    import logging
    logger = logging.getLogger("vera.sessions")
    db_session = session_service.get_session(db=db, session_id=session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _, ext = os.path.splitext(file.filename.lower())
    content_type = file.content_type or ""
    logger.info(
        "Audio upload: filename=%r ext=%r content_type=%r",
        file.filename, ext, content_type,
    )
    if ext not in ALLOWED_EXTS and not content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: extension={ext!r}, "
                f"content_type={content_type!r}. "
                "Accepted: wav, mp3, ogg, flac, webm, m4a, aac."
            ),
        )
    return db_session


# ---------------------------------------------------------------------------
# POST /{session_id}/audio  – ingestion only, no ML
# ---------------------------------------------------------------------------

@router.post("/{session_id}/audio")
async def upload_audio(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_audio(file, session_id, db)
    from app.services import audio_service
    try:
        result = await audio_service.process_uploaded_audio(session_id, file)
        return ok(session_id, result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /{session_id}/verify  – voice integrity only
# ---------------------------------------------------------------------------

@router.post("/{session_id}/verify")
async def verify_voice(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, voice_integrity_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = voice_integrity_service.analyze_voice(y, sr)
    return ok(session_id, {"filename": filename, "voice_integrity": result})


# ---------------------------------------------------------------------------
# POST /{session_id}/transcript  – ASR only
# ---------------------------------------------------------------------------

@router.post("/{session_id}/transcript")
async def transcribe_session_audio(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = asr_service.transcribe_audio(y, sr)
    return ok(session_id, {"filename": filename, "transcription": result})


# ---------------------------------------------------------------------------
# POST /{session_id}/intent  – ASR → Intent
# ---------------------------------------------------------------------------

@router.post("/{session_id}/intent")
async def analyze_session_intent(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service, intent_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    asr_result = asr_service.transcribe_audio(y, sr)
    transcript = asr_result.get("transcript", "")
    intent_result = intent_service.analyze_intent(transcript)
    return ok(session_id, {"filename": filename, "transcript": transcript, "intent_analysis": intent_result})


# ---------------------------------------------------------------------------
# POST /{session_id}/action-context  – ASR → Intent → Action/Context
# ---------------------------------------------------------------------------

@router.post("/{session_id}/action-context")
async def analyze_session_action_context(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service, intent_service, action_context_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    asr_result = asr_service.transcribe_audio(y, sr)
    transcript = asr_result.get("transcript", "")
    intent_result = intent_service.analyze_intent(transcript)
    action_context_result = action_context_service.analyze_action_context(transcript, intent_result)
    return ok(session_id, {"filename": filename, "transcript": transcript, "action_context_analysis": action_context_result})


# ---------------------------------------------------------------------------
# POST /{session_id}/risk  – full pipeline except policy/evidence
# ---------------------------------------------------------------------------

@router.post("/{session_id}/risk")
async def analyze_session_risk(
    session_id: str,
    file: UploadFile = File(...),
    profile_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service, intent_service, action_context_service
    from app.services import voice_integrity_service, risk_fusion_service, voice_profile_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    voice_result = voice_integrity_service.analyze_voice(y, sr)

    speaker_result = None
    if profile_id:
        try:
            speaker_result = voice_profile_service.compare_audio_to_profile(db, profile_id, y, sr)
        except Exception:
            pass  # Unknown caller – speaker component excluded from fusion

    asr_result = asr_service.transcribe_audio(y, sr)
    transcript = asr_result.get("transcript", "")
    intent_result = intent_service.analyze_intent(transcript)
    action_context_result = action_context_service.analyze_action_context(transcript, intent_result)

    risk_result = risk_fusion_service.calculate_risk(
        voice_analysis=voice_result,
        speaker_analysis=speaker_result,
        intent_analysis=intent_result,
        action_context_analysis=action_context_result,
    )
    return ok(session_id, {"filename": filename, "transcript": transcript, "risk_analysis": risk_result})


# ---------------------------------------------------------------------------
# POST /{session_id}/decision  – full pipeline + policy
# ---------------------------------------------------------------------------

@router.post("/{session_id}/decision")
async def evaluate_session_decision(
    session_id: str,
    file: UploadFile = File(...),
    profile_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service, intent_service, action_context_service
    from app.services import voice_integrity_service, risk_fusion_service, voice_profile_service, policy_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        action_context_analysis=action_context_result,
    )
    policy_result = policy_service.evaluate_policy(risk_result, action_context_result)
    
    session_service.update_session(db, session_id, {
        "risk_level": risk_result.get("risk_level"),
        "decision": policy_result.get("decision"),
        "status": "completed"
    })
    
    return ok(session_id, {"filename": filename, "policy": policy_result})


# ---------------------------------------------------------------------------
# GET  /{session_id}/evidence – status check (no audio)
# POST /{session_id}/evidence – full pipeline + evidence record + hash
# ---------------------------------------------------------------------------

@router.get("/{session_id}/evidence")
async def get_session_evidence(session_id: str, db: Session = Depends(get_db)):
    db_session = session_service.get_session(db=db, session_id=session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "status": "no_analysis_recorded",
        "data": {"evidence_record": None, "hash": None, "algorithm": "SHA-256"},
        "error": None,
    }


@router.post("/{session_id}/evidence")
async def generate_session_evidence(
    session_id: str,
    file: UploadFile = File(...),
    profile_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _validate_audio(file, session_id, db)
    from app.services import audio_service, asr_service, intent_service, action_context_service
    from app.services import voice_integrity_service, risk_fusion_service, voice_profile_service, policy_service
    try:
        y, sr, _, filename = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        action_context_analysis=action_context_result,
    )
    policy_result = policy_service.evaluate_policy(risk_result, action_context_result)
    
    session_service.update_session(db, session_id, {
        "risk_level": risk_result.get("risk_level"),
        "decision": policy_result.get("decision"),
        "status": "completed"
    })

    analysis_data = {
        "voice_analysis": voice_result,
        "speaker_analysis": speaker_result,
        "transcript": transcript,
        "intent": intent_result,
        "action_context": action_context_result,
        "risk": risk_result,
        "policy_decision": policy_result,
    }
    evidence = evidence_service.generate_evidence_record(session_id, analysis_data)
    return ok(session_id, {"filename": filename, **evidence})
