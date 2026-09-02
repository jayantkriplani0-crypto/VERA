from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.voice_profile import VoiceProfileResponse
from app.services import voice_profile_service, audio_service
import os

router = APIRouter()

@router.post("", response_model=VoiceProfileResponse, status_code=status.HTTP_201_CREATED)
async def enroll_voice_profile(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    allowed_exts = {".wav", ".mp3", ".ogg", ".flac"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_exts and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio format")
        
    try:
        y, sr, _, _ = await audio_service.load_and_normalize_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        profile = voice_profile_service.enroll_profile(db, user_id, y, sr)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{profile_id}", response_model=VoiceProfileResponse)
def get_voice_profile(profile_id: str, db: Session = Depends(get_db)):
    profile = voice_profile_service.get_profile(db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile
