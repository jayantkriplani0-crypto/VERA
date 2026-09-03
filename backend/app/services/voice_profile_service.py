import uuid
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import VoiceProfileModel
from app.services import speaker_verification_service
from fastapi import HTTPException

def enroll_profile(db: Session, user_id: str, audio_array: np.ndarray, sample_rate: int = 16000) -> VoiceProfileModel:
    if len(audio_array) == 0:
        raise ValueError("Audio array is empty")
        
    embedding = speaker_verification_service.extract_embedding(audio_array, sample_rate)
    
    # Store numpy array as bytes
    embedding_bytes = embedding.tobytes()
    
    profile_id = str(uuid.uuid4())
    
    db_profile = VoiceProfileModel(
        profile_id=profile_id,
        user_id=user_id,
        model_name="anton-l/wav2vec2-base-superb-sv",
        embedding=embedding_bytes
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    return db_profile

def get_profile(db: Session, profile_id: str) -> VoiceProfileModel:
    return db.query(VoiceProfileModel).filter(VoiceProfileModel.profile_id == profile_id).first()

def compare_audio_to_profile(db: Session, profile_id: str, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    profile = get_profile(db, profile_id)
    if not profile:
        raise ValueError("Profile not found")
        
    # Reconstruct the embedding array
    stored_embedding = np.frombuffer(profile.embedding, dtype=np.float32).reshape(1, -1)
    
    new_embedding = speaker_verification_service.extract_embedding(audio_array, sample_rate)
    
    return speaker_verification_service.compare_embeddings(stored_embedding, new_embedding)
