from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.db.database import Base

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    caller_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")
    risk_level = Column(String, nullable=True)
    decision = Column(String, nullable=True)

from sqlalchemy import LargeBinary

class VoiceProfileModel(Base):
    __tablename__ = "voice_profiles"

    profile_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    embedding = Column(LargeBinary)
