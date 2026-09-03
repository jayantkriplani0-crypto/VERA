import uuid
from sqlalchemy.orm import Session
from app.db.models import SessionModel
from app.schemas.session import SessionCreate

def create_session(db: Session, session_in: SessionCreate) -> SessionModel:
    db_session = SessionModel(
        session_id=str(uuid.uuid4()),
        caller_id=session_in.caller_id
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_session(db: Session, session_id: str) -> SessionModel:
    return db.query(SessionModel).filter(SessionModel.session_id == session_id).first()

def list_sessions(db: Session, limit: int = 50):
    return db.query(SessionModel).order_by(SessionModel.created_at.desc()).limit(limit).all()

def update_session(db: Session, session_id: str, updates: dict) -> SessionModel:
    db_session = get_session(db, session_id)
    if db_session:
        for key, value in updates.items():
            setattr(db_session, key, value)
        db.commit()
        db.refresh(db_session)
    return db_session
