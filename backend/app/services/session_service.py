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
