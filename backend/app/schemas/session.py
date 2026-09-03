from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SessionCreate(BaseModel):
    caller_id: Optional[str] = None

class SessionResponse(BaseModel):
    id: int
    session_id: str
    caller_id: Optional[str] = None
    created_at: datetime
    status: str
    risk_level: Optional[str] = None
    decision: Optional[str] = None

    class Config:
        from_attributes = True
