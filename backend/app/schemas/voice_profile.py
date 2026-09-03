from pydantic import BaseModel
from datetime import datetime

class VoiceProfileResponse(BaseModel):
    profile_id: str
    user_id: str
    created_at: datetime
    model_name: str
    
    class Config:
        from_attributes = True
