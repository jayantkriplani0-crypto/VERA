"""FastAPI dependency injection providers."""
from __future__ import annotations

from typing import Optional
from fastapi import HTTPException, status

from api.services.voice_service import VoiceService

_global_voice_service: Optional[VoiceService] = None


def set_voice_service(service: VoiceService) -> None:
    """Register the initialized VoiceService instance at startup."""
    global _global_voice_service
    _global_voice_service = service


def get_voice_service() -> VoiceService:
    """Dependency provider returning the shared VoiceService singleton."""
    if _global_voice_service is None or not _global_voice_service.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice Authenticity service is not initialized or model is unavailable.",
        )
    return _global_voice_service
