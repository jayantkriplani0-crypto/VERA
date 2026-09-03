"""Health and operational readiness route for VERA Layer 1."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from api.dependencies import get_voice_service
from api.schemas import HealthResponse
from api.services.voice_service import VoiceService

router = APIRouter(prefix="/api/v1/voice", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Voice Authenticity Service Health Check",
    description="Returns service health status, model initialization state, and calibrated decision thresholds.",
)
async def check_health(service: VoiceService = Depends(get_voice_service)) -> HealthResponse:
    """Return health status and operational parameters."""
    return HealthResponse(
        status="healthy" if service.is_healthy else "degraded",
        model_loaded=service.detector is not None,
        calibration_loaded=service.scorer is not None,
        model_name=service.model_name,
        api_version="1.0.0",
        calibrated_boundary=service.calibrated_boundary,
        operational_spoof_threshold=service.operational_threshold,
    )
