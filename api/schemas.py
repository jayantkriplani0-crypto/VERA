"""Pydantic v2 data schemas for the VERA Layer 1 Voice Authenticity API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health and model readiness status schema."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall service status (e.g. 'healthy', 'degraded')")
    model_loaded: bool = Field(..., description="True if Spectra-AASIST3 model is loaded into memory")
    calibration_loaded: bool = Field(..., description="True if Platt calibration parameters are loaded")
    model_name: str = Field(..., description="Identifier of the active frozen model")
    api_version: str = Field(default="1.0.0", description="VERA Layer-1 API version")
    calibrated_boundary: float = Field(..., description="Frozen calibrated decision boundary logit (P=0.50)")
    operational_spoof_threshold: float = Field(..., description="Operational calibrated spoof signal threshold (0-100)")


class VoiceAnalysisResponse(BaseModel):
    """Structured response for single-audio voice authenticity analysis."""
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Unique UUID for this analysis request")
    model: str = Field(..., description="Model identifier used for inference")
    raw_bona_fide_logit: float = Field(..., description="Raw bona-fide logit from frozen Spectra-AASIST3")
    calibrated_bona_fide_probability: float = Field(..., ge=0.0, le=1.0, description="Posterior P(Bona Fide | s)")
    calibrated_spoof_probability: float = Field(..., ge=0.0, le=1.0, description="Posterior P(Spoof | s)")
    voice_integrity_score: float = Field(..., ge=0.0, le=100.0, description="Voice integrity scale (0-100, 100=genuine)")
    spoof_signal: float = Field(..., ge=0.0, le=100.0, description="Calibrated spoof signal (0-100, 100=synthetic)")
    decision_confidence: float = Field(..., ge=0.0, le=1.0, description="Distance from 0.5 decision boundary")
    classification: Literal["BONAFIDE", "SPOOF"] = Field(..., description="Binary authenticity decision")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of evaluation",
    )
    duration_seconds: float = Field(..., ge=0.0, description="Duration of processed audio in seconds")
    processing_latency_ms: float = Field(..., ge=0.0, description="Inference compute latency in milliseconds")


class StreamingVoiceEvent(BaseModel):
    """Event emitted over WebSocket whenever a complete sliding window is evaluated."""
    model_config = ConfigDict(extra="forbid")

    event: str = Field(default="voice_authenticity", description="Event name identifier")
    request_id: str = Field(..., description="Session / stream identifier")
    window_id: int = Field(..., ge=0, description="Zero-based sequence index of evaluated window")
    timestamp_start: float = Field(..., ge=0.0, description="Stream start timestamp in seconds")
    timestamp_end: float = Field(..., ge=0.0, description="Stream end timestamp in seconds")
    raw_logit: float = Field(..., description="Raw bona-fide logit from frozen Spectra-AASIST3")
    p_bonafide: float = Field(..., ge=0.0, le=1.0, description="Calibrated posterior P(Bona Fide)")
    p_spoof: float = Field(..., ge=0.0, le=1.0, description="Calibrated posterior P(Spoof)")
    voice_integrity_score: float = Field(..., ge=0.0, le=100.0, description="Calibrated integrity score (0-100)")
    spoof_signal: float = Field(..., ge=0.0, le=100.0, description="Calibrated spoof signal (0-100)")
    smoothed_spoof_signal: float = Field(..., ge=0.0, le=100.0, description="EMA-smoothed spoof signal (0-100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision (0.0-1.0)")
    state: Literal["BONAFIDE", "SUSPICIOUS", "SPOOF"] = Field(..., description="Hysteresis operational state")
    is_alert: bool = Field(..., description="True if state is SPOOF (high-confidence deepfake)")
    latency_ms: float = Field(..., ge=0.0, description="Compute latency for this window in milliseconds")


class StreamControlMessage(BaseModel):
    """Control instruction received over WebSocket from client."""
    model_config = ConfigDict(extra="ignore")

    action: Literal["chunk", "flush", "reset", "close"] = Field(..., description="Action to perform")
    data_b64: Optional[str] = Field(default=None, description="Optional base64-encoded audio chunk")


class ErrorResponse(BaseModel):
    """Standardized API error response schema."""
    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Error category code")
    detail: str = Field(..., description="Detailed human-readable explanation")
    request_id: Optional[str] = Field(default=None, description="Request identifier if available")
