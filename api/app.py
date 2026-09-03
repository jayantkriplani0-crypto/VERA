"""Main FastAPI application for VERA Layer 1 Voice Authenticity API."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import set_voice_service
from api.routes.health import router as health_router
from api.routes.voice import router as voice_router
from api.services.voice_service import VoiceService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads frozen models once at application startup."""
    print("[Lifespan] Starting VERA Layer 1 Voice Authenticity API...")
    # Initialize VoiceService singleton (loads Spectra-AASIST3 model and Platt calibrator once)
    service = VoiceService(device="cpu", num_threads=6)
    set_voice_service(service)
    print("[Lifespan] Model and calibration successfully registered in dependency injector.")
    try:
        yield
    finally:
        print("[Lifespan] Shutting down VERA Layer 1 Voice Authenticity API...")


def create_app() -> FastAPI:
    """Application factory for VERA Layer 1 Voice Authenticity API."""
    application = FastAPI(
        title="VERA Layer-1: Voice Authenticity API",
        version="1.0.0",
        description=(
            "Production-ready near-real-time voice authenticity detection service using "
            "the frozen `lab260/Spectra-AASIST3` anti-spoofing transformer model and "
            "empirical Platt calibration."
        ),
        lifespan=lifespan,
    )

    # Enable CORS for web and microservice clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    application.include_router(health_router)
    application.include_router(voice_router)

    return application


# Global application instance for uvicorn
app = create_app()
