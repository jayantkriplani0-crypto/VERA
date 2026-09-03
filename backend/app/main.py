import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, websocket, sessions, voice_profiles
from app.db.database import engine, Base

logger = logging.getLogger("vera.startup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload all ML models at startup to eliminate per-request cold-start latency."""
    logger.info("VERA startup: preloading ML models...")

    try:
        from app.services.voice_integrity_service import get_model as vi_get
        vi_get()
        logger.info("  [OK] voice_integrity  MelodyMachine/Deepfake-Audio-Detection-V2")
    except Exception as e:
        logger.error(f"  [FAIL] voice_integrity model could not load: {e}")
        raise RuntimeError(f"Startup failed: voice_integrity model error: {e}") from e

    try:
        from app.services.speaker_verification_service import get_model as sv_get
        sv_get()
        logger.info("  [OK] speaker_verification  anton-l/wav2vec2-base-superb-sv")
    except Exception as e:
        logger.error(f"  [FAIL] speaker_verification model could not load: {e}")
        raise RuntimeError(f"Startup failed: speaker_verification model error: {e}") from e

    try:
        from app.services.asr_service import get_model as asr_get
        asr_get()
        logger.info("  [OK] asr  openai/whisper-tiny")
    except Exception as e:
        logger.error(f"  [FAIL] asr model could not load: {e}")
        raise RuntimeError(f"Startup failed: asr model error: {e}") from e

    logger.info("VERA startup: all models ready.")
    yield
    logger.info("VERA shutdown complete.")


# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="VERA – Voice Evidence & Risk Authentication API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["health"])
app.include_router(sessions.router, prefix=f"{settings.API_V1_STR}/sessions", tags=["sessions"])
app.include_router(websocket.router, tags=["websocket"])
app.include_router(voice_profiles.router, prefix=f"{settings.API_V1_STR}/voice-profiles", tags=["voice-profiles"])

@app.get("/")
def root():
    return {"message": "VERA Backend is running. Use /api/v1/health to check status."}
