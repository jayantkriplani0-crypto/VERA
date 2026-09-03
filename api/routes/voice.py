"""Voice authenticity analysis and streaming routes."""
from __future__ import annotations

import json
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import ValidationError

from api.dependencies import get_voice_service
from api.schemas import (
    ErrorResponse,
    StreamControlMessage,
    StreamingVoiceEvent,
    VoiceAnalysisResponse,
)
from api.services.voice_service import VoiceService

router = APIRouter(prefix="/api/v1/voice", tags=["Voice Authenticity"])

# Allowed audio MIME types and extensions
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}


@router.post(
    "/analyze",
    response_model=VoiceAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio data or empty file"},
        413: {"model": ErrorResponse, "description": "File exceeds maximum upload limit (25 MB)"},
        415: {"model": ErrorResponse, "description": "Unsupported audio format"},
        503: {"model": ErrorResponse, "description": "Voice Authenticity service unavailable"},
    },
    summary="Analyze Voice Authenticity (Single Audio File)",
    description="Upload a WAV or FLAC audio file to evaluate authenticity against frozen Spectra-AASIST3 and Platt calibration.",
)
async def analyze_voice(
    file: UploadFile = File(..., description="Audio file in WAV or FLAC format (mono 16 kHz recommended)"),
    service: VoiceService = Depends(get_voice_service),
) -> VoiceAnalysisResponse:
    """Evaluate voice authenticity of an uploaded audio file."""
    # Check extension if filename provided
    if file.filename:
        suffix = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format '{suffix}'. Supported formats: {sorted(ALLOWED_EXTENSIONS)}.",
            )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {e}",
        ) from e

    try:
        result = await service.analyze_audio_bytes(content, filename=file.filename)
    except ValueError as e:
        msg = str(e)
        if "exceeds limit" in msg:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {e}",
        ) from e

    return result


@router.websocket("/stream")
async def stream_voice_authenticity(
    websocket: WebSocket,
    service: VoiceService = Depends(get_voice_service),
) -> None:
    """Continuous real-time audio streaming endpoint via WebSocket.

    Protocol:
      - Client sends binary frames: raw PCM / WAV / FLAC bytes (16-kHz mono recommended).
      - Client sends text frames: JSON control messages (e.g. `{"action": "flush"}`).
      - Server emits JSON `StreamingVoiceEvent` frames whenever a complete 64,600-sample window is evaluated.
    """
    await websocket.accept()
    session = service.create_streaming_session()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                try:
                    events = await session.push_chunk(chunk_bytes)
                    for ev in events:
                        await websocket.send_text(ev.model_dump_json())
                except TimeoutError as te:
                    await websocket.send_json({"error": "stream_timeout", "detail": str(te)})
                    await websocket.close(code=1008)
                    break
                except Exception as e:
                    await websocket.send_json({"error": "processing_error", "detail": str(e)})

            elif "text" in message and message["text"]:
                text_data = message["text"].strip()
                try:
                    data = json.loads(text_data)
                    ctrl = StreamControlMessage(**data)
                except Exception:
                    await websocket.send_json({
                        "error": "malformed_control_message",
                        "detail": "Expected valid JSON with 'action' field ('flush', 'reset', 'close').",
                    })
                    continue

                if ctrl.action == "flush":
                    flush_events = await session.flush()
                    for ev in flush_events:
                        await websocket.send_text(ev.model_dump_json())
                elif ctrl.action == "reset":
                    session.reset()
                    await websocket.send_json({"status": "reset_complete"})
                elif ctrl.action == "close":
                    flush_events = await session.flush()
                    for ev in flush_events:
                        await websocket.send_text(ev.model_dump_json())
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": "stream_error", "detail": str(e)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        session.release()
