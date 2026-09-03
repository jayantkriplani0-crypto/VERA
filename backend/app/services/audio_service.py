import tempfile
import os
import logging
import librosa
from fastapi import UploadFile

logger = logging.getLogger("vera.audio_service")


async def load_and_normalize_audio(file: UploadFile):
    """
    Decode an uploaded audio file and return (y, sr, duration, filename) where:
      - y   : float32 numpy array, mono, 16 000 Hz
      - sr  : always 16 000
      - duration : seconds (float)
      - filename : original upload filename

    Accepted formats: WAV, MP3, OGG, FLAC (via soundfile), WebM/Opus, M4A,
    AAC (via librosa/audioread fallback). The frontend is expected to send WAV
    produced by the Web Audio API, but the backend is robust to browser-native
    formats regardless.

    Raises ValueError with a human-readable message on decode failure.
    Raw audio is NEVER persisted — the temp file is always deleted.
    """
    content = await file.read()
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        # Guess extension from content_type when filename has no extension
        ct = (file.content_type or "").lower()
        if "webm" in ct:
            ext = ".webm"
        elif "ogg" in ct:
            ext = ".ogg"
        elif "mp3" in ct or "mpeg" in ct:
            ext = ".mp3"
        elif "mp4" in ct or "m4a" in ct:
            ext = ".m4a"
        else:
            ext = ".wav"

    logger.info(
        "Decoding audio: filename=%r ext=%r content_type=%r bytes=%d",
        file.filename, ext, file.content_type, len(content),
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(
            "Audio decoded ok: duration=%.2fs sr=%d channels=1 (mono) samples=%d",
            duration, sr, len(y),
        )
        return y, sr, duration, file.filename
    except Exception as e:
        logger.warning(
            "librosa.load failed for ext=%r content_type=%r: %s",
            ext, file.content_type, e,
        )
        raise ValueError(
            f"Could not decode audio file (ext={ext!r}, "
            f"content_type={file.content_type!r}). "
            f"Ensure you are sending a valid WAV, MP3, OGG, or FLAC file. "
            f"Internal error: {e}"
        ) from e
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Temp file deleted: %s", tmp_path)


async def process_uploaded_audio(session_id: str, file: UploadFile):
    y, sr, duration, filename = await load_and_normalize_audio(file)

    return {
        "session_id": session_id,
        "filename": filename,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "channels": 1,
        "status": "processed",
    }
