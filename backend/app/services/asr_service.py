import numpy as np
import tempfile
import os
import soundfile as sf

_transcriber = None

def get_model():
    global _transcriber
    if _transcriber is None:
        from asr.transcriber import ASRTranscriber
        _transcriber = ASRTranscriber()
    return _transcriber, None, None

def transcribe_audio(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    if len(audio_array) == 0:
        return {
            "transcript": "",
            "language": "unknown",
            "model_name": "faster-whisper",
            "confidence": 0.0
        }
        
    transcriber, _, _ = get_model()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        
    try:
        sf.write(tmp_path, audio_array, sample_rate)
        res = transcriber.transcribe(tmp_path)
        return {
            "transcript": res.get("text", ""),
            "language": res.get("language", "auto"),
            "model_name": "faster-whisper",
            "confidence": res.get("language_probability", None)
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
