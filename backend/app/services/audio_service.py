import tempfile
import os
import librosa
from fastapi import UploadFile

async def load_and_normalize_audio(file: UploadFile):
    content = await file.read()
    
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".wav"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        return y, sr, duration, file.filename
    except Exception as e:
        raise ValueError(f"Malformed audio file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

async def process_uploaded_audio(session_id: str, file: UploadFile):
    y, sr, duration, filename = await load_and_normalize_audio(file)
    
    return {
        "session_id": session_id,
        "filename": filename,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "channels": 1,
        "status": "processed"
    }
