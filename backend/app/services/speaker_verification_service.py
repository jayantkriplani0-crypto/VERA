import numpy as np
import tempfile
import os
import soundfile as sf

_enroller = None
_verifier = None

def get_model():
    global _enroller, _verifier
    if _enroller is None:
        from speaker.enroll import SpeakerEnroller
        from speaker.verifier import SpeakerVerifier
        _enroller = SpeakerEnroller()
        _verifier = SpeakerVerifier(_enroller)
    return _enroller, _verifier

def extract_embedding(audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    if len(audio_array) == 0:
        return np.zeros(192, dtype=np.float32)
        
    enroller, _ = get_model()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        
    try:
        sf.write(tmp_path, audio_array, sample_rate)
        emb = enroller.extract_embedding(tmp_path)
        return emb
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    _, verifier = get_model()
    
    if hasattr(verifier, 'cosine_similarity'):
        sim = verifier.cosine_similarity(emb1.flatten(), emb2.flatten())
    else:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        sim = float(np.dot(emb1.flatten(), emb2.flatten()) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0

    score = max(0.0, min(1.0, sim))
    threshold = 0.5 # Speechbrain ECAPA threshold typically around 0.5-0.6
    
    # Check if verifier has a threshold
    if hasattr(verifier, 'threshold'):
        threshold = verifier.threshold

    is_match = score >= threshold
    confidence = float(min(1.0, 0.5 + abs(score - threshold)))
    
    return {
        "speaker_similarity_score": score,
        "match": is_match,
        "model_name": "speechbrain/spkrec-ecapa-voxceleb",
        "confidence": confidence
    }

def verify_speaker(audio_array1: np.ndarray, audio_array2: np.ndarray, sample_rate: int = 16000) -> dict:
    if len(audio_array1) == 0 or len(audio_array2) == 0:
        return {
            "speaker_similarity_score": 0.0,
            "match": False,
            "model_name": "speechbrain/spkrec-ecapa-voxceleb",
            "confidence": 0.0
        }
        
    emb1 = extract_embedding(audio_array1, sample_rate)
    emb2 = extract_embedding(audio_array2, sample_rate)
    
    return compare_embeddings(emb1, emb2)

