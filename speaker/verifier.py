import os
import numpy as np
import config
from .enroll import SpeakerEnroller, sanitize_speaker_id

class SpeakerVerifier:
    """
    Compares live incoming audio against enrolled reference embeddings using Cosine Similarity.
    Outputs speaker consistency score and confidence without making definitive fraud claims.
    """
    def __init__(self, enroller: SpeakerEnroller = None):
        self.enroller = enroller or SpeakerEnroller()
        self.threshold = config.SPEAKER_MATCH_THRESHOLD

    def get_reference_embedding(self, speaker_id: str) -> np.ndarray:
        """Safely loads enrolled reference embedding with allow_pickle=False and path sanitization."""
        clean_id = sanitize_speaker_id(speaker_id)
        path = os.path.join(config.ENROLLED_SPEAKERS_DIR, f"{clean_id}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No enrolled profile found for speaker_id: '{clean_id}'. Please enroll first.")
        # Fix 3: Strict safe loading with allow_pickle=False
        return np.load(path, allow_pickle=False)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def verify(self, speaker_id: str, live_audio_path: str) -> dict:
        """
        Compares live audio chunk against stored reference embedding.
        """
        clean_id = sanitize_speaker_id(speaker_id)
        ref_emb = self.get_reference_embedding(clean_id)
        live_emb = self.enroller.extract_embedding(live_audio_path)
        
        sim = self.cosine_similarity(ref_emb, live_emb)
        match_score = max(0.0, min(1.0, sim))
        match_score = round(float(match_score), 3)

        # Confidence metric
        confidence = round(float(min(1.0, 0.5 + abs(match_score - self.threshold))), 2)

        return {
            "speaker_id": clean_id,
            "match_score": match_score,
            "confidence": confidence,
            "is_consistent": match_score >= self.threshold
        }
