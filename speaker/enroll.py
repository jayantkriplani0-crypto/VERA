import os
import re
import torch
import soundfile as sf
import numpy as np
import config

def sanitize_speaker_id(speaker_id: str) -> str:
    """
    Sanitizes speaker_id to prevent path traversal vulnerabilities.
    Strips out slashes, backslashes, dots, and non-alphanumeric characters.
    """
    if not speaker_id or not isinstance(speaker_id, str):
        raise ValueError("Invalid speaker_id provided: must be a non-empty string.")
    # Remove any slashes, dots, and path traversal sequences
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", speaker_id).strip()
    if not cleaned:
        raise ValueError(f"speaker_id '{speaker_id}' contains no valid alphanumeric characters.")
    return cleaned

class SpeakerEnroller:
    """
    Enrolls reference speaker voices by extracting 192-dimensional ECAPA-TDNN embeddings
    using the pretrained SpeechBrain 'spkrec-ecapa-voxceleb' neural network.
    """
    def __init__(self, model_source=config.SPEAKER_MODEL_SOURCE):
        self.model_source = model_source
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            from speechbrain.utils.fetching import LocalStrategy
            print(f"[ECAPA] Loading pretrained SpeechBrain ECAPA-TDNN model ('{self.model_source}')...")
            self.model = SpeakerRecognition.from_hparams(
                source=self.model_source,
                savedir=os.path.join(config.DATA_DIR, "pretrained_models", "ecapa"),
                local_strategy=LocalStrategy.COPY
            )
            print("[ECAPA] SpeechBrain ECAPA-TDNN loaded successfully.")
        except Exception as e:
            print(f"[ECAPA Error] Failed to load SpeechBrain model: {e}")
            raise e

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extracts 192-dim speaker embedding from audio file using ECAPA-TDNN."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Read audio using soundfile
        data, sample_rate = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        
        # Ensure (1, time) format
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.transpose(0, 1)

        # Ensure mono channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract 192-dim embedding from ECAPA-TDNN
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
            emb_np = embedding.squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            return emb_np

    def enroll(self, speaker_id: str, audio_path: str) -> str:
        """Extracts and stores reference embedding for sanitized speaker_id."""
        os.makedirs(config.ENROLLED_SPEAKERS_DIR, exist_ok=True)
        clean_id = sanitize_speaker_id(speaker_id)
        embedding = self.extract_embedding(audio_path)
        out_path = os.path.join(config.ENROLLED_SPEAKERS_DIR, f"{clean_id}.npy")
        np.save(out_path, embedding)
        print(f"[ECAPA] Enrolled reference speaker '{clean_id}' -> 192-dim embedding saved ({out_path})")
        return out_path
