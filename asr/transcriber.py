import os
import config

class ASRTranscriber:
    """
    Automatic Speech Recognition (ASR) worker powered by pretrained faster-whisper.
    Converts incoming live audio chunks into text locally on the machine.
    """
    def __init__(self, model_size=config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
            print(f"[ASR] Loading pretrained faster-whisper ('{self.model_size}') on {self.device} ({self.compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("[ASR] faster-whisper neural model loaded successfully.")
        except Exception as e:
            print(f"[ASR Error] Failed to load faster-whisper: {e}")
            raise e

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes audio file to text using faster-whisper neural network.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(audio_path, beam_size=5, vad_filter=True)
        full_text = " ".join([seg.text.strip() for seg in segments]).strip()

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration_sec": round(info.duration, 2)
        }
