import os
from pathlib import Path

# Allow multiple OpenMP runtimes (standard for PyTorch + CTranslate2 on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ENROLLED_SPEAKERS_DIR = DATA_DIR / "enrolled_speakers"
TEST_DATA_DIR = BASE_DIR / "test_data"

# Audio parameters
SAMPLE_RATE = 16000  # Standard 16kHz required by ECAPA and Whisper
MAX_AUDIO_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB max limit to prevent DoS attacks

# Speaker Verification (ECAPA-TDNN)
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MATCH_THRESHOLD = 0.65  # Cosine similarity threshold for consistency

# ASR (faster-whisper)
WHISPER_MODEL_SIZE = "tiny"   # Options: 'tiny', 'base', 'small', 'medium'
WHISPER_DEVICE = "cpu"        # 'cuda' if GPU available, 'cpu' otherwise
WHISPER_COMPUTE_TYPE = "int8" # 'int8', 'float16', 'float32'

# LLM (Gemini API)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
