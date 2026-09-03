"""Voice Detector package for Layer 1 Voice Authenticity."""
from ml.voice_detector.detector import (
    DetectionResult,
    VoiceAuthenticityDetector,
)
from ml.voice_detector.model_loader import (
    MODEL_HUB_REPO,
    load_spectra_aasist3,
    resolve_device,
)
from ml.voice_detector.scorer import (
    BONAFIDE_CLASS_INDEX,
    OFFICIAL_EER_THRESHOLD,
    ScoreInterpretation,
    interpret_score,
)

__all__ = [
    "DetectionResult",
    "VoiceAuthenticityDetector",
    "MODEL_HUB_REPO",
    "load_spectra_aasist3",
    "resolve_device",
    "BONAFIDE_CLASS_INDEX",
    "OFFICIAL_EER_THRESHOLD",
    "ScoreInterpretation",
    "interpret_score",
]
