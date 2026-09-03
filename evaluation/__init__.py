"""Layer 1 Evaluation Package for Spectra-AASIST3 Voice Authenticity."""
from evaluation.dataset import (
    ASVDataset,
    AudioSample,
    SpeakerLeakageError,
)
from evaluation.metrics import (
    EvaluationMetrics,
    calculate_metrics,
    compute_eer,
)
from evaluation.run_evaluation import (
    evaluate_dataset,
    save_reports,
)

__all__ = [
    "ASVDataset",
    "AudioSample",
    "SpeakerLeakageError",
    "EvaluationMetrics",
    "calculate_metrics",
    "compute_eer",
    "evaluate_dataset",
    "save_reports",
]
