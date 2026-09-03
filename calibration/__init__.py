"""Calibration package for Layer 1 Voice Authenticity."""
from calibration.apply_calibration import (
    CalibratedScoreResult,
    CalibratedVoiceScorer,
    apply_calibration,
)
from calibration.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    compute_ece,
    run_calibration_pipeline,
)

__all__ = [
    "CalibratedScoreResult",
    "CalibratedVoiceScorer",
    "apply_calibration",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "IsotonicCalibrator",
    "compute_ece",
    "run_calibration_pipeline",
]
