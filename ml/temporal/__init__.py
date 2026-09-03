"""VERA Layer 1 Temporal / Rolling Real-Time Inference module."""

from ml.temporal.engine import RollingInferenceEngine, TemporalInferenceEvent
from ml.temporal.rolling_buffer import AudioWindow, RollingAudioBuffer, WINDOW_SIZE_SAMPLES
from ml.temporal.smoother import (
    ExponentialMovingAverageSmoother,
    MovingAverageSmoother,
    TemporalSmoother,
    create_smoother,
)
from ml.temporal.state_machine import DetectionState, HysteresisStateMachine

__all__ = [
    "AudioWindow",
    "DetectionState",
    "ExponentialMovingAverageSmoother",
    "HysteresisStateMachine",
    "MovingAverageSmoother",
    "RollingAudioBuffer",
    "RollingInferenceEngine",
    "TemporalInferenceEvent",
    "TemporalSmoother",
    "WINDOW_SIZE_SAMPLES",
    "create_smoother",
]
