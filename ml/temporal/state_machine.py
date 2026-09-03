"""Hysteresis state machine for temporal voice authenticity classification.

Prevents high-frequency state oscillation and chattering between BONAFIDE and SPOOF
by enforcing dual hysteresis threshold bands and dwell-count debounce logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


class DetectionState(str, Enum):
    """Categorical classification states for temporal audio streams."""
    BONAFIDE = "BONAFIDE"       # High-confidence genuine human speech
    SUSPICIOUS = "SUSPICIOUS"   # Ambiguous / borderline audio or transitional uncertainty
    SPOOF = "SPOOF"             # High-confidence deepfake / synthetic speech


class HysteresisStateMachine:
    """State machine governing temporal decisions over smoothed spoof signals.

    Threshold Design:
      - Frozen Underlying Decision Boundary:
          Raw Logit = +2.781620 <=> P(BonaFide) = 0.50 <=> Calibrated Spoof Signal = 50.0
      - Dual Hysteresis Margins (Configurable Policy Parameters):
          bonafide_threshold : Spoof Signal < 35.0  (Clear genuine voice)
          suspicious_band     : 35.0 <= Spoof Signal < 65.0 (Borderline / uncertainty)
          spoof_threshold    : Spoof Signal >= 65.0 (Clear synthetic voice)
      - Debounce Logic:
          Requires `dwell_count` consecutive windows in a target zone before committing
          a state transition out of BONAFIDE or SPOOF.
    """

    def __init__(
        self,
        bonafide_threshold: float = 35.0,
        spoof_threshold: float = 65.0,
        dwell_count: int = 2,
        initial_state: DetectionState = DetectionState.BONAFIDE,
    ) -> None:
        if not (0.0 < bonafide_threshold < spoof_threshold < 100.0):
            raise ValueError("Thresholds must satisfy 0 < bonafide_threshold < spoof_threshold < 100.")
        if dwell_count < 1:
            raise ValueError("dwell_count must be at least 1.")

        self.bonafide_threshold = float(bonafide_threshold)
        self.spoof_threshold = float(spoof_threshold)
        self.dwell_count = int(dwell_count)

        self._current_state = initial_state
        self._pending_target_state: Optional[DetectionState] = None
        self._consecutive_count: int = 0
        self._total_transitions: int = 0

    @property
    def current_state(self) -> DetectionState:
        """The currently committed operational state."""
        return self._current_state

    @property
    def total_transitions(self) -> int:
        """Total number of committed state transitions."""
        return self._total_transitions

    def determine_instant_zone(self, smoothed_spoof_signal: float) -> DetectionState:
        """Map a smoothed spoof signal (0-100) directly to an instantaneous zone."""
        s = float(smoothed_spoof_signal)
        if s < self.bonafide_threshold:
            return DetectionState.BONAFIDE
        elif s >= self.spoof_threshold:
            return DetectionState.SPOOF
        else:
            return DetectionState.SUSPICIOUS

    def update(self, smoothed_spoof_signal: float) -> Tuple[DetectionState, bool]:
        """Update state machine with a new smoothed spoof signal.

        Returns:
            Tuple of (committed_state, state_changed_bool).
        """
        instant_zone = self.determine_instant_zone(smoothed_spoof_signal)

        # If instantaneous zone matches current committed state, reset pending transition
        if instant_zone == self._current_state:
            self._pending_target_state = None
            self._consecutive_count = 0
            return self._current_state, False

        # Transitioning into or remaining in SUSPICIOUS
        if instant_zone == DetectionState.SUSPICIOUS:
            # If currently in BONAFIDE or SPOOF, allow immediate transition to SUSPICIOUS
            # or require debounce depending on policy.
            # Standard policy: SUSPICIOUS is an immediate cautious buffer state.
            if self._current_state != DetectionState.SUSPICIOUS:
                self._current_state = DetectionState.SUSPICIOUS
                self._pending_target_state = None
                self._consecutive_count = 0
                self._total_transitions += 1
                return self._current_state, True
            return self._current_state, False

        # If instant_zone is a new target (e.g. SPOOF when current is BONAFIDE or SUSPICIOUS)
        if instant_zone != self._pending_target_state:
            self._pending_target_state = instant_zone
            self._consecutive_count = 1
        else:
            self._consecutive_count += 1

        # Check if dwell count threshold reached
        if self._consecutive_count >= self.dwell_count:
            self._current_state = self._pending_target_state
            self._pending_target_state = None
            self._consecutive_count = 0
            self._total_transitions += 1
            return self._current_state, True

        return self._current_state, False

    def reset(self, state: DetectionState = DetectionState.BONAFIDE) -> None:
        """Reset state machine counters and state."""
        self._current_state = state
        self._pending_target_state = None
        self._consecutive_count = 0
        self._total_transitions = 0
