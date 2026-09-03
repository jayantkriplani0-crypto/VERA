"""Dedicated deterministic QA test suite for VERA Layer 1 Temporal State Machine.

Validates:
  1. Genuine sequence: multiple BONAFIDE windows -> state remains BONAFIDE.
  2. Spoof sequence: first high-spoof window -> state remains pending; second consecutive window -> commits to SPOOF.
  3. Borderline sequence: exact boundary testing at 34.99, 35.0, 49.99, 50.0, 64.99, 65.0.
  4. Glitch rejection: isolated spoof spike -> debounces without entering SPOOF state.
  5. Recovery: state SPOOF -> consecutive clean bona-fide windows -> cleanly recovers to BONAFIDE.
  6. Session isolation: new session / reset produces zero history bleed-over.
"""
import pytest
from ml.temporal.state_machine import DetectionState, HysteresisStateMachine


class TestTemporalStateMachineQA:
    """Comprehensive QA suite covering all transition rules of HysteresisStateMachine."""

    def test_1_genuine_sequence(self):
        """Sequence of clean bona fide windows keeps state in BONAFIDE."""
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)
        assert sm.current_state == DetectionState.BONAFIDE

        signals = [0.0, 5.0, 12.0, 20.0, 34.0, 10.0, 2.0]
        for sig in signals:
            state, changed = sm.update(sig)
            assert state == DetectionState.BONAFIDE
            assert changed is False
            assert sm.current_state == DetectionState.BONAFIDE

    def test_2_spoof_sequence_pending_then_committed(self):
        """First spoof window sets pending state; second consecutive spoof window commits state."""
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)
        assert sm.current_state == DetectionState.BONAFIDE

        # Window 1: High spoof signal (95.0)
        state1, changed1 = sm.update(95.0)
        # Must still be BONAFIDE because dwell_count=2 requires 2 consecutive windows
        assert state1 == DetectionState.BONAFIDE
        assert changed1 is False
        assert sm.current_state == DetectionState.BONAFIDE
        assert sm._pending_target_state == DetectionState.SPOOF
        assert sm._consecutive_count == 1

        # Window 2: Second consecutive high spoof signal (92.0)
        state2, changed2 = sm.update(92.0)
        # Now state must transition to SPOOF
        assert state2 == DetectionState.SPOOF
        assert changed2 is True
        assert sm.current_state == DetectionState.SPOOF
        assert sm._pending_target_state is None
        assert sm._consecutive_count == 0

    def test_3_borderline_inclusive_exclusive_semantics(self):
        """Verify exact boundary behaviors:
        - BONAFIDE: signal < 35.0
        - SUSPICIOUS: 35.0 <= signal < 65.0
        - SPOOF: signal >= 65.0
        """
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)

        # 34.99 -> BONAFIDE (< 35.0)
        assert sm.determine_instant_zone(34.99) == DetectionState.BONAFIDE
        assert sm.determine_instant_zone(0.0) == DetectionState.BONAFIDE

        # Exactly 35.0 -> SUSPICIOUS (inclusive lower bound)
        assert sm.determine_instant_zone(35.0) == DetectionState.SUSPICIOUS
        # 49.99 and 50.0 -> SUSPICIOUS
        assert sm.determine_instant_zone(49.99) == DetectionState.SUSPICIOUS
        assert sm.determine_instant_zone(50.0) == DetectionState.SUSPICIOUS
        # 64.99 -> SUSPICIOUS (< 65.0)
        assert sm.determine_instant_zone(64.99) == DetectionState.SUSPICIOUS

        # Exactly 65.0 -> SPOOF (inclusive upper threshold)
        assert sm.determine_instant_zone(65.0) == DetectionState.SPOOF
        # 65.01 -> SPOOF
        assert sm.determine_instant_zone(65.01) == DetectionState.SPOOF
        assert sm.determine_instant_zone(100.0) == DetectionState.SPOOF

    def test_4_glitch_rejection(self):
        """An isolated spoof spike must not cause a false alarm."""
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)

        # Clean baseline
        state1, changed1 = sm.update(10.0)
        assert state1 == DetectionState.BONAFIDE

        # Isolated spike to 98.0
        state2, changed2 = sm.update(98.0)
        assert state2 == DetectionState.BONAFIDE
        assert changed2 is False
        assert sm._pending_target_state == DetectionState.SPOOF
        assert sm._consecutive_count == 1

        # Immediate return to genuine speech (12.0)
        state3, changed3 = sm.update(12.0)
        assert state3 == DetectionState.BONAFIDE
        assert changed3 is False
        assert sm._pending_target_state is None
        assert sm._consecutive_count == 0

    def test_5_recovery_from_spoof_to_bonafide(self):
        """When in SPOOF state, consecutive bona-fide windows must trigger recovery."""
        sm = HysteresisStateMachine(bonafide_threshold=35.0, spoof_threshold=65.0, dwell_count=2)

        # Push into SPOOF state
        sm.update(90.0)
        state_spoof, changed_spoof = sm.update(90.0)
        assert state_spoof == DetectionState.SPOOF
        assert changed_spoof is True

        # Recovery Window 1: Clean audio (15.0)
        state_rec1, changed_rec1 = sm.update(15.0)
        # Should still be in SPOOF (debouncing recovery, dwell_count=2)
        assert state_rec1 == DetectionState.SPOOF
        assert changed_rec1 is False
        assert sm._pending_target_state == DetectionState.BONAFIDE
        assert sm._consecutive_count == 1

        # Recovery Window 2: Second consecutive clean audio (10.0)
        state_rec2, changed_rec2 = sm.update(10.0)
        # Now successfully recovered to BONAFIDE
        assert state_rec2 == DetectionState.BONAFIDE
        assert changed_rec2 is True
        assert sm.current_state == DetectionState.BONAFIDE

    def test_6_new_session_zero_leakage(self):
        """Resetting a state machine completely restores initial state and wipes history."""
        sm = HysteresisStateMachine(dwell_count=2)

        # Drive to SPOOF
        sm.update(99.0)
        sm.update(99.0)
        assert sm.current_state == DetectionState.SPOOF

        # Reset session
        sm.reset()
        assert sm.current_state == DetectionState.BONAFIDE
        assert sm._pending_target_state is None
        assert sm._consecutive_count == 0

        # Feed a single clean sample
        state, changed = sm.update(20.0)
        assert state == DetectionState.BONAFIDE
        assert changed is False
