"""Unit tests for Layer 1 Calibration module (calibration/ package)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from calibration.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    compute_ece,
    evaluate_calibration_candidates,
)
from calibration.apply_calibration import (
    CalibratedScoreResult,
    CalibratedVoiceScorer,
    apply_calibration,
)
from evaluation.dataset import ASVDataset, AudioSample, SpeakerLeakageError


class TestCalibrators:
    """Test individual calibration algorithms and score directionality."""

    def test_platt_calibrator_fitting(self):
        """Platt scaling must monotonically map higher raw logits to higher P(BonaFide)."""
        raw_scores = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        y_true = np.array([0, 0, 0, 1, 1, 1])

        calibrator = PlattCalibrator().fit(raw_scores, y_true)
        probs = calibrator.predict_proba(raw_scores)

        assert len(probs) == 6
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        # Verify strict monotonicity
        assert np.all(np.diff(probs) > 0)
        # High logit (+5.0) -> high bona fide prob (> 0.90), low logit (-5.0) -> low (< 0.10)
        assert probs[-1] > 0.90
        assert probs[0] < 0.10

    def test_score_directionality_spoof_signal(self):
        """CRITICAL: Higher VERA spoof signal must correspond to LOWER raw bona-fide score."""
        raw_scores = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        calibrator = PlattCalibrator(slope_w=1.0, intercept_b=0.0)

        spoof_signals = calibrator.to_spoof_signal(raw_scores)
        integrity_scores = calibrator.to_integrity_score(raw_scores)

        # Bounds check
        assert np.all(spoof_signals >= 0.0) and np.all(spoof_signals <= 100.0)
        assert np.all(integrity_scores >= 0.0) and np.all(integrity_scores <= 100.0)

        # Complementary property: spoof_signal + integrity_score == 100.0
        np.testing.assert_allclose(spoof_signals + integrity_scores, 100.0, atol=1e-2)

        # Directionality check: as raw score increases, spoof signal MUST decrease monotonically
        assert np.all(np.diff(spoof_signals) < 0)

        # Deep negative logit (-5.0) must produce very high spoof signal (> 95.0)
        assert spoof_signals[0] > 95.0
        # High positive logit (+5.0) must produce very low spoof signal (< 5.0)
        assert spoof_signals[-1] < 5.0

    def test_confidence_metric(self):
        """Confidence metric must be in [0.0, 1.0], with minimum at decision boundary (p=0.5)."""
        calibrator = PlattCalibrator(slope_w=1.0, intercept_b=0.0)
        conf_boundary = calibrator.compute_confidence(0.0)  # logit 0.0 -> p = 0.5
        conf_extreme = calibrator.compute_confidence(10.0)  # logit 10.0 -> p ~ 1.0

        assert abs(conf_boundary - 0.0) < 1e-3
        assert conf_extreme > 0.99

    def test_temperature_scaling(self):
        raw_scores = np.array([-3.0, -1.0, 1.0, 3.0])
        y_true = np.array([0, 0, 1, 1])
        temp = TemperatureCalibrator().fit(raw_scores, y_true)

        probs = temp.predict_proba(raw_scores)
        assert len(probs) == 4
        assert np.all(np.diff(probs) > 0)

    def test_ece_computation(self):
        """ECE must be near 0 for well-calibrated predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.05, 0.05, 0.95, 0.95])
        ece = compute_ece(y_true, y_prob, n_bins=2)
        assert ece < 0.10

    def test_evaluate_calibration_candidates(self):
        val_scores = np.array([-4.0, -3.0, 1.0, 2.0])
        val_labels = np.array([0, 0, 1, 1])
        results = evaluate_calibration_candidates(val_scores, val_labels)

        assert "PlattScaling" in results
        assert "TemperatureScaling" in results
        assert "IsotonicRegression" in results
        assert "brier_score" in results["PlattScaling"]


class TestSpeakerDisjointnessBeforeCalibration:
    """Ensure data rule: strict speaker isolation is validated before calibration."""

    def test_speaker_separation_verified(self):
        samples = [
            AudioSample("f1.wav", "bonafide", "SPK_A", "-"),
            AudioSample("f2.wav", "spoof", "SPK_A", "TTS"),
            AudioSample("f3.wav", "bonafide", "SPK_B", "-"),
            AudioSample("f4.wav", "spoof", "SPK_B", "VC"),
            AudioSample("f5.wav", "bonafide", "SPK_C", "-"),
            AudioSample("f6.wav", "spoof", "SPK_C", "TTS"),
        ]
        ds = ASVDataset(samples)
        splits = ds.split_by_speaker(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        ASVDataset.verify_speaker_disjointness(splits)

        assert splits["val"].speakers.isdisjoint(splits["test"].speakers)

    def test_leakage_detected(self):
        leak_splits = {
            "val": ASVDataset([AudioSample("f1.wav", "bonafide", "SPK_LEAK", "-")]),
            "test": ASVDataset([AudioSample("f2.wav", "spoof", "SPK_LEAK", "TTS")]),
        }
        with pytest.raises(SpeakerLeakageError):
            ASVDataset.verify_speaker_disjointness(leak_splits)


class TestCalibratedVoiceScorer:
    """Test score application and formatting."""

    def test_apply_calibration_from_config(self, tmp_path):
        config_data = {
            "model_identifier": "lab260/Spectra-AASIST3",
            "calibration_method": "PlattScaling",
            "parameters": {
                "slope_w": 1.5,
                "intercept_b": 0.0,
            },
            "operational_thresholds": {
                "calibrated_spoof_threshold": 50.0,
            }
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        scorer = CalibratedVoiceScorer(config_path=config_file)

        # Test positive logit (genuine speech)
        res_bonafide = scorer.calibrate_raw_score(2.382637)
        assert isinstance(res_bonafide, CalibratedScoreResult)
        assert res_bonafide.raw_bonafide_score == 2.382637
        assert res_bonafide.calibrated_spoof_signal < 10.0  # low spoof signal
        assert res_bonafide.voice_integrity_score > 90.0     # high integrity
        assert res_bonafide.is_bona_fide is True
        assert "BONA FIDE" in res_bonafide.decision
        assert res_bonafide.decision_confidence > 0.80

        # Test negative logit (spoofed speech)
        res_spoof = scorer.calibrate_raw_score(-4.500000)
        assert res_spoof.calibrated_spoof_signal > 95.0    # high spoof signal
        assert res_spoof.voice_integrity_score < 5.0       # low integrity
        assert res_spoof.is_bona_fide is False
        assert "SPOOF" in res_spoof.decision

    def test_extreme_edge_cases(self, tmp_path):
        """Test extreme values (-100, +100, 0.0)."""
        config_data = {
            "model_identifier": "lab260/Spectra-AASIST3",
            "calibration_method": "PlattScaling",
            "parameters": {"slope_w": 1.0, "intercept_b": 0.0},
            "operational_thresholds": {"calibrated_spoof_threshold": 50.0}
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        scorer = CalibratedVoiceScorer(config_path=config_file)

        res_inf_pos = scorer.calibrate_raw_score(100.0)
        assert res_inf_pos.calibrated_spoof_signal == 0.0
        assert res_inf_pos.voice_integrity_score == 100.0

        res_inf_neg = scorer.calibrate_raw_score(-100.0)
        assert res_inf_neg.calibrated_spoof_signal == 100.0
        assert res_inf_neg.voice_integrity_score == 0.0

        res_zero = scorer.calibrate_raw_score(0.0)
        assert abs(res_zero.calibrated_spoof_signal - 50.0) < 1e-2
        assert abs(res_zero.decision_confidence - 0.0) < 1e-2
