"""Application module to apply learned calibration parameters to raw Spectra-AASIST3 scores.

Outputs:
  - raw_bonafide_score: Preserved unnormalized model logit (higher = more bona fide)
  - calibrated_spoof_signal: VERA Normalized Spoof Signal on a 0–100 scale (HIGHER = MORE SUSPICIOUS SYNTHETIC/SPOOF)
  - voice_integrity_score: Complementary 0–100 authenticity score (100 - calibrated_spoof_signal)
  - decision_confidence: Calibrated certainty metric in [0.0, 1.0]
  - Authentic/Spoof decision
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.voice_detector.detector import VoiceAuthenticityDetector


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "calibration" / "config.json"


@dataclass(frozen=True)
class CalibratedScoreResult:
    """Structured result containing both raw and calibrated score representations."""
    raw_bonafide_score: float
    calibrated_spoof_signal: float        # 0–100 (HIGHER = MORE SUSPICIOUS SPOOF / DEEPFAKE)
    voice_integrity_score: float          # 0–100 (100 - calibrated_spoof_signal, higher = more authentic)
    calibrated_probability_spoof: float   # in [0.0, 1.0]
    calibrated_probability_bonafide: float# in [0.0, 1.0]
    decision_confidence: float            # in [0.0, 1.0] distance from decision boundary
    decision_threshold: float             # calibrated cutoff (default: 50.0)
    decision: str
    is_bona_fide: bool
    explanation: str

    @property
    def raw_score(self) -> float:
        """Alias for backwards compatibility."""
        return self.raw_bonafide_score

    @property
    def spoof_risk_index(self) -> float:
        """Alias for backwards compatibility."""
        return self.calibrated_spoof_signal

    @property
    def calibrated_probability(self) -> float:
        """Alias for backwards compatibility (P(Bona Fide))."""
        return self.calibrated_probability_bonafide

    def to_dict(self) -> dict:
        return asdict(self)

    def print_report(self) -> None:
        """Print clean summary."""
        banner = "=" * 65
        lines = [
            banner,
            "  VERA LAYER 1: CALIBRATED VOICE INTEGRITY & SPOOF REPORT",
            banner,
            f"Raw Bona-Fide Score (Logit)  : {self.raw_bonafide_score:+.6f}  [Higher = more genuine human voice]",
            f"Calibrated Spoof Signal      : {self.calibrated_spoof_signal:.1f} / 100   [HIGHER = MORE SUSPICIOUS SPOOF]",
            f"Voice Integrity Score        : {self.voice_integrity_score:.1f} / 100   [Higher = more genuine voice]",
            f"Calibrated P(Spoof | s)      : {self.calibrated_probability_spoof:.4f}",
            f"Calibrated P(BonaFide | s)   : {self.calibrated_probability_bonafide:.4f}",
            f"Decision Confidence          : {self.decision_confidence:.4f}  [0.0 = boundary uncertainty, 1.0 = certain]",
            f"Operational Spoof Cutoff     : {self.decision_threshold:.1f} / 100",
            f"Classification Decision      : {self.decision}",
            banner,
            "Assessment Summary:",
            f"  {self.explanation}",
            banner,
            "Note: Calibrated scores represent posterior probability transforms. Never refer to this score as accuracy.",
            banner,
        ]
        print("\n".join(lines))


class CalibratedVoiceScorer:
    """Loads calibration parameters from config.json and transforms raw logits."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Calibration config not found at '{path.resolve()}'. "
                f"Please run 'python calibration/calibrate.py' first."
            )

        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        params = self.config["parameters"]
        self.w = float(params["slope_w"])
        self.b = float(params["intercept_b"])
        self.method = self.config.get("calibration_method", "PlattScaling")
        self.threshold = float(
            self.config.get("operational_thresholds", {}).get("calibrated_spoof_threshold", 50.0)
        )

    def calibrate_raw_score(self, raw_score: float) -> CalibratedScoreResult:
        """Transform a single raw bona fide logit into a calibrated score representation.

        Formula:
          P(Bona Fide | s) = 1 / (1 + exp(-(w * s + b)))
          P(Spoof | s)     = 1 - P(Bona Fide | s)
          calibrated_spoof_signal = P(Spoof | s) * 100.0  (HIGHER = MORE SPOOF)
          voice_integrity_score   = P(Bona Fide | s) * 100.0
        """
        s = float(raw_score)
        logit_scaled = self.w * s + self.b
        if logit_scaled >= 0:
            p_bonafide = float(1.0 / (1.0 + np.exp(-logit_scaled)))
        else:
            p_bonafide = float(np.exp(logit_scaled) / (1.0 + np.exp(logit_scaled)))

        p_bonafide = float(np.clip(p_bonafide, 0.0, 1.0))
        p_spoof = float(1.0 - p_bonafide)

        # Calibrated 0-100 scales
        calibrated_spoof_signal = round(p_spoof * 100.0, 2)
        voice_integrity_score = round(p_bonafide * 100.0, 2)

        # Confidence: distance from decision threshold (p=0.5)
        decision_confidence = round(2.0 * abs(p_bonafide - 0.5), 4)

        # Classification based on spoof signal vs 50.0 threshold
        is_bona_fide = bool(calibrated_spoof_signal < self.threshold)

        if is_bona_fide:
            decision = "BONA FIDE (Genuine Human Speech)"
            if calibrated_spoof_signal <= 20.0:
                explanation = (
                    f"Low Spoof Signal ({calibrated_spoof_signal:.1f}/100, Integrity: {voice_integrity_score:.1f}/100). "
                    f"Acoustic and spectral representations exhibit natural human vocal tract characteristics. High confidence ({decision_confidence:.2f})."
                )
            else:
                explanation = (
                    f"Moderate Spoof Signal ({calibrated_spoof_signal:.1f}/100, Integrity: {voice_integrity_score:.1f}/100). "
                    f"Remains below the spoof threshold ({self.threshold:.1f}/100) and consistent with genuine speech. Moderate confidence ({decision_confidence:.2f})."
                )
        else:
            decision = "SPOOF / DEEPFAKE (Synthetic / Cloned Speech)"
            if calibrated_spoof_signal >= 80.0:
                explanation = (
                    f"Critical Spoof Signal ({calibrated_spoof_signal:.1f}/100, Integrity: {voice_integrity_score:.1f}/100). "
                    f"Strong synthetic generation / neural vocoder phase artifacts detected. High confidence ({decision_confidence:.2f})."
                )
            else:
                explanation = (
                    f"Elevated Spoof Signal ({calibrated_spoof_signal:.1f}/100). Exceeds the operational "
                    f"spoof threshold ({self.threshold:.1f}/100) indicating likely synthetic manipulation."
                )

        return CalibratedScoreResult(
            raw_bonafide_score=s,
            calibrated_spoof_signal=calibrated_spoof_signal,
            voice_integrity_score=voice_integrity_score,
            calibrated_probability_spoof=round(p_spoof, 4),
            calibrated_probability_bonafide=round(p_bonafide, 4),
            decision_confidence=decision_confidence,
            decision_threshold=self.threshold,
            decision=decision,
            is_bona_fide=is_bona_fide,
            explanation=explanation,
        )

    def calibrate_audio_file(
        self,
        audio_path: str | Path,
        detector: Optional[VoiceAuthenticityDetector] = None,
        device: str = "auto"
    ) -> CalibratedScoreResult:
        """Run Spectra-AASIST3 on a WAV file and return the calibrated result."""
        if detector is None:
            detector = VoiceAuthenticityDetector(device=device)

        res = detector.predict_file(audio_path)
        return self.calibrate_raw_score(res.raw_score)


def apply_calibration(raw_score: float, config_path: str | Path = DEFAULT_CONFIG_PATH) -> CalibratedScoreResult:
    """Convenience functional interface to calibrate a raw score."""
    scorer = CalibratedVoiceScorer(config_path=config_path)
    return scorer.calibrate_raw_score(raw_score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Layer 1 Voice Authenticity Calibration"
    )
    parser.add_argument(
        "--score",
        type=float,
        default=None,
        help="Raw bona fide logit score to calibrate"
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Path to WAV audio file to score and calibrate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to calibration config JSON (default: calibration/config.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model execution ('auto', 'cpu', 'cuda')"
    )
    args = parser.parse_args()

    if args.score is None and args.wav is None:
        print("Error: Specify either --score <float> or --wav <path/to/audio.wav>.", file=sys.stderr)
        sys.exit(1)

    scorer = CalibratedVoiceScorer(config_path=args.config)

    if args.score is not None:
        res = scorer.calibrate_raw_score(args.score)
    else:
        res = scorer.calibrate_audio_file(args.wav, device=args.device)

    res.print_report()


if __name__ == "__main__":
    main()
