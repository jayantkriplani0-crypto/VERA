"""Spectra-AASIST3 CLI inference tool for Layer 1 Voice Authenticity.

Follows the official Hugging Face model card exactly:
  - Model: lab260/Spectra-AASIST3 (Wav2Vec2 XLS-R-300M + KAN-AASIST)
  - Preprocessing: 16 kHz mono, preemphasis (0.97), 64,600-sample window
  - Score: Raw bona fide logit (higher = more bona fide / genuine speech)
  - Threshold: -1.0625009 (official EER decision threshold)

Usage:
    python infer_spectra_aasist3.py path/to/audio.wav
    python infer_spectra_aasist3.py path/to/audio.wav --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.voice_detector.detector import VoiceAuthenticityDetector
from ml.voice_detector.model_loader import MODEL_HUB_REPO
from ml.voice_detector.scorer import OFFICIAL_EER_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectra-AASIST3 Layer 1 Voice Authenticity Inference CLI"
    )
    parser.add_argument(
        "wav",
        type=str,
        help="Path to the input WAV audio file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: 'auto' (default), 'cpu', or 'cuda'"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=OFFICIAL_EER_THRESHOLD,
        help=f"Decision threshold (default: {OFFICIAL_EER_THRESHOLD} from official model card)"
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"Error: audio file not found at '{wav_path.resolve()}'.", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Loading {MODEL_HUB_REPO} on device '{args.device}'...")
        detector = VoiceAuthenticityDetector(device=args.device)
        print("Running inference...")
        result = detector.predict_file(wav_path, threshold=args.threshold)
        print()
        result.print_report()
    except Exception as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
