"""Layer 1 Voice Authenticity Detector interface using Spectra-AASIST3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ml.preprocessing.audio_loader import (
    AudioMetadata,
    TARGET_SAMPLE_RATE,
    load_audio,
    resample_audio,
)
from ml.preprocessing.preprocessor import preprocess_waveform
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


@dataclass(frozen=True)
class DetectionResult:
    """Complete result from Voice Authenticity detection on an audio file/waveform."""
    file_path: Optional[str]
    sample_rate: int
    duration_seconds: float
    model_name: str
    device: str
    raw_score: float
    interpretation: ScoreInterpretation
    metadata: Optional[AudioMetadata] = None

    def print_report(self) -> None:
        """Print a structured, clear verification report matching all audit criteria."""
        print(self.format_report())

    def format_report(self) -> str:
        """Format the report into a clean string."""
        banner = "=" * 60
        sub_banner = "-" * 60
        filename_display = Path(self.file_path).name if self.file_path else "<in-memory waveform>"
        lines = [
            banner,
            "  VERA LAYER 1: VOICE AUTHENTICITY DETECTION REPORT",
            banner,
            f"File Name           : {filename_display}",
            f"Full Path           : {self.file_path or 'N/A'}",
            f"Original Sample Rate: {self.sample_rate} Hz",
            f"Duration            : {self.duration_seconds:.2f} seconds",
            f"Model Identifier    : {self.model_name}",
            f"Compute Device      : {self.device}",
            sub_banner,
            f"Raw Model Score     : {self.raw_score:+.6f}  [Bona Fide Logit]",
            f"Score Scale Note    : Unbounded real-valued logit (higher = more bona fide)",
            f"Official Threshold  : {self.interpretation.threshold:+.6f} (EER cutoff from model card)",
            f"Classification      : {self.interpretation.predicted_label}",
            sub_banner,
            "Score Interpretation:",
            f"  {self.interpretation.description}",
            banner,
        ]
        return "\n".join(lines)


class VoiceAuthenticityDetector:
    """Detector class encapsulating audio loading, preprocessing, model execution, and scoring."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "auto",
        model_repo: str = MODEL_HUB_REPO
    ) -> None:
        """Initialize detector.

        Args:
            model: Optional pre-loaded SpectraAASIST3 instance.
            device: Target device ('auto', 'cpu', 'cuda').
            model_repo: Hugging Face model repository.
        """
        self.model_repo = model_repo
        if model is not None:
            self.model = model
            self.device = str(next(model.parameters()).device)
        else:
            self.model, self.device = load_spectra_aasist3(repo_or_path=model_repo, device=device)

    @torch.inference_mode()
    def _run_forward(self, preprocessed_waveform: np.ndarray) -> float:
        """Execute model forward pass on preprocessed 1D waveform."""
        tensor = torch.from_numpy(preprocessed_waveform).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        raw_score = float(logits[0, BONAFIDE_CLASS_INDEX].cpu())
        return raw_score

    @torch.inference_mode()
    def _run_forward_batch(self, preprocessed_waveforms: list[np.ndarray] | np.ndarray) -> list[float]:
        """Execute model forward pass on a batch of preprocessed 1D waveforms (shape: [B, 64600])."""
        if isinstance(preprocessed_waveforms, list):
            if len(preprocessed_waveforms) == 0:
                return []
            stacked = np.stack(preprocessed_waveforms)
        else:
            stacked = preprocessed_waveforms

        tensor = torch.from_numpy(stacked).to(self.device)
        logits = self.model(tensor)
        scores = logits[:, BONAFIDE_CLASS_INDEX].cpu().tolist()
        return [float(s) for s in scores]

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
        threshold: float = OFFICIAL_EER_THRESHOLD,
        file_path: Optional[str] = None
    ) -> DetectionResult:
        """Run voice authenticity detection on a raw in-memory waveform."""
        if waveform.size == 0:
            raise ValueError("Input waveform is empty.")

        duration_sec = len(waveform) / float(sample_rate)

        # Ensure mono
        if waveform.ndim > 1:
            mono_audio = np.mean(waveform, axis=1, dtype=np.float32)
        else:
            mono_audio = waveform.astype(np.float32)

        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            mono_audio = resample_audio(mono_audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

        # Preprocess (0.97 preemphasis + 64,600 windowing)
        preprocessed = preprocess_waveform(mono_audio)

        # Run forward pass
        raw_score = self._run_forward(preprocessed)

        # Interpret score
        interp = interpret_score(raw_score, threshold=threshold)

        return DetectionResult(
            file_path=file_path,
            sample_rate=sample_rate,
            duration_seconds=duration_sec,
            model_name=self.model_repo,
            device=self.device,
            raw_score=raw_score,
            interpretation=interp,
            metadata=None,
        )

    def predict_file(
        self,
        file_path: str | Path,
        threshold: float = OFFICIAL_EER_THRESHOLD
    ) -> DetectionResult:
        """Run voice authenticity detection on a local WAV file."""
        waveform, meta = load_audio(file_path, target_sr=TARGET_SAMPLE_RATE, allow_resample=True)

        # Preprocess
        preprocessed = preprocess_waveform(waveform)

        # Forward pass
        raw_score = self._run_forward(preprocessed)

        # Interpret score
        interp = interpret_score(raw_score, threshold=threshold)

        return DetectionResult(
            file_path=meta.file_path,
            sample_rate=meta.original_sample_rate,
            duration_seconds=meta.duration_seconds,
            model_name=self.model_repo,
            device=self.device,
            raw_score=raw_score,
            interpretation=interp,
            metadata=meta,
        )
