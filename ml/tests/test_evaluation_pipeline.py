"""Unit tests for the Layer 1 Evaluation Pipeline (evaluation/ package)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import numpy as np

from evaluation.metrics import (
    EvaluationMetrics,
    calculate_metrics,
    compute_eer,
)
from evaluation.dataset import (
    ASVDataset,
    AudioSample,
    SpeakerLeakageError,
)
from evaluation.run_evaluation import (
    evaluate_dataset,
    save_reports,
)
from ml.tests.test_voice_detector import DummySpectraNet
from ml.voice_detector.detector import VoiceAuthenticityDetector


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------
class TestEvaluationMetrics:
    """Test standard anti-spoofing metrics calculations."""

    def test_eer_perfect_separation(self):
        """When bona fide scores are completely above spoof scores, EER should be 0.0%."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([-5.0, -3.0, -2.0, 2.0, 3.5, 5.0])
        eer, thresh = compute_eer(y_true, y_scores)

        assert eer == 0.0
        assert -2.0 <= thresh <= 2.0

    def test_eer_overlapping_distributions(self):
        """Test EER calculation on realistic overlapping score distributions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([-3.0, -1.0, 0.5, -2.0, -0.5, 1.0, 2.0, 3.0])
        eer, thresh = compute_eer(y_true, y_scores)

        assert 0.0 < eer < 50.0
        assert isinstance(thresh, float)

    def test_calculate_metrics_comprehensive(self):
        """Verify all fields in EvaluationMetrics are correctly computed."""
        y_true = [0, 0, 1, 1]
        y_scores = [-2.0, 0.5, 1.5, 2.5]  # threshold = 0.0 -> TN=1, FP=1, FN=0, TP=2
        metrics = calculate_metrics(
            y_true=y_true,
            y_scores=y_scores,
            speaker_ids=["S1", "S2", "S3", "S4"],
            threshold=0.0,
            total_time_sec=1.0,
        )

        assert metrics.total_samples == 4
        assert metrics.num_bonafide == 2
        assert metrics.num_spoof == 2
        assert metrics.true_positives == 2
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 0
        assert metrics.precision == 2.0 / 3.0
        assert metrics.recall == 1.0
        assert metrics.accuracy == 0.75
        assert metrics.false_positive_rate == 0.5
        assert metrics.false_negative_rate == 0.0
        assert metrics.avg_latency_ms == 250.0
        assert metrics.throughput_samples_per_sec == 4.0

    def test_metrics_markdown_and_json(self, tmp_path):
        """Verify JSON export and Markdown rendering."""
        metrics = calculate_metrics(
            y_true=[0, 1],
            y_scores=[-1.5, 2.0],
            threshold=-1.0625009,
            total_time_sec=0.5,
        )
        json_path = tmp_path / "metrics.json"
        metrics.save_json(json_path)

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["total_samples"] == 2
        assert "eer_percent" in data

        md = metrics.to_markdown()
        assert "# VERA Layer 1 Evaluation Metrics Report" in md
        assert "Equal Error Rate (EER)" in md


# ---------------------------------------------------------------------------
# Dataset & Speaker Disjointness Tests
# ---------------------------------------------------------------------------
class TestDatasetAndSplits:
    """Test dataset loaders and strict speaker isolation controls."""

    def test_csv_manifest_loading(self, tmp_path):
        csv_file = tmp_path / "test_manifest.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "label", "speaker_id", "attack_type", "language", "split"])
            writer.writerow(["audio1.wav", "bonafide", "SPK_01", "-", "en", "test"])
            writer.writerow(["audio2.wav", "spoof", "SPK_02", "TTS", "en", "test"])

        ds = ASVDataset.from_csv(csv_file)
        assert len(ds) == 2
        assert ds.bonafide_count == 1
        assert ds.spoof_count == 1
        assert ds.speakers == {"SPK_01", "SPK_02"}

    def test_speaker_disjoint_splitting(self, tmp_path):
        """Verify train, val, and test splits have ZERO overlapping speakers."""
        samples = []
        for i in range(10):
            spk = f"SPK_{i:02d}"
            samples.append(AudioSample(f"path/{spk}_1.wav", "bonafide", spk, "-"))
            samples.append(AudioSample(f"path/{spk}_2.wav", "spoof", spk, "TTS"))

        ds = ASVDataset(samples)
        splits = ds.split_by_speaker(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

        train_spks = splits["train"].speakers
        val_spks = splits["val"].speakers
        test_spks = splits["test"].speakers

        assert len(train_spks) > 0
        assert len(val_spks) > 0
        assert len(test_spks) > 0

        # Assert disjoint sets
        assert train_spks.isdisjoint(val_spks)
        assert train_spks.isdisjoint(test_spks)
        assert val_spks.isdisjoint(test_spks)

    def test_speaker_leakage_error_detection(self):
        """Ensure verify_speaker_disjointness raises SpeakerLeakageError if overlap occurs."""
        s1 = AudioSample("a1.wav", "bonafide", "SPK_LEAK", "-")
        s2 = AudioSample("a2.wav", "spoof", "SPK_LEAK", "TTS")

        split_train = ASVDataset([s1])
        split_test = ASVDataset([s2])

        with pytest.raises(SpeakerLeakageError, match="CRITICAL: Speaker 'SPK_LEAK' leaked"):
            ASVDataset.verify_speaker_disjointness({"train": split_train, "test": split_test})

    def test_synthetic_benchmark_generation(self, tmp_path):
        """Verify synthetic benchmark dataset generation and validity."""
        ds, manifest_path = ASVDataset.generate_synthetic_benchmark(
            output_dir=tmp_path / "bench",
            num_speakers=4,
            samples_per_speaker=2,
        )

        assert len(ds) == 8
        assert len(ds.speakers) == 4
        assert ds.bonafide_count == 4
        assert ds.spoof_count == 4
        assert manifest_path.exists()


# ---------------------------------------------------------------------------
# End-to-End Evaluation Runner Tests
# ---------------------------------------------------------------------------
class TestEvaluationRunner:
    """Test full evaluation pipeline execution with dummy detector."""

    def test_evaluate_dataset_and_save_reports(self, tmp_path):
        ds, _ = ASVDataset.generate_synthetic_benchmark(
            output_dir=tmp_path / "bench",
            num_speakers=3,
            samples_per_speaker=2,
        )

        dummy_model = DummySpectraNet(bona_fide_logit=2.0)
        detector = VoiceAuthenticityDetector(model=dummy_model, device="cpu")

        predictions, metrics = evaluate_dataset(
            dataset=ds,
            detector=detector,
            threshold=0.0,
            progress_bar=False,
        )

        assert len(predictions) == len(ds)
        assert metrics.total_samples == len(ds)

        # Save reports
        reports = save_reports(
            predictions=predictions,
            metrics=metrics,
            output_dir=tmp_path / "reports",
            split_name="test_run",
        )

        assert reports["predictions_csv"].exists()
        assert reports["predictions_json"].exists()
        assert reports["metrics_json"].exists()
        assert reports["report_md"].exists()
