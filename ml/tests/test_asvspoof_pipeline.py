"""Unit tests for ASVspoof benchmark loader, attack breakdown, and pre-flight checks."""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from evaluation.dataset import ASVDataset, AudioSample
from evaluation.metrics import calculate_breakdown_by_attack, calculate_metrics, compute_eer
from evaluation.run_asvspoof_benchmark import run_benchmark


class TestASVspoofProtocolLoader:
    """Test parsing of ASVspoof 2019 and 2021 protocol formats."""

    def test_parse_asvspoof_2019_format(self, tmp_path):
        proto_content = (
            "# ASVspoof 2019 LA evaluation protocol\n"
            "LA_0079 LA_E_2833633 - - bonafide\n"
            "LA_0079 LA_E_8877456 - A07 spoof\n"
            "LA_0080 LA_E_1122334 - A12 spoof\n"
            "LA_0080 LA_E_9988776 - - bonafide\n"
        )
        proto_file = tmp_path / "asvspoof2019_sample.txt"
        proto_file.write_text(proto_content, encoding="utf-8")

        # Fake audio directory with mock flac files
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "LA_E_2833633.flac").touch()
        (audio_dir / "LA_E_8877456.flac").touch()

        dataset = ASVDataset.from_asvspoof_protocol(
            protocol_file=proto_file,
            audio_dir=audio_dir,
            file_ext=None  # Test auto-detection of .flac
        )

        assert len(dataset) == 4
        assert dataset.bonafide_count == 2
        assert dataset.spoof_count == 2
        assert dataset.speakers == {"LA_0079", "LA_0080"}
        assert "A07" in dataset.attack_types
        assert "A12" in dataset.attack_types
        assert "-" in dataset.attack_types

        # Check sample details
        s0 = dataset.samples[0]
        assert s0.speaker_id == "LA_0079"
        assert s0.is_bonafide is True
        assert s0.attack_type == "-"
        assert s0.file_path.endswith(".flac")

        s1 = dataset.samples[1]
        assert s1.speaker_id == "LA_0079"
        assert s1.is_bonafide is False
        assert s1.attack_type == "A07"

    def test_parse_asvspoof_2021_trial_metadata(self, tmp_path):
        proto_content = (
            "# ASVspoof 2021 trial metadata format\n"
            "LA_0023 LA_E_0000001 alaw pstn - bonafide notrim asvspoof\n"
            "LA_0023 LA_E_0000002 g722 ip A07 spoof notrim asvspoof\n"
            "LA_0045 LA_E_0000003 opus voip A19 spoof notrim asvspoof\n"
        )
        proto_file = tmp_path / "asvspoof2021_metadata.txt"
        proto_file.write_text(proto_content, encoding="utf-8")

        audio_dir = tmp_path / "audio21"
        audio_dir.mkdir()

        dataset = ASVDataset.from_asvspoof_protocol(
            protocol_file=proto_file,
            audio_dir=audio_dir,
            file_ext=".flac"
        )

        assert len(dataset) == 3
        assert dataset.bonafide_count == 1
        assert dataset.spoof_count == 2
        s0 = dataset.samples[0]
        assert s0.codec == "alaw"
        assert s0.transmission == "pstn"
        assert s0.is_bonafide is True

        s1 = dataset.samples[1]
        assert s1.codec == "g722"
        assert s1.transmission == "ip"
        assert s1.attack_type == "A07"
        assert s1.is_bonafide is False


class TestAttackBreakdownMetrics:
    """Test performance breakdown by attack category."""

    def test_calculate_breakdown_by_attack(self):
        # 2 bonafide, 2 of A01, 2 of A02
        y_true = np.array([1, 1, 0, 0, 0, 0])
        y_scores = np.array([1.5, 0.5, -3.0, -4.0, 0.2, -5.0])  # Note: 0.2 is missed spoof
        attacks = ["-", "-", "A01", "A01", "A02", "A02"]
        threshold = -1.0

        breakdown = calculate_breakdown_by_attack(y_true, y_scores, attacks, threshold=threshold)

        assert "-" in breakdown
        assert "A01" in breakdown
        assert "A02" in breakdown

        # Bona fide
        assert breakdown["-"]["category"] == "bonafide"
        assert breakdown["-"]["accuracy_pct"] == 100.0  # Both > -1.0

        # A01: scores -3.0, -4.0 (both <= -1.0 -> 100% detected)
        assert breakdown["A01"]["detection_rate_pct"] == 100.0
        assert breakdown["A01"]["detected_count"] == 2
        assert breakdown["A01"]["missed_count"] == 0

        # A02: scores 0.2 (missed!), -5.0 (detected) -> 50% detection rate
        assert breakdown["A02"]["detection_rate_pct"] == 50.0
        assert breakdown["A02"]["detected_count"] == 1
        assert breakdown["A02"]["missed_count"] == 1
        assert breakdown["A02"]["miss_rate_pct"] == 50.0


class TestBenchmarkPreflightChecks:
    """Ensure benchmark runner diagnoses missing files and stops safely without crash."""

    def test_preflight_missing_protocol(self, tmp_path):
        non_existent_proto = tmp_path / "missing.txt"
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            run_benchmark(
                protocol_path=non_existent_proto,
                audio_dir=audio_dir,
            )

    def test_preflight_missing_audio_files(self, tmp_path):
        proto_content = "LA_0079 LA_E_2833633 - - bonafide\n"
        proto_file = tmp_path / "test_proto.txt"
        proto_file.write_text(proto_content, encoding="utf-8")

        empty_audio_dir = tmp_path / "empty_audio"
        empty_audio_dir.mkdir()

        res = run_benchmark(
            protocol_path=proto_file,
            audio_dir=empty_audio_dir,
            output_dir=tmp_path / "out",
        )

        assert res["status"] == "DATASET_NOT_FOUND"
        assert res["existing_audio_files"] == 0
        assert res["missing_audio_files"] == 1
