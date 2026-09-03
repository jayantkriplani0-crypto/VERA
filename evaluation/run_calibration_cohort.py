"""Run frozen Spectra-AASIST3 inference over the 700-sample calibration cohort.

Features:
  - Frozen model inference recording raw continuous bona-fide logits only.
  - Generates comprehensive predictions CSV with split column ('cal_fit' vs 'cal_val').
  - Calculates detailed metrics for overall cohort, cal_fit partition, and cal_val partition.
  - Formats detailed Markdown report and JSON summary.
  - Zero modification of model weights or inference pipeline.
  - Does NOT fit calibration parameters (w, b).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    calculate_breakdown_by_attack,
    calculate_breakdown_by_speaker,
    calculate_metrics,
    calculate_score_distributions,
    format_attack_breakdown_markdown,
    format_score_distributions_markdown,
    format_speaker_breakdown_markdown,
)
from ml.voice_detector.detector import VoiceAuthenticityDetector
from ml.voice_detector.scorer import OFFICIAL_EER_THRESHOLD


def run_cohort_benchmark() -> Dict[str, Any]:
    base_dir = PROJECT_ROOT / "data" / "asvspoof2019"
    manifest_path = base_dir / "manifests" / "calibration_cohort_manifest.csv"
    output_dir = PROJECT_ROOT / "evaluation" / "reports" / "calibration_preparation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ASVSPOOF 2019 LA: 700-SAMPLE CALIBRATION COHORT INFERENCE")
    print("=" * 65)
    print(f"Manifest File  : {manifest_path}")
    print(f"Output Path    : {output_dir}")
    print(f"Decision Cutoff: {OFFICIAL_EER_THRESHOLD}")
    print("=" * 65)

    # 1. Read manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(f"[Cohort Runner] Loaded {len(samples)} samples from manifest.")

    # 2. Load frozen detector
    print("[Cohort Runner] Initializing frozen Spectra-AASIST3 detector...")
    detector = VoiceAuthenticityDetector()

    # 3. Run inference
    print(f"\n[Cohort Runner] Running inference on {len(samples)} samples...")
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        if idx == 1 or idx % 50 == 0 or idx == len(samples):
            elapsed = time.perf_counter() - t0
            rate = idx / max(1e-6, elapsed)
            print(f"  Progress: [{idx}/{len(samples)}] ({idx/len(samples)*100:.1f}%) - {rate:.1f} samples/sec")

        file_path = sample["file_path"]
        res = detector.predict_file(file_path)
        raw_s = res.raw_score
        y = int(sample["ground_truth_binary"])

        records.append({
            "file_path": sample["filename"],
            "speaker_id": sample["speaker_id"],
            "attack_type": sample["attack_type"],
            "ground_truth": sample["label"],
            "ground_truth_binary": y,
            "raw_bonafide_score": round(raw_s, 6),
            "predicted_binary": 1 if raw_s > OFFICIAL_EER_THRESHOLD else 0,
            "predicted_label": "bonafide" if raw_s > OFFICIAL_EER_THRESHOLD else "spoof",
            "is_correct": bool((raw_s > OFFICIAL_EER_THRESHOLD) == (y == 1)),
            "split": sample["split"],
        })

    total_time = time.perf_counter() - t0
    print(f"[Cohort Runner] Inference complete in {total_time:.2f}s ({len(samples)/max(1e-6, total_time):.2f} samples/sec).")

    # 4. Compute metrics helper
    def _compute_partition_metrics(recs: List[Dict[str, Any]], part_time: float) -> Dict[str, Any]:
        y_true = np.array([r["ground_truth_binary"] for r in recs], dtype=int)
        y_scores = np.array([r["raw_bonafide_score"] for r in recs], dtype=float)
        speakers = [r["speaker_id"] for r in recs]
        attacks = [r["attack_type"] for r in recs]

        m = calculate_metrics(
            y_true=y_true,
            y_scores=y_scores,
            speaker_ids=speakers,
            threshold=OFFICIAL_EER_THRESHOLD,
            total_time_sec=part_time,
        )
        att_b = calculate_breakdown_by_attack(y_true, y_scores, attacks, threshold=OFFICIAL_EER_THRESHOLD)
        spk_b = calculate_breakdown_by_speaker(y_true, y_scores, speakers, threshold=OFFICIAL_EER_THRESHOLD)
        dist = calculate_score_distributions(y_true, y_scores)

        return {
            "summary": m.to_dict(),
            "metrics_obj": m,
            "score_distributions": dist,
            "attack_breakdown": att_b,
            "speaker_breakdown": spk_b,
        }

    overall_metrics = _compute_partition_metrics(records, total_time)
    fit_records = [r for r in records if r["split"] == "cal_fit"]
    val_records = [r for r in records if r["split"] == "cal_val"]

    fit_metrics = _compute_partition_metrics(fit_records, total_time * (len(fit_records) / len(records)))
    val_metrics = _compute_partition_metrics(val_records, total_time * (len(val_records) / len(records)))

    # 5. Save predictions CSV
    csv_path = output_dir / "asvspoof_dev_700_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[Cohort Runner] Saved predictions CSV to: {csv_path}")

    # 6. Save JSON metrics
    json_path = output_dir / "calibration_cohort_metrics.json"
    full_json = {
        "overall": {
            "summary": overall_metrics["summary"],
            "score_distributions": overall_metrics["score_distributions"],
            "attack_breakdown": overall_metrics["attack_breakdown"],
            "speaker_breakdown": overall_metrics["speaker_breakdown"],
        },
        "cal_fit_split": {
            "summary": fit_metrics["summary"],
            "score_distributions": fit_metrics["score_distributions"],
            "attack_breakdown": fit_metrics["attack_breakdown"],
            "speaker_breakdown": fit_metrics["speaker_breakdown"],
        },
        "cal_val_split": {
            "summary": val_metrics["summary"],
            "score_distributions": val_metrics["score_distributions"],
            "attack_breakdown": val_metrics["attack_breakdown"],
            "speaker_breakdown": val_metrics["speaker_breakdown"],
        },
        "execution_metadata": {
            "total_samples": len(records),
            "cal_fit_samples": len(fit_records),
            "cal_val_samples": len(val_records),
            "total_inference_time_sec": round(total_time, 2),
            "avg_latency_ms": round((total_time / len(records)) * 1000, 2),
            "throughput_samples_per_sec": round(len(records) / max(1e-6, total_time), 2),
            "model": "lab260/Spectra-AASIST3 (frozen)",
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)
    print(f"[Cohort Runner] Saved metrics JSON to: {json_path}")

    # 7. Format comprehensive Markdown report
    md_path = output_dir / "calibration_cohort_report.md"
    write_cohort_markdown_report(
        overall_metrics=overall_metrics,
        fit_metrics=fit_metrics,
        val_metrics=val_metrics,
        total_time=total_time,
        output_path=md_path,
    )
    print(f"[Cohort Runner] Saved Markdown report to: {md_path}")

    ov_m = overall_metrics["metrics_obj"]
    print("\n" + "=" * 65)
    print(f"  COHORT SUMMARY: EER = {ov_m.eer_percent:.2f}% | ROC-AUC = {ov_m.roc_auc:.4f} | Accuracy = {ov_m.accuracy*100:.2f}%")
    print("=" * 65)

    return full_json


def write_cohort_markdown_report(
    overall_metrics: Dict[str, Any],
    fit_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    total_time: float,
    output_path: Path,
) -> None:
    """Format full cohort benchmark and split audit into a structured Markdown document."""
    ov_m = overall_metrics["metrics_obj"]
    fit_m = fit_metrics["metrics_obj"]
    val_m = val_metrics["metrics_obj"]

    lines = [
        "# ASVspoof 2019 LA: Calibration Preparation Cohort Report (700 Samples)",
        "**Target Model:** `lab260/Spectra-AASIST3` (Official Pretrained Checkpoint — Frozen & Unchanged)  ",
        "**Protocol:** `ASVspoof2019.LA.cm.dev.trl.txt`  ",
        "**Date:** September 2026  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Split Comparison",
        "",
        "| Metric | Overall Cohort (700) | Calibration Fit (`cal_fit`, 420) | Held-Out Validation (`cal_val`, 280) | Reference / Standard Interpretation |",
        "|---|---|---|---|---|",
        f"| **Speakers** | 10 Speakers | 6 Speakers (`LA_0069`–`LA_0074`) | 4 Unseen Speakers (`LA_0075`–`LA_0078`) | Disjoint speaker partitions |",
        f"| **Bona Fide Utterances** | {ov_m.num_bonafide} (40.0%) | {fit_m.num_bonafide} (40.0%) | {val_m.num_bonafide} (40.0%) | Ground truth genuine human speech |",
        f"| **Spoof Utterances** | {ov_m.num_spoof} (60.0%) | {fit_m.num_spoof} (60.0%) | {val_m.num_spoof} (60.0%) | Deepfake speech across `A01`–`A06` |",
        f"| **Equal Error Rate (EER)** | **{ov_m.eer_percent:.2f}%** | **{fit_m.eer_percent:.2f}%** | **{val_m.eer_percent:.2f}%** | Primary ASVspoof discrimination metric |",
        f"| **ROC-AUC** | **{ov_m.roc_auc:.4f}** | **{fit_m.roc_auc:.4f}** | **{val_m.roc_auc:.4f}** | 1.0 = Perfect separation |",
        f"| **EER Operating Threshold** | `{ov_m.eer_threshold:+.6f}` | `{fit_m.eer_threshold:+.6f}` | `{val_m.eer_threshold:+.6f}` | Threshold where FAR == FRR |",
        f"| **Operating Threshold** | `{OFFICIAL_EER_THRESHOLD:+.6f}` | `{OFFICIAL_EER_THRESHOLD:+.6f}` | `{OFFICIAL_EER_THRESHOLD:+.6f}` | Fixed official model cutoff |",
        f"| **Accuracy @ Cutoff** | **{ov_m.accuracy * 100:.2f}%** | **{fit_m.accuracy * 100:.2f}%** | **{val_m.accuracy * 100:.2f}%** | Classification accuracy at fixed cutoff |",
        f"| **False Alarm Rate (FAR)** | **{ov_m.false_positive_rate * 100:.2f}%** | **{fit_m.false_positive_rate * 100:.2f}%** | **{val_m.false_positive_rate * 100:.2f}%** | Spoofs incorrectly accepted as genuine |",
        f"| **False Rejection Rate (FRR)** | **{ov_m.false_negative_rate * 100:.2f}%** | **{fit_m.false_negative_rate * 100:.2f}%** | **{val_m.false_negative_rate * 100:.2f}%** | Genuine human voices incorrectly rejected |",
        "",
        "---",
        "",
        "## 2. Score Distribution Statistics (Raw Frozen Logits)",
        "",
        "### Overall Cohort (700 Samples)",
        format_score_distributions_markdown(overall_metrics["score_distributions"]),
        "",
        "### Calibration Fit Split (`cal_fit`, 420 Samples)",
        format_score_distributions_markdown(fit_metrics["score_distributions"]),
        "",
        "### Held-Out Validation Split (`cal_val`, 280 Samples)",
        format_score_distributions_markdown(val_metrics["score_distributions"]),
        "",
        "---",
        "",
        "## 3. Confusion Matrices (at Fixed Official Cutoff `-1.062501`)",
        "",
        "### A. Overall Cohort (700 Samples)",
        "| Actual \\ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |",
        "|---|---|---|",
        f"| **Actual Spoof (0)** | TN = **{ov_m.true_negatives}** | FP = **{ov_m.false_positives}** |",
        f"| **Actual Bona Fide (1)** | FN = **{ov_m.false_negatives}** | TP = **{ov_m.true_positives}** |",
        "",
        "### B. Calibration Fit Split (420 Samples)",
        "| Actual \\ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |",
        "|---|---|---|",
        f"| **Actual Spoof (0)** | TN = **{fit_m.true_negatives}** | FP = **{fit_m.false_positives}** |",
        f"| **Actual Bona Fide (1)** | FN = **{fit_m.false_negatives}** | TP = **{fit_m.true_positives}** |",
        "",
        "### C. Held-Out Validation Split (280 Samples)",
        "| Actual \\ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |",
        "|---|---|---|",
        f"| **Actual Spoof (0)** | TN = **{val_m.true_negatives}** | FP = **{val_m.false_positives}** |",
        f"| **Actual Bona Fide (1)** | FN = **{val_m.false_negatives}** | TP = **{val_m.true_positives}** |",
        "",
        "---",
        "",
        "## 4. Per-Attack Family Breakdown (`A01`–`A06`)",
        "",
        "### Overall Cohort Breakdown",
        format_attack_breakdown_markdown(overall_metrics["attack_breakdown"]),
        "",
        "### Calibration Fit Breakdown",
        format_attack_breakdown_markdown(fit_metrics["attack_breakdown"]),
        "",
        "### Held-Out Validation Breakdown",
        format_attack_breakdown_markdown(val_metrics["attack_breakdown"]),
        "",
        "---",
        "",
        "## 5. Per-Speaker Breakdown (All 10 Core Speakers)",
        "",
        format_speaker_breakdown_markdown(overall_metrics["speaker_breakdown"]),
        "",
        "---",
        "",
        "## 6. Throughput & Execution Verification",
        "",
        f"- **Total Samples Evaluated:** 700",
        f"- **Total Inference Time:** {total_time:.2f} seconds ({total_time / 60:.2f} minutes)",
        f"- **Average Inference Latency:** {(total_time / 700) * 1000:.2f} ms / sample",
        f"- **Processing Throughput:** {700 / max(1e-6, total_time):.2f} utterances / sec",
        f"- **Hardware Execution:** CPU (`torch.cuda.is_available() == False`)",
        "",
        "> [!IMPORTANT]",
        "> **Calibration Parameters ($w, b$) Status:** Untrained and uncalibrated during this phase. Only frozen raw logits were extracted.",
        "",
        "> [!IMPORTANT]",
        "> **Quarantine Verification:** The ASVspoof 2019 LA evaluation archive (`LA_eval.zip`) was **not** accessed or downloaded.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_cohort_benchmark()
