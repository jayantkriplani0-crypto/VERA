"""Script to execute Layer 1 Baseline Evaluation and generate comprehensive artifacts.

Generates:
  - evaluation/reports/baseline_results.csv
  - evaluation/reports/baseline_report.md
  - evaluation/reports/score_distribution.png
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import ASVDataset
from evaluation.metrics import calculate_metrics, compute_eer
from ml.voice_detector.detector import VoiceAuthenticityDetector
from ml.voice_detector.scorer import OFFICIAL_EER_THRESHOLD


def run_baseline_evaluation(
    manifest_path: Path,
    output_dir: Path,
    device: str = "auto"
) -> tuple[dict, list[dict]]:
    """Execute baseline evaluation and measure all latency, memory, and discrimination metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Load dataset
    dataset = ASVDataset.from_csv(manifest_path)
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Dataset is empty.")

    # 2. Speaker-separated split analysis
    splits = dataset.split_by_speaker(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    ASVDataset.verify_speaker_disjointness(splits)

    # 3. Attack type distribution
    attack_counts: dict[str, int] = {}
    for s in dataset:
        attack_counts[s.attack_type] = attack_counts.get(s.attack_type, 0) + 1

    # 4. Load detector
    t0_load = time.perf_counter()
    detector = VoiceAuthenticityDetector(device=device)
    load_time_sec = time.perf_counter() - t0_load

    # Measure memory after model load
    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = max(0.0, mem_after_load_mb - mem_before_mb)
    gpu_mem_mb = 0.0
    if torch.cuda.is_available() and "cuda" in detector.device:
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 5. Run inference and measure latencies
    latencies_ms: list[float] = []
    results: list[dict] = []
    y_true: list[int] = []
    y_scores: list[float] = []
    speaker_ids: list[str] = []

    t_start_inf = time.perf_counter()
    for idx, sample in enumerate(dataset, start=1):
        t_sample = time.perf_counter()
        pred = detector.predict_file(sample.file_path, threshold=OFFICIAL_EER_THRESHOLD)
        dt_ms = (time.perf_counter() - t_sample) * 1000.0

        latencies_ms.append(dt_ms)
        raw_score = pred.raw_score
        gt_binary = sample.ground_truth_binary
        y_true.append(gt_binary)
        y_scores.append(raw_score)
        speaker_ids.append(sample.speaker_id)

        is_pred_bonafide = bool(raw_score > OFFICIAL_EER_THRESHOLD)
        is_correct = bool(is_pred_bonafide == (gt_binary == 1))

        error_type = "None"
        if not is_correct:
            error_type = "False Positive (FAR)" if is_pred_bonafide else "False Negative (FRR)"

        rec = {
            "sample_id": idx,
            "filename": Path(sample.file_path).name,
            "file_path": sample.file_path,
            "speaker_id": sample.speaker_id,
            "attack_type": sample.attack_type,
            "language": sample.language,
            "split": sample.split,
            "ground_truth": sample.label,
            "ground_truth_binary": gt_binary,
            "raw_score": round(raw_score, 6),
            "threshold": OFFICIAL_EER_THRESHOLD,
            "predicted_label": "bonafide" if is_pred_bonafide else "spoof",
            "is_correct": is_correct,
            "error_type": error_type,
            "duration_sec": round(pred.duration_seconds, 3),
            "sample_rate_hz": pred.sample_rate,
            "latency_ms": round(dt_ms, 2),
        }
        results.append(rec)

    total_inf_sec = time.perf_counter() - t_start_inf

    # Compute overall metrics
    metrics = calculate_metrics(
        y_true=y_true,
        y_scores=y_scores,
        speaker_ids=speaker_ids,
        threshold=OFFICIAL_EER_THRESHOLD,
        total_time_sec=total_inf_sec,
    )

    # 6. Score distributions breakdown
    bonafide_scores = [r["raw_score"] for r in results if r["ground_truth_binary"] == 1]
    spoof_scores = [r["raw_score"] for r in results if r["ground_truth_binary"] == 0]

    def get_stats(arr: list[float]) -> dict:
        a = np.array(arr)
        return {
            "count": len(a),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        }

    bonafide_stats = get_stats(bonafide_scores)
    spoof_stats = get_stats(spoof_scores)

    # Latency statistics
    lat_arr = np.array(latencies_ms)
    latency_stats = {
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
        "total_sec": total_inf_sec,
        "throughput_samples_per_sec": len(lat_arr) / max(1e-6, total_inf_sec),
    }

    # False positives and false negatives list
    false_positives = [r for r in results if r["error_type"] == "False Positive (FAR)"]
    false_negatives = [r for r in results if r["error_type"] == "False Negative (FRR)"]

    report_data = {
        "dataset_composition": {
            "total_samples": total_samples,
            "bonafide_samples": len(bonafide_scores),
            "spoof_samples": len(spoof_scores),
            "speakers_count": len(dataset.speakers),
            "speakers": sorted(list(dataset.speakers)),
        },
        "speaker_splits": {
            "train": {"speakers": sorted(list(splits["train"].speakers)), "samples": len(splits["train"])},
            "val": {"speakers": sorted(list(splits["val"].speakers)), "samples": len(splits["val"])},
            "test": {"speakers": sorted(list(splits["test"].speakers)), "samples": len(splits["test"])},
        },
        "attack_distribution": attack_counts,
        "score_distributions": {
            "bonafide": bonafide_stats,
            "spoof": spoof_stats,
        },
        "discrimination_metrics": metrics.to_dict(),
        "latency_stats": latency_stats,
        "memory_usage": {
            "process_rss_mb": round(mem_after_load_mb, 2),
            "model_ram_mb": round(model_ram_mb, 2),
            "gpu_vram_mb": round(gpu_mem_mb, 2),
            "device": detector.device,
        },
        "error_analysis": {
            "false_positive_count": len(false_positives),
            "false_positives": false_positives,
            "false_negative_count": len(false_negatives),
            "false_negatives": false_negatives,
        }
    }

    # 7. Write baseline_results.csv
    csv_path = output_dir / "baseline_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 8. Generate Score Distribution Visualization
    plot_path = output_dir / "score_distribution.png"
    create_score_distribution_plot(
        bonafide_scores=bonafide_scores,
        spoof_scores=spoof_scores,
        threshold=OFFICIAL_EER_THRESHOLD,
        eer_threshold=metrics.eer_threshold,
        output_path=plot_path
    )

    # 9. Generate baseline_report.md
    report_md_path = output_dir / "baseline_report.md"
    write_markdown_report(report_data, report_md_path, plot_path.name)

    return report_data, results


def create_score_distribution_plot(
    bonafide_scores: list[float],
    spoof_scores: list[float],
    threshold: float,
    eer_threshold: float,
    output_path: Path
) -> None:
    """Create a clean, publication-ready score distribution plot."""
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot raw scores as jittered scatter / strip plot and boxplots
    y_bf = np.ones(len(bonafide_scores)) * 1.0 + np.random.uniform(-0.04, 0.04, len(bonafide_scores))
    y_sp = np.zeros(len(spoof_scores)) * 1.0 + np.random.uniform(-0.04, 0.04, len(spoof_scores))

    plt.scatter(bonafide_scores, y_bf, color="#2ecc71", s=100, alpha=0.85, label="Bona Fide (Genuine Speech)", edgecolors="black", linewidths=1.2, zorder=4)
    plt.scatter(spoof_scores, y_sp, color="#e74c3c", s=100, alpha=0.85, label="Spoof (Synthetic / Deepfake)", edgecolors="black", linewidths=1.2, zorder=4)

    # Vertical threshold lines
    plt.axvline(x=threshold, color="#3498db", linestyle="--", linewidth=2.0, label=f"Official Decision Cutoff ({threshold:+.2f})", zorder=3)
    if abs(threshold - eer_threshold) > 1e-4:
        plt.axvline(x=eer_threshold, color="#9b59b6", linestyle=":", linewidth=2.0, label=f"EER Operating Point ({eer_threshold:+.2f})", zorder=3)

    plt.yticks([0.0, 1.0], ["Spoof Cohort", "Bona Fide Cohort"], fontsize=12, fontweight="bold")
    plt.ylim(-0.3, 1.3)
    plt.xlabel("Spectra-AASIST3 Raw Bona Fide Logit (higher = more genuine)", fontsize=12, fontweight="bold")
    plt.title("Layer 1 (Voice Authenticity): Spectra-AASIST3 Score Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add score range annotations
    plt.text(np.mean(bonafide_scores), 1.15, f"Mean: {np.mean(bonafide_scores):+.2f}", ha="center", va="center", color="#27ae60", fontweight="bold", fontsize=10)
    plt.text(np.mean(spoof_scores), 0.15, f"Mean: {np.mean(spoof_scores):+.2f}", ha="center", va="center", color="#c0392b", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_markdown_report(data: dict, output_path: Path, plot_filename: str) -> None:
    """Generate comprehensive baseline markdown report."""
    comp = data["dataset_composition"]
    splits = data["speaker_splits"]
    attacks = data["attack_distribution"]
    bf_stats = data["score_distributions"]["bonafide"]
    sp_stats = data["score_distributions"]["spoof"]
    m = data["discrimination_metrics"]
    lat = data["latency_stats"]
    mem = data["memory_usage"]
    err = data["error_analysis"]

    md = [
        "# Layer 1 (Voice Authenticity) Baseline Evaluation Report",
        f"**Model:** `lab260/Spectra-AASIST3`  ",
        f"**Evaluation Date:** September 2026  ",
        f"**Status:** Fixed Pretrained Baseline (Zero Fine-Tuning)  ",
        "",
        "---",
        "",
        "## 1. Dataset Composition",
        f"- **Total Audio Utterances:** {comp['total_samples']}",
        f"- **Bona Fide (Genuine Human Speech):** {comp['bonafide_samples']} ({comp['bonafide_samples']/comp['total_samples']*100:.1f}%)",
        f"- **Spoof (Deepfake / Synthetic):** {comp['spoof_samples']} ({comp['spoof_samples']/comp['total_samples']*100:.1f}%)",
        f"- **Total Unique Speakers:** {comp['speakers_count']} ({', '.join(comp['speakers'])})",
        "",
        "## 2. Speaker-Separated Split Information",
        "Strict speaker-disjoint partitioning is applied with **ZERO** speaker identity overlap between splits:",
        "",
        "| Split | Speaker Count | Speaker IDs | Utterance Count |",
        "|---|---|---|---|",
        f"| **Train** | {len(splits['train']['speakers'])} | `{', '.join(splits['train']['speakers'])}` | {splits['train']['samples']} samples |",
        f"| **Validation** | {len(splits['val']['speakers'])} | `{', '.join(splits['val']['speakers'])}` | {splits['val']['samples']} samples |",
        f"| **Test** | {len(splits['test']['speakers'])} | `{', '.join(splits['test']['speakers'])}` | {splits['test']['samples']} samples |",
        "",
        "> [!IMPORTANT]",
        "> Speaker disjointness assertion: $\\text{speakers}(\\text{train}) \\cap \\text{speakers}(\\text{val}) = \\emptyset$, $\\text{speakers}(\\text{train}) \\cap \\text{speakers}(\\text{test}) = \\emptyset$, and $\\text{speakers}(\\text{val}) \\cap \\text{speakers}(\\text{test}) = \\emptyset$.",
        "",
        "## 3. Attack-Type Distribution",
        "| Attack Type / Category | Samples | Description |",
        "|---|---|---|",
    ]

    for atk, cnt in sorted(attacks.items()):
        desc = "Genuine human speech (no synthetic manipulation)" if atk in ("-", "bonafide") else f"Synthetic speech generation ({atk})"
        md.append(f"| `{atk}` | **{cnt}** | {desc} |")

    md.extend([
        "",
        "## 4. Raw Spectra-AASIST3 Score Distributions",
        "Raw unnormalized bona fide logits ($logits[:, 1]$) produced across the evaluation cohort:",
        "",
        "| Cohort | Count | Min | Max | Mean | Std | Median | IQR (Q25 - Q75) |",
        "|---|---|---|---|---|---|---|---|",
        f"| **Bona Fide** | {bf_stats['count']} | `{bf_stats['min']:+.4f}` | `{bf_stats['max']:+.4f}` | `{bf_stats['mean']:+.4f}` | `{bf_stats['std']:.4f}` | `{bf_stats['median']:+.4f}` | `{bf_stats['p25']:+.4f}` to `{bf_stats['p75']:+.4f}` |",
        f"| **Spoof** | {sp_stats['count']} | `{sp_stats['min']:+.4f}` | `{sp_stats['max']:+.4f}` | `{sp_stats['mean']:+.4f}` | `{sp_stats['std']:.4f}` | `{sp_stats['median']:+.4f}` | `{sp_stats['p25']:+.4f}` to `{sp_stats['p75']:+.4f}` |",
        "",
        f"![Score Distribution]({plot_filename})",
        "",
        "## 5. Core Discrimination Metrics",
        "| Metric | Measured Value | Benchmark Description |",
        "|---|---|---|",
        f"| **Equal Error Rate (EER)** | **{m['eer_percent']:.2f}%** | Primary ASVspoof / anti-spoofing benchmark metric |",
        f"| **EER Operating Threshold** | `{m['eer_threshold']:+.6f}` | Operating cutoff where FAR equals FRR |",
        f"| **ROC-AUC** | **{m['roc_auc']:.4f}** | Area Under Receiver Operating Characteristic Curve |",
        f"| **Operational Decision Threshold** | `{m['decision_threshold']:+.6f}` | Official model card decision cutoff |",
        f"| **Accuracy** | **{m['accuracy'] * 100:.2f}%** | Overall classification accuracy |",
        f"| **Precision (Bona Fide)** | **{m['precision'] * 100:.2f}%** | True Bona Fide / Total Classified Bona Fide |",
        f"| **Recall (Bona Fide)** | **{m['recall'] * 100:.2f}%** | True Bona Fide / Total Actual Bona Fide |",
        f"| **F1-Score** | **{m['f1_score']:.4f}** | Harmonic mean of precision and recall |",
        f"| **False Positive Rate (FPR / FAR)** | **{m['false_positive_rate'] * 100:.2f}%** | Spoofs incorrectly accepted as genuine speech |",
        f"| **False Negative Rate (FNR / FRR)** | **{m['false_negative_rate'] * 100:.2f}%** | Genuine speech incorrectly rejected |",
        "",
        "## 6. Confusion Matrix",
        "| Actual \\ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |",
        "|---|---|---|",
        f"| **Actual Spoof (0)** | TN = **{m['true_negatives']}** | FP = **{m['false_positives']}** |",
        f"| **Actual Bona Fide (1)** | FN = **{m['false_negatives']}** | TP = **{m['true_positives']}** |",
        "",
        "## 7. Error Case Breakdown",
        f"- **False Positives (FAR):** {err['false_positive_count']} samples",
        f"- **False Negatives (FRR):** {err['false_negative_count']} samples",
    ])

    if err['false_positive_count'] == 0 and err['false_negative_count'] == 0:
        md.append("- *Zero misclassifications observed across the baseline benchmark set under the official operational cutoff.*")
    else:
        for fp in err['false_positives']:
            md.append(f"  - FP: `{fp['filename']}` (Score: `{fp['raw_score']:+.4f}`, Spk: `{fp['speaker_id']}`, Attack: `{fp['attack_type']}`)")
        for fn in err['false_negatives']:
            md.append(f"  - FN: `{fn['filename']}` (Score: `{fn['raw_score']:+.4f}`, Spk: `{fn['speaker_id']}`)")

    md.extend([
        "",
        "## 8. Inference Latency & Resource Utilization",
        f"- **Compute Device:** `{mem['device']}`",
        f"- **Total Inference Execution Time:** {lat['total_sec']:.2f} seconds ({comp['total_samples']} samples)",
        f"- **Mean Latency:** **{lat['mean_ms']:.2f} ms / utterance**",
        f"- **Median Latency:** **{lat['median_ms']:.2f} ms / utterance**",
        f"- **p95 Latency:** **{lat['p95_ms']:.2f} ms / utterance**",
        f"- **Min / Max Latency:** {lat['min_ms']:.2f} ms / {lat['max_ms']:.2f} ms",
        f"- **Throughput:** **{lat['throughput_samples_per_sec']:.2f} utterances / second**",
        f"- **Process Resident Memory (RSS):** {mem['process_rss_mb']:.2f} MB",
        f"- **Model Parameter RAM Allocation:** ~{mem['model_ram_mb']:.2f} MB",
    ])

    if mem['gpu_vram_mb'] > 0:
        md.append(f"- **Peak GPU VRAM Allocation:** {mem['gpu_vram_mb']:.2f} MB")

    md.extend([
        "",
        "---",
        "> [!NOTE]",
        "> All metrics are strictly computed from the labeled evaluation benchmark using the fixed `lab260/Spectra-AASIST3` checkpoint. No fine-tuning, hyperparameter search, or weight optimization was performed.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main() -> None:
    manifest_path = PROJECT_ROOT / "evaluation" / "benchmark_data" / "benchmark_manifest.csv"
    output_dir = PROJECT_ROOT / "evaluation" / "reports"

    print(f"[Baseline] Starting Layer 1 baseline evaluation...")
    print(f"[Baseline] Manifest: {manifest_path}")
    print(f"[Baseline] Output Dir: {output_dir}")

    report_data, _ = run_baseline_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        device="auto"
    )

    print("\n" + "=" * 60)
    print("  BASELINE REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  - Report Markdown  : {output_dir / 'baseline_report.md'}")
    print(f"  - Results CSV      : {output_dir / 'baseline_results.csv'}")
    print(f"  - Distribution Plot: {output_dir / 'score_distribution.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
