"""Reproducible Layer 1 Evaluation Pipeline for lab260/Spectra-AASIST3.

Executes inference across a labeled dataset, saves detailed per-sample predictions,
computes ASVspoof / anti-spoofing metrics (EER, ROC-AUC, F1, Confusion Matrix, Latency),
and generates comprehensive JSON/Markdown reports in evaluation/reports/.

Usage:
    python evaluation/run_evaluation.py --manifest path/to/manifest.csv
    python evaluation/run_evaluation.py --generate-benchmark --device auto
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import ASVDataset, AudioSample
from evaluation.metrics import EvaluationMetrics, calculate_metrics
from ml.voice_detector.detector import VoiceAuthenticityDetector
from ml.voice_detector.model_loader import MODEL_HUB_REPO
from ml.voice_detector.scorer import OFFICIAL_EER_THRESHOLD


def evaluate_dataset(
    dataset: ASVDataset,
    detector: VoiceAuthenticityDetector,
    threshold: float = OFFICIAL_EER_THRESHOLD,
    progress_bar: bool = True
) -> tuple[List[Dict[str, Any]], EvaluationMetrics]:
    """Run full evaluation across all samples in the dataset.

    For every sample:
      - Runs Spectra-AASIST3 inference
      - Records: file_path, label, ground_truth, speaker_id, attack_type,
        raw_score, predicted_label, latency_ms.

    Returns:
        tuple of (sample_predictions_list, EvaluationMetrics)
    """
    predictions: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_scores: List[float] = []
    speaker_ids: List[str] = []

    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    print(f"\n[Evaluation] Starting inference on {total_samples} samples...")
    print(f"[Evaluation] Model: {detector.model_repo} on {detector.device}")
    print(f"[Evaluation] Decision Threshold: {threshold:+.6f}")
    print("-" * 60)

    t_start_total = time.perf_counter()

    for idx, sample in enumerate(dataset, start=1):
        t0 = time.perf_counter()
        try:
            res = detector.predict_file(sample.file_path, threshold=threshold)
            raw_score = res.raw_score
            pred_label = res.interpretation.predicted_label
            duration = res.duration_seconds
            sr = res.sample_rate
        except Exception as exc:
            print(f"WARNING: Error processing '{sample.file_path}': {exc}", file=sys.stderr)
            raw_score = -999.0
            pred_label = "ERROR"
            duration = 0.0
            sr = 0

        dt_ms = (time.perf_counter() - t0) * 1000.0

        gt_binary = sample.ground_truth_binary
        y_true.append(gt_binary)
        y_scores.append(raw_score)
        speaker_ids.append(sample.speaker_id)

        rec = {
            "sample_id": idx,
            "file_path": sample.file_path,
            "speaker_id": sample.speaker_id,
            "attack_type": sample.attack_type,
            "language": sample.language,
            "split": sample.split,
            "ground_truth_label": sample.label,
            "ground_truth_binary": gt_binary,
            "raw_score": round(raw_score, 6),
            "predicted_label": pred_label,
            "is_correct": bool((raw_score > threshold) == (gt_binary == 1)),
            "duration_sec": round(duration, 3),
            "sample_rate_hz": sr,
            "latency_ms": round(dt_ms, 2),
        }
        predictions.append(rec)

        if progress_bar:
            status = "CORRECT" if rec["is_correct"] else "MISS"
            print(
                f"[{idx:03d}/{total_samples:03d}] Spk: {sample.speaker_id:<8} "
                f"GT: {sample.label:<8} Score: {raw_score:+8.4f} "
                f"[{status}] ({dt_ms:5.1f} ms)"
            )

    total_time_sec = time.perf_counter() - t_start_total

    # Calculate standard ASVspoof / anti-spoofing metrics
    metrics = calculate_metrics(
        y_true=y_true,
        y_scores=y_scores,
        speaker_ids=speaker_ids,
        threshold=threshold,
        total_time_sec=total_time_sec,
    )

    return predictions, metrics


def save_reports(
    predictions: List[Dict[str, Any]],
    metrics: EvaluationMetrics,
    output_dir: str | Path,
    split_name: str = "eval"
) -> Dict[str, Path]:
    """Save predictions (CSV & JSON) and metrics report (JSON & Markdown) to output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{split_name}_{timestamp}"

    # 1. Predictions CSV
    csv_path = out_dir / f"predictions_{prefix}.csv"
    if predictions:
        fieldnames = list(predictions[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

    # 2. Predictions JSON
    json_preds_path = out_dir / f"predictions_{prefix}.json"
    with open(json_preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # 3. Metrics JSON
    metrics_json_path = out_dir / f"metrics_{prefix}.json"
    metrics.save_json(metrics_json_path)

    # 4. Metrics Markdown Report
    metrics_md_path = out_dir / f"report_{prefix}.md"
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(metrics.to_markdown())

    return {
        "predictions_csv": csv_path,
        "predictions_json": json_preds_path,
        "metrics_json": metrics_json_path,
        "report_md": metrics_md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectra-AASIST3 Layer 1 Evaluation Runner"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to CSV/JSON dataset manifest or ASVspoof protocol file"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory containing audio files (for ASVspoof protocol)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Dataset split to evaluate ('all', 'test', 'val', 'train')"
    )
    parser.add_argument(
        "--split-speakers",
        action="store_true",
        help="Partition the dataset into speaker-disjoint train/val/test splits"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=OFFICIAL_EER_THRESHOLD,
        help=f"Operational decision threshold (default: {OFFICIAL_EER_THRESHOLD})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (default), 'cpu', or 'cuda'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "reports"),
        help="Directory where evaluation artifacts and reports will be saved"
    )
    parser.add_argument(
        "--generate-benchmark",
        action="store_true",
        help="Generate synthetic multi-speaker benchmark dataset and evaluate"
    )
    parser.add_argument(
        "--num-benchmark-speakers",
        type=int,
        default=6,
        help="Number of synthetic benchmark speakers (default: 6)"
    )
    args = parser.parse_args()

    # Step 1: Load or generate dataset
    if args.generate_benchmark or (args.manifest is None and not args.generate_benchmark):
        bench_dir = PROJECT_ROOT / "evaluation" / "benchmark_data"
        print(f"[Dataset] Generating synthetic benchmark dataset with {args.num_benchmark_speakers} speakers at '{bench_dir}'...")
        dataset, manifest_path = ASVDataset.generate_synthetic_benchmark(
            output_dir=bench_dir,
            num_speakers=args.num_benchmark_speakers,
            samples_per_speaker=4,
        )
        print(f"[Dataset] Generated {len(dataset)} samples across {len(dataset.speakers)} speakers. Manifest saved: {manifest_path}")
    else:
        print(f"[Dataset] Loading dataset from '{args.manifest}'...")
        dataset = ASVDataset.auto_load(args.manifest, audio_dir=args.audio_dir)

    print(f"[Dataset] Loaded {len(dataset)} samples ({dataset.bonafide_count} bona fide, {dataset.spoof_count} spoof).")
    print(f"[Dataset] Unique speakers: {len(dataset.speakers)}")

    # Step 2: Handle speaker-disjoint splitting if requested
    target_dataset = dataset
    if args.split_speakers:
        print("\n[Splitting] Partitioning dataset into speaker-disjoint subsets (60/20/20)...")
        splits = dataset.split_by_speaker(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        print(f"[Splitting] Train: {len(splits['train'])} samples ({len(splits['train'].speakers)} spks)")
        print(f"[Splitting] Val  : {len(splits['val'])} samples ({len(splits['val'].speakers)} spks)")
        print(f"[Splitting] Test : {len(splits['test'])} samples ({len(splits['test'].speakers)} spks)")
        ASVDataset.verify_speaker_disjointness(splits)
        print("[Splitting] Verified: ZERO speaker overlap across splits.")

        if args.split.lower() in splits:
            target_dataset = splits[args.split.lower()]
            print(f"[Evaluation] Focusing on split: '{args.split.lower()}' ({len(target_dataset)} samples)")
    elif args.split.lower() != "all":
        filtered = dataset.filter_by_split(args.split)
        if len(filtered) > 0:
            target_dataset = filtered
            print(f"[Evaluation] Filtered to split '{args.split}': {len(target_dataset)} samples")

    # Step 3: Load Model & Detector
    detector = VoiceAuthenticityDetector(device=args.device)

    # Step 4: Run Evaluation
    predictions, metrics = evaluate_dataset(
        dataset=target_dataset,
        detector=detector,
        threshold=args.threshold,
        progress_bar=True
    )

    # Step 5: Save Reports & Display Summary
    saved_files = save_reports(
        predictions=predictions,
        metrics=metrics,
        output_dir=args.output_dir,
        split_name=args.split if args.split != "all" else "full",
    )

    print("\n" + "=" * 60)
    print("  LAYER 1 EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(metrics.to_markdown())
    print("\n[Artifacts Generated]")
    for k, v in saved_files.items():
        print(f"  - {k:<18}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
