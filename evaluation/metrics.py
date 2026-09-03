"""Evaluation metrics module for Layer 1 Voice Authenticity (Spectra-AASIST3).

Calculates standard audio deepfake detection and ASVspoof metrics:
  - Equal Error Rate (EER) and operating threshold
  - ROC-AUC (Area Under Receiver Operating Characteristic)
  - Confusion Matrix (TP, FP, TN, FN)
  - Precision, Recall, F1-Score, Accuracy
  - False Positive Rate (FPR / False Alarm Rate)
  - False Negative Rate (FNR / Miss Rate)
  - Latency and throughput benchmarks
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding decision threshold.

    In binary speech anti-spoofing:
      y_true = 1 -> Bona Fide (Genuine speech)
      y_true = 0 -> Spoof (Synthetic/Deepfake)
      y_scores   -> Bona fide logit (higher = more genuine)

    EER is the operating point where False Positive Rate (FPR) == False Negative Rate (FNR).

    Args:
        y_true: 1D array of binary ground-truth labels (1 = bona fide, 0 = spoof).
        y_scores: 1D array of continuous bona fide logits.

    Returns:
        tuple of (eer_percentage, eer_threshold)
    """
    if len(np.unique(y_true)) < 2:
        # Cannot compute EER if only one class is present
        return 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the index where FPR and FNR are closest
    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    # Interpolate for precise EER calculation
    eer = float((fpr[idx] + fnr[idx]) / 2.0) * 100.0  # as percentage
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


@dataclass(frozen=True)
class EvaluationMetrics:
    """Comprehensive evaluation metrics summary."""
    # Dataset composition
    total_samples: int
    num_bonafide: int
    num_spoof: int
    num_speakers: int

    # Operating decision point
    decision_threshold: float

    # Confusion matrix
    true_positives: int    # Correctly accepted bona fide
    false_positives: int   # Spoof incorrectly accepted as bona fide (FAR)
    true_negatives: int    # Correctly rejected spoof
    false_negatives: int   # Bona fide incorrectly rejected (FRR)

    # Core classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float  # FPR / FAR
    false_negative_rate: float  # FNR / FRR

    # Threshold-independent metrics
    roc_auc: float
    eer_percent: float
    eer_threshold: float

    # Timing and throughput
    total_inference_time_sec: float
    avg_latency_ms: float
    throughput_samples_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a Python dictionary."""
        return asdict(self)

    def save_json(self, output_path: str | Path) -> None:
        """Save metrics to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """Format metrics as a GitHub-flavored Markdown report."""
        lines = [
            "# VERA Layer 1 Evaluation Metrics Report",
            "",
            "## 1. Dataset Overview",
            f"- **Total Samples Evaluated:** {self.total_samples}",
            f"- **Bona Fide (Genuine) Samples:** {self.num_bonafide} ({self.num_bonafide / max(1, self.total_samples) * 100:.1f}%)",
            f"- **Spoof (Deepfake) Samples:** {self.num_spoof} ({self.num_spoof / max(1, self.total_samples) * 100:.1f}%)",
            f"- **Unique Speakers:** {self.num_speakers}",
            "",
            "## 2. Discrimination Performance",
            "| Metric | Value | Reference Note |",
            "|---|---|---|",
            f"| **Equal Error Rate (EER)** | **{self.eer_percent:.2f}%** | Primary ASVspoof benchmark metric (lower is better) |",
            f"| **EER Operating Threshold** | `{self.eer_threshold:+.6f}` | Threshold where FAR == FRR |",
            f"| **ROC-AUC** | **{self.roc_auc:.4f}** | Area under ROC curve (1.0 = perfect separation) |",
            f"| **Decision Threshold** | `{self.decision_threshold:+.6f}` | Threshold applied for binary classification |",
            f"| **Accuracy** | {self.accuracy * 100:.2f}% | Overall classification accuracy |",
            f"| **Precision (Bona Fide)** | {self.precision * 100:.2f}% | True bona fide / total classified bona fide |",
            f"| **Recall (Bona Fide)** | {self.recall * 100:.2f}% | True bona fide / total actual bona fide |",
            f"| **F1-Score** | {self.f1_score:.4f} | Harmonic mean of Precision and Recall |",
            f"| **False Positive Rate (FPR / FAR)** | {self.false_positive_rate * 100:.2f}% | Spoofs incorrectly accepted as genuine |",
            f"| **False Negative Rate (FNR / FRR)** | {self.false_negative_rate * 100:.2f}% | Genuine voices incorrectly rejected |",
            "",
            "## 3. Confusion Matrix",
            "| Actual \\ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |",
            "|---|---|---|",
            f"| **Actual Spoof (0)** | TN = **{self.true_negatives}** | FP = **{self.false_positives}** |",
            f"| **Actual Bona Fide (1)** | FN = **{self.false_negatives}** | TP = **{self.true_positives}** |",
            "",
            "## 4. Benchmark Throughput & Latency",
            f"- **Total Inference Time:** {self.total_inference_time_sec:.2f} seconds",
            f"- **Average Latency:** {self.avg_latency_ms:.2f} ms / sample",
            f"- **Inference Throughput:** {self.throughput_samples_per_sec:.2f} utterances / sec",
            "",
            "> [!NOTE]",
            "> All scores are raw bona fide logits produced by `lab260/Spectra-AASIST3`. No fake probabilities or percentage conversions were used.",
        ]
        return "\n".join(lines)


def calculate_metrics(
    y_true: np.ndarray | list[int],
    y_scores: np.ndarray | list[float],
    speaker_ids: list[str] | None = None,
    threshold: float = -1.0625009,
    total_time_sec: float = 0.0,
) -> EvaluationMetrics:
    """Compute all evaluation metrics from ground-truth labels and raw model logit scores.

    Args:
        y_true: Binary ground-truth labels (1 = bona fide, 0 = spoof).
        y_scores: Continuous model logit scores (higher = bona fide).
        speaker_ids: Optional list of speaker IDs corresponding to each sample.
        threshold: Operational decision threshold (default: -1.0625009).
        total_time_sec: Total elapsed inference execution time in seconds.

    Returns:
        EvaluationMetrics instance.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)

    if len(y_t) != len(y_s):
        raise ValueError(f"Mismatch in length: y_true ({len(y_t)}) vs y_scores ({len(y_s)})")
    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty dataset.")

    total_samples = len(y_t)
    num_bonafide = int(np.sum(y_t == 1))
    num_spoof = int(np.sum(y_t == 0))
    num_speakers = len(set(speaker_ids)) if speaker_ids else 0

    # Binary predictions using the specified threshold
    y_pred = (y_s > threshold).astype(int)

    # Confusion matrix elements
    # confusion_matrix format: [[TN, FP], [FN, TP]] for labels [0, 1]
    if len(np.unique(y_t)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    else:
        # Edge case: only one class in evaluation set
        if y_t[0] == 1:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            tn, fp = 0, 0
        else:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            tp, fn = 0, 0

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    # ROC-AUC & EER
    if num_bonafide > 0 and num_spoof > 0:
        auc = float(roc_auc_score(y_t, y_s))
        eer, eer_thresh = compute_eer(y_t, y_s)
    else:
        auc = 1.0 if acc == 1.0 else 0.5
        eer, eer_thresh = 0.0, threshold

    # Performance
    avg_latency_ms = (total_time_sec / max(1, total_samples)) * 1000.0
    throughput = total_samples / max(1e-6, total_time_sec)

    return EvaluationMetrics(
        total_samples=total_samples,
        num_bonafide=num_bonafide,
        num_spoof=num_spoof,
        num_speakers=num_speakers,
        decision_threshold=threshold,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        roc_auc=auc,
        eer_percent=eer,
        eer_threshold=eer_thresh,
        total_inference_time_sec=total_time_sec,
        avg_latency_ms=avg_latency_ms,
        throughput_samples_per_sec=throughput,
    )


def calculate_breakdown_by_attack(
    y_true: np.ndarray | list[int],
    y_scores: np.ndarray | list[float],
    attack_types: list[str],
    threshold: float = -1.0625009,
) -> dict[str, dict[str, Any]]:
    """Compute performance metrics broken down by individual spoof attack type or source.

    Args:
        y_true: 1D array of ground truth labels (1 = bona fide, 0 = spoof).
        y_scores: 1D array of raw bona fide logit scores.
        attack_types: List of attack identifiers (e.g. 'A01', 'TTS-VITS', '-' for bonafide).
        threshold: Decision threshold for classification (scores > threshold -> bona fide).

    Returns:
        Dictionary mapping attack_type to breakdown metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)
    attacks = list(attack_types)

    if len(y_t) != len(y_s) or len(y_t) != len(attacks):
        raise ValueError("Length mismatch between y_true, y_scores, and attack_types")

    unique_attacks = sorted(list(set(attacks)))
    breakdown: dict[str, dict[str, Any]] = {}

    for att in unique_attacks:
        mask = [a == att for a in attacks]
        att_y_t = y_t[mask]
        att_y_s = y_s[mask]
        n_samples = len(att_y_s)

        is_bonafide_group = bool(att == "-" or (len(att_y_t) > 0 and att_y_t[0] == 1))

        if is_bonafide_group:
            # For genuine speech: accepted (TP) vs rejected (FN)
            accepted = int(np.sum(att_y_s > threshold))
            rejected = int(np.sum(att_y_s <= threshold))
            acceptance_rate = float(accepted / max(1, n_samples)) * 100.0
            breakdown[att] = {
                "category": "bonafide",
                "sample_count": n_samples,
                "correct_count": accepted,
                "error_count": rejected,
                "accuracy_pct": round(acceptance_rate, 2),
                "error_type": "False Rejection (FRR)",
                "mean_raw_score": round(float(np.mean(att_y_s)), 4),
                "min_raw_score": round(float(np.min(att_y_s)), 4),
                "max_raw_score": round(float(np.max(att_y_s)), 4),
            }
        else:
            # For spoof attacks: correctly detected as spoof (TN: score <= threshold)
            # vs missed/false alarm (FP: score > threshold, spoof accepted as real)
            detected = int(np.sum(att_y_s <= threshold))
            missed = int(np.sum(att_y_s > threshold))
            detection_rate = float(detected / max(1, n_samples)) * 100.0
            miss_rate = float(missed / max(1, n_samples)) * 100.0
            breakdown[att] = {
                "category": "spoof",
                "sample_count": n_samples,
                "detected_count": detected,
                "missed_count": missed,
                "detection_rate_pct": round(detection_rate, 2),
                "miss_rate_pct": round(miss_rate, 2),
                "error_type": "False Alarm (FAR)",
                "mean_raw_score": round(float(np.mean(att_y_s)), 4),
                "min_raw_score": round(float(np.min(att_y_s)), 4),
                "max_raw_score": round(float(np.max(att_y_s)), 4),
            }

    return breakdown


def format_attack_breakdown_markdown(breakdown: dict[str, dict[str, Any]]) -> str:
    """Format attack breakdown dictionary as a Markdown table."""
    lines = [
        "| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for att, data in breakdown.items():
        cat = data["category"]
        count = data["sample_count"]
        mean_s = f"{data['mean_raw_score']:+.4f}"
        s_range = f"[{data['min_raw_score']:+.4f}, {data['max_raw_score']:+.4f}]"

        if cat == "bonafide":
            acc = data["accuracy_pct"]
            lines.append(
                f"| **{att} (Genuine)** | Bona Fide | {count} | {data['correct_count']} | {data['error_count']} | **{acc:.1f}%** | `{mean_s}` | `{s_range}` |"
            )
        else:
            det = data["detection_rate_pct"]
            lines.append(
                f"| **{att}** | Spoof Attack | {count} | {data['detected_count']} | {data['missed_count']} | **{det:.1f}%** | `{mean_s}` | `{s_range}` |"
            )

    return "\n".join(lines)


def calculate_breakdown_by_speaker(
    y_true: np.ndarray | list[int],
    y_scores: np.ndarray | list[float],
    speaker_ids: list[str],
    threshold: float = -1.0625009,
) -> dict[str, dict[str, Any]]:
    """Compute performance metrics broken down by individual speaker ID."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)
    speakers = list(speaker_ids)

    if len(y_t) != len(y_s) or len(y_t) != len(speakers):
        raise ValueError("Length mismatch between y_true, y_scores, and speaker_ids")

    unique_speakers = sorted(list(set(speakers)))
    breakdown: dict[str, dict[str, Any]] = {}

    for spk in unique_speakers:
        mask = [s == spk for s in speakers]
        s_y_t = y_t[mask]
        s_y_s = y_s[mask]
        n_samples = len(s_y_s)

        n_bonafide = int(np.sum(s_y_t == 1))
        n_spoof = int(np.sum(s_y_t == 0))

        # Predictions: score > threshold -> bonafide (1), else spoof (0)
        s_pred = (s_y_s > threshold).astype(int)
        correct = int(np.sum(s_pred == s_y_t))
        accuracy = float(correct / max(1, n_samples)) * 100.0

        # Bona fide correct (TP) and missed (FN)
        tp = int(np.sum((s_y_t == 1) & (s_pred == 1)))
        fn = int(np.sum((s_y_t == 1) & (s_pred == 0)))

        # Spoof detected (TN) and missed (FP)
        tn = int(np.sum((s_y_t == 0) & (s_pred == 0)))
        fp = int(np.sum((s_y_t == 0) & (s_pred == 1)))

        breakdown[spk] = {
            "sample_count": n_samples,
            "bonafide_count": n_bonafide,
            "spoof_count": n_spoof,
            "correct_count": correct,
            "accuracy_pct": round(accuracy, 2),
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
            "mean_raw_score": round(float(np.mean(s_y_s)), 4),
            "min_raw_score": round(float(np.min(s_y_s)), 4),
            "max_raw_score": round(float(np.max(s_y_s)), 4),
        }

    return breakdown


def format_speaker_breakdown_markdown(breakdown: dict[str, dict[str, Any]]) -> str:
    """Format speaker breakdown dictionary as a Markdown table."""
    lines = [
        "| Speaker ID | Samples | Bona Fide (TP / Total) | Spoof Detected (TN / Total) | Accuracy (%) | Mean Score | Score Range [Min, Max] |",
        "|---|---|---|---|---|---|---|",
    ]

    for spk, data in breakdown.items():
        n = data["sample_count"]
        bf_str = f"{data['true_positives']} / {data['bonafide_count']}" if data['bonafide_count'] > 0 else "0 / 0"
        spf_str = f"{data['true_negatives']} / {data['spoof_count']}" if data['spoof_count'] > 0 else "0 / 0"
        acc = data["accuracy_pct"]
        mean_s = f"{data['mean_raw_score']:+.4f}"
        s_range = f"[{data['min_raw_score']:+.4f}, {data['max_raw_score']:+.4f}]"

        lines.append(
            f"| **{spk}** | {n} | {bf_str} | {spf_str} | **{acc:.1f}%** | `{mean_s}` | `{s_range}` |"
        )

    return "\n".join(lines)


def calculate_score_distributions(
    y_true: np.ndarray | list[int],
    y_scores: np.ndarray | list[float],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for raw scores across bona fide and spoof cohorts."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)

    bf_scores = y_s[y_t == 1]
    spf_scores = y_s[y_t == 0]

    def _stats(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {
                "count": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0,
            }
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
        }

    bf_stat = _stats(bf_scores)
    spf_stat = _stats(spf_scores)
    all_stat = _stats(y_s)

    sep_margin = round(bf_stat["min"] - spf_stat["max"], 4) if bf_stat["count"] > 0 and spf_stat["count"] > 0 else 0.0

    return {
        "bonafide": bf_stat,
        "spoof": spf_stat,
        "overall": all_stat,
        "separation_margin": sep_margin,
    }


def format_score_distributions_markdown(dist_stats: dict[str, Any]) -> str:
    """Format score distribution statistics into a Markdown table."""
    lines = [
        "| Cohort | Utterances | Mean ± Std | Median | 25th – 75th Percentile | Range [Min, Max] |",
        "|---|---|---|---|---|---|",
    ]

    for key, label in [("bonafide", "**Bona Fide (Genuine)**"), ("spoof", "**Spoof (Deepfake)**"), ("overall", "**Overall Dataset**")]:
        s = dist_stats[key]
        p_str = f"[{s['p25']:+.4f}, {s['p75']:+.4f}]"
        r_str = f"[{s['min']:+.4f}, {s['max']:+.4f}]"
        lines.append(
            f"| {label} | {s['count']} | `{s['mean']:+.4f} ± {s['std']:.4f}` | `{s['median']:+.4f}` | `{p_str}` | `{r_str}` |"
        )

    sep = dist_stats.get("separation_margin", 0.0)
    lines.append("")
    lines.append(f"**Separation Margin (Min Bona Fide − Max Spoof):** `{sep:+.4f}` logits")
    return "\n".join(lines)


def calculate_calibrated_prob_distributions(
    y_true: np.ndarray | list[int],
    prob_bonafide: np.ndarray | list[float],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for calibrated posterior probabilities."""
    y_t = np.asarray(y_true, dtype=int)
    p_bf = np.asarray(prob_bonafide, dtype=float)
    p_spf = 1.0 - p_bf

    bf_p_bf = p_bf[y_t == 1]
    spf_p_spf = p_spf[y_t == 0]

    def _p_stats(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    return {
        "bonafide_prob_distribution": _p_stats(bf_p_bf),
        "spoof_prob_distribution": _p_stats(spf_p_spf),
    }


def format_calibrated_prob_distributions_markdown(prob_stats: dict[str, Any]) -> str:
    """Format calibrated probability summary into a Markdown table."""
    lines = [
        "| Cohort | Metric Evaluated | Utterances | Mean ± Std | Median | Range [Min, Max] |",
        "|---|---|---|---|---|---|",
    ]

    bf = prob_stats["bonafide_prob_distribution"]
    lines.append(
        f"| **Bona Fide (Genuine)** | Calibrated P(Bona Fide) | {bf['count']} | `{bf['mean']:.4f} ± {bf['std']:.4f}` | `{bf['median']:.4f}` | `[{bf['min']:.4f}, {bf['max']:.4f}]` |"
    )

    spf = prob_stats["spoof_prob_distribution"]
    lines.append(
        f"| **Spoof (Deepfake)** | Calibrated P(Spoof) | {spf['count']} | `{spf['mean']:.4f} ± {spf['std']:.4f}` | `{spf['median']:.4f}` | `[{spf['min']:.4f}, {spf['max']:.4f}]` |"
    )

    return "\n".join(lines)
