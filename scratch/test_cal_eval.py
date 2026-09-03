import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from calibration.calibrate import evaluate_calibration_candidates, compute_ece, PlattCalibrator
from evaluation.metrics import calculate_metrics

df_preds = pd.read_csv('evaluation/reports/calibration_preparation/asvspoof_dev_700_predictions.csv')
fit_df = df_preds[df_preds['split'] == 'cal_fit']
val_df = df_preds[df_preds['split'] == 'cal_val']

fit_scores = fit_df['raw_bonafide_score'].values
fit_labels = fit_df['ground_truth_binary'].values

val_scores = val_df['raw_bonafide_score'].values
val_labels = val_df['ground_truth_binary'].values

comp = evaluate_calibration_candidates(fit_scores, fit_labels)
platt: PlattCalibrator = comp["PlattScaling"]["calibrator"]

# Threshold at P=0.50
cal_threshold = -platt.b / platt.w
print(f"Platt Calibrated Threshold: {cal_threshold:+.6f}")

print("\n=== CAL_FIT EVALUATION (420 SAMPLES) ===")
fit_probs = platt.predict_proba(fit_scores)
fit_brier = brier_score_loss(fit_labels, fit_probs)
fit_ece = compute_ece(fit_labels, fit_probs)
fit_m = calculate_metrics(fit_labels, fit_scores, fit_df['speaker_id'].tolist(), threshold=cal_threshold)
print(f"ROC-AUC  : {fit_m.roc_auc:.4f}")
print(f"EER      : {fit_m.eer_percent:.2f}% (Threshold: {fit_m.eer_threshold:+.6f})")
print(f"Brier    : {fit_brier:.6f}")
print(f"ECE      : {fit_ece:.6f}")
print(f"Accuracy : {fit_m.accuracy * 100:.2f}%")
print(f"FRR      : {fit_m.false_negative_rate * 100:.2f}%")
print(f"FAR      : {fit_m.false_positive_rate * 100:.2f}%")
print(f"Precision: {fit_m.precision * 100:.2f}%")
print(f"Recall   : {fit_m.recall * 100:.2f}%")
print(f"F1       : {fit_m.f1_score:.4f}")

print("\n=== CAL_VAL EVALUATION (280 UNSEEN SAMPLES) ===")
val_probs = platt.predict_proba(val_scores)
val_brier = brier_score_loss(val_labels, val_probs)
val_ece = compute_ece(val_labels, val_probs)
val_m = calculate_metrics(val_labels, val_scores, val_df['speaker_id'].tolist(), threshold=cal_threshold)
print(f"ROC-AUC  : {val_m.roc_auc:.4f}")
print(f"EER      : {val_m.eer_percent:.2f}% (Threshold: {val_m.eer_threshold:+.6f})")
print(f"Brier    : {val_brier:.6f}")
print(f"ECE      : {val_ece:.6f}")
print(f"Accuracy : {val_m.accuracy * 100:.2f}%")
print(f"FRR      : {val_m.false_negative_rate * 100:.2f}%")
print(f"FAR      : {val_m.false_positive_rate * 100:.2f}%")
print(f"Precision: {val_m.precision * 100:.2f}%")
print(f"Recall   : {val_m.recall * 100:.2f}%")
print(f"F1       : {val_m.f1_score:.4f}")
