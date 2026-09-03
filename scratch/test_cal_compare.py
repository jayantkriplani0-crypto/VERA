import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from calibration.calibrate import evaluate_calibration_candidates

df_preds = pd.read_csv('evaluation/reports/calibration_preparation/asvspoof_dev_700_predictions.csv')
fit_df = df_preds[df_preds['split'] == 'cal_fit']
val_df = df_preds[df_preds['split'] == 'cal_val']

fit_scores = fit_df['raw_bonafide_score'].values
fit_labels = fit_df['ground_truth_binary'].values

val_scores = val_df['raw_bonafide_score'].values
val_labels = val_df['ground_truth_binary'].values

comp = evaluate_calibration_candidates(fit_scores, fit_labels)
print("=== CALIBRATION CANDIDATE COMPARISON ON CAL_FIT (420 SAMPLES) ===")
for method, info in comp.items():
    print(f"\nMethod: {method}")
    print(f"  Brier Score: {info['brier_score']:.6f}")
    print(f"  ECE        : {info['ece']:.6f}")
    print(f"  Log Loss   : {info['log_loss']:.6f}")
    if 'params' in info:
        print(f"  Params     : {info['params']}")
