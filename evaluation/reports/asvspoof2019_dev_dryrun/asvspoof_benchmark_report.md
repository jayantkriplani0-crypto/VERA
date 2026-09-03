# VERA Layer 1: ASVspoof Benchmark Evaluation Report
**Target Model:** `lab260/Spectra-AASIST3` (Official Pretrained Weights)  
**Protocol Source:** `ASVspoof2019.LA.cm.dev.trl.txt`  
**Audio Source Directory:** `flac`  
**Date:** September 2026  

---

## 1. Executive Summary

| Benchmark Metric | Result | Target / Standard Interpretation |
|---|---|---|
| **Equal Error Rate (EER)** | **0.00%** | Primary ASVspoof competition metric (lower is better) |
| **EER Threshold** | `-1.062501` | Threshold where False Alarm Rate == False Rejection Rate |
| **ROC-AUC** | **1.0000** | Area under ROC curve (1.0 = Perfect separation) |
| **Operating Threshold** | `-1.062501` | Official model cutoff (higher = bona fide) |
| **Classification Accuracy** | **100.00%** | Overall sample accuracy at operating threshold |
| **Precision (Bona Fide)** | 100.00% | True bona fide / total classified bona fide |
| **Recall (Bona Fide)** | 100.00% | True bona fide / total actual bona fide |
| **F1-Score** | 1.0000 | Harmonic mean of Precision and Recall |
| **False Alarm Rate (FAR / FPR)** | **0.00%** | Spoofed voices incorrectly accepted as genuine |
| **False Rejection Rate (FRR / FNR)** | **0.00%** | Genuine human voices incorrectly flagged as spoof |

---

## 2. Dataset Composition & Throughput

- **Total Utterances Evaluated:** 100
- **Bona Fide (Genuine) Utterances:** 100 (100.0%)
- **Spoof (Deepfake) Utterances:** 0 (0.0%)
- **Unique Speakers:** 1
- **Average Inference Latency:** 803.70 ms / sample
- **Processing Throughput:** 1.24 utterances / sec

---

## 3. Confusion Matrix

| Actual Class \ Predicted Class | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **0** | FP (FAR) = **0** |
| **Actual Bona Fide (1)** | FN (FRR) = **0** | TP = **100** |

---

## 4. Per-Attack & Category Breakdown

| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 100 | 100 | 0 | **100.0%** | `+13.5827` | `[+13.0732, +13.7845]` |

> [!NOTE]
> Detection Rate indicates percentage of spoof attacks successfully caught (score <= threshold). Miss Rate (FAR) indicates percentage of spoofs that bypassed detection.