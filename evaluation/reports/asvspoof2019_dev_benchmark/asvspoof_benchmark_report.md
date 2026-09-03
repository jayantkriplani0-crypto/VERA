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
| **EER Threshold** | `+13.073249` | Threshold where False Alarm Rate == False Rejection Rate |
| **ROC-AUC** | **1.0000** | Area under ROC curve (1.0 = Perfect separation) |
| **Operating Threshold** | `-1.062501` | Official model cutoff (higher = bona fide) |
| **Classification Accuracy** | **75.00%** | Overall sample accuracy at operating threshold |
| **Precision (Bona Fide)** | 71.43% | True bona fide / total classified bona fide |
| **Recall (Bona Fide)** | 100.00% | True bona fide / total actual bona fide |
| **F1-Score** | 0.8333 | Harmonic mean of Precision and Recall |
| **False Alarm Rate (FAR / FPR)** | **66.67%** | Spoofed voices incorrectly accepted as genuine |
| **False Rejection Rate (FRR / FNR)** | **0.00%** | Genuine human voices incorrectly flagged as spoof |

---

## 2. Dataset Composition & Throughput

- **Total Utterances Evaluated:** 160
- **Bona Fide (Genuine) Utterances:** 100 (62.5%)
- **Spoof (Deepfake) Utterances:** 60 (37.5%)
- **Unique Speakers:** 1
- **Average Inference Latency:** 804.23 ms / sample
- **Processing Throughput:** 1.24 utterances / sec

---

## 3. Confusion Matrix

| Actual Class \ Predicted Class | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **20** | FP (FAR) = **40** |
| **Actual Bona Fide (1)** | FN (FRR) = **0** | TP = **100** |

---

## 4. Per-Attack & Category Breakdown

| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 100 | 100 | 0 | **100.0%** | `+13.5827` | `[+13.0732, +13.7845]` |
| **A01** | Spoof Attack | 10 | 4 | 6 | **40.0%** | `-1.0794` | `[-1.3952, -0.9164]` |
| **A02** | Spoof Attack | 10 | 0 | 10 | **0.0%** | `-0.3817` | `[-0.6134, -0.2571]` |
| **A03** | Spoof Attack | 10 | 1 | 9 | **10.0%** | `-0.7859` | `[-1.1292, -0.5093]` |
| **A04** | Spoof Attack | 10 | 1 | 9 | **10.0%** | `-0.8324` | `[-2.3539, -0.3887]` |
| **A05** | Spoof Attack | 10 | 5 | 5 | **50.0%** | `-1.0719` | `[-1.6905, -0.5355]` |
| **A06** | Spoof Attack | 10 | 9 | 1 | **90.0%** | `-1.5677` | `[-2.3036, -0.6596]` |

> [!NOTE]
> Detection Rate indicates percentage of spoof attacks successfully caught (score <= threshold). Miss Rate (FAR) indicates percentage of spoofs that bypassed detection.