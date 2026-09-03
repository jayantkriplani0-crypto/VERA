# VERA Layer 1: ASVspoof Benchmark Evaluation Report
**Target Model:** `lab260/Spectra-AASIST3` (Official Pretrained Weights — Frozen)  
**Protocol Source:** `ASVspoof2019.LA.cm.dev.trl.txt`  
**Audio Source Directory:** `flac`  
**Date:** September 2026  

---

## 1. Executive Summary

| Benchmark Metric | Result | Target / Standard Interpretation |
|---|---|---|
| **Equal Error Rate (EER)** | **0.00%** | Primary ASVspoof competition metric (lower is better) |
| **EER Operating Threshold** | `+3.299379` | Threshold where False Alarm Rate == False Rejection Rate |
| **ROC-AUC** | **1.0000** | Area under ROC curve (1.0 = Perfect separation) |
| **Operating Threshold** | `-1.062501` | Official model cutoff (higher = bona fide) |
| **Classification Accuracy** | **73.82%** | Overall sample accuracy at operating threshold |
| **Precision (Bona Fide)** | 68.10% | True bona fide / total classified bona fide |
| **Recall (Bona Fide)** | 100.00% | True bona fide / total actual bona fide |
| **F1-Score** | 0.8102 | Harmonic mean of Precision and Recall |
| **False Alarm Rate (FAR / FPR)** | **59.33%** | Spoofed voices incorrectly accepted as genuine |
| **False Rejection Rate (FRR / FNR)** | **0.00%** | Genuine human voices incorrectly flagged as spoof |

---

## 2. Protocol Coverage & Audit

- **Total Trials in Protocol:** 24,844
- **Audio Trials Evaluated:** 340 (1.37% coverage of protocol)
- **Missing Audio Trials:** 24,504
- **Bona Fide (Genuine) Utterances:** 190 (55.9%)
- **Spoof (Deepfake) Utterances:** 150 (44.1%)
- **Unique Speakers Evaluated:** 10
- **Average Inference Latency:** 821.01 ms / sample
- **Processing Throughput:** 1.22 utterances / sec

---

## 3. Score Distribution Statistics (Bona Fide vs. Spoof)

| Cohort | Utterances | Mean ± Std | Median | 25th – 75th Percentile | Range [Min, Max] |
|---|---|---|---|---|---|
| **Bona Fide (Genuine)** | 190 | `+13.4642 ± 0.8294` | `+13.5885` | `[+13.4779, +13.6676]` | `[+3.2994, +13.8569]` |
| **Spoof (Deepfake)** | 150 | `-1.0318 ± 0.4188` | `-0.9664` | `[-1.2457, -0.7571]` | `[-2.5266, -0.2571]` |
| **Overall Dataset** | 340 | `+7.0689 ± 7.2297` | `+13.3211` | `[-0.9395, +13.5991]` | `[-2.5266, +13.8569]` |

**Separation Margin (Min Bona Fide − Max Spoof):** `+3.5565` logits

---

## 4. Confusion Matrix

| Actual Class \ Predicted Class | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **61** | FP (FAR) = **89** |
| **Actual Bona Fide (1)** | FN (FRR) = **0** | TP = **190** |

---

## 5. Per-Attack & Category Breakdown (A01–A06)

| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 190 | 190 | 0 | **100.0%** | `+13.4642` | `[+3.2994, +13.8569]` |
| **A01** | Spoof Attack | 100 | 45 | 55 | **45.0%** | `-1.0838` | `[-2.5266, -0.5010]` |
| **A02** | Spoof Attack | 10 | 0 | 10 | **0.0%** | `-0.3817` | `[-0.6134, -0.2571]` |
| **A03** | Spoof Attack | 10 | 1 | 9 | **10.0%** | `-0.7859` | `[-1.1292, -0.5093]` |
| **A04** | Spoof Attack | 10 | 1 | 9 | **10.0%** | `-0.8324` | `[-2.3539, -0.3887]` |
| **A05** | Spoof Attack | 10 | 5 | 5 | **50.0%** | `-1.0719` | `[-1.6905, -0.5355]` |
| **A06** | Spoof Attack | 10 | 9 | 1 | **90.0%** | `-1.5677` | `[-2.3036, -0.6596]` |

---

## 6. Per-Speaker Breakdown

| Speaker ID | Samples | Bona Fide (TP / Total) | Spoof Detected (TN / Total) | Accuracy (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|
| **LA_0069** | 160 | 100 / 100 | 20 / 60 | **75.0%** | `+8.1317` | `[-2.3539, +13.7845]` |
| **LA_0070** | 20 | 10 / 10 | 5 / 10 | **75.0%** | `+5.9930` | `[-1.4382, +13.7267]` |
| **LA_0071** | 20 | 10 / 10 | 4 / 10 | **70.0%** | `+6.0441` | `[-1.7827, +13.6929]` |
| **LA_0072** | 20 | 10 / 10 | 3 / 10 | **65.0%** | `+6.3890` | `[-1.3676, +13.7407]` |
| **LA_0073** | 20 | 10 / 10 | 3 / 10 | **65.0%** | `+5.5984` | `[-1.6721, +13.7030]` |
| **LA_0074** | 20 | 10 / 10 | 3 / 10 | **65.0%** | `+6.2101` | `[-1.2828, +13.7792]` |
| **LA_0075** | 20 | 10 / 10 | 6 / 10 | **80.0%** | `+6.2798` | `[-1.5029, +13.8560]` |
| **LA_0076** | 20 | 10 / 10 | 7 / 10 | **85.0%** | `+6.0408` | `[-2.5266, +13.6454]` |
| **LA_0077** | 20 | 10 / 10 | 5 / 10 | **75.0%** | `+6.2037` | `[-1.5683, +13.7137]` |
| **LA_0078** | 20 | 10 / 10 | 5 / 10 | **75.0%** | `+6.3585` | `[-1.4026, +13.8569]` |

> [!NOTE]
> All evaluations were conducted using the frozen pretrained `lab260/Spectra-AASIST3` weights without fine-tuning or threshold re-fitting.