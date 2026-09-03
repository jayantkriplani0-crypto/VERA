# ASVspoof 2019 LA: Calibration Preparation Cohort Report (700 Samples)
**Target Model:** `lab260/Spectra-AASIST3` (Official Pretrained Checkpoint — Frozen & Unchanged)  
**Protocol:** `ASVspoof2019.LA.cm.dev.trl.txt`  
**Date:** September 2026  

---

## 1. Executive Summary & Split Comparison

| Metric | Overall Cohort (700) | Calibration Fit (`cal_fit`, 420) | Held-Out Validation (`cal_val`, 280) | Reference / Standard Interpretation |
|---|---|---|---|---|
| **Speakers** | 10 Speakers | 6 Speakers (`LA_0069`–`LA_0074`) | 4 Unseen Speakers (`LA_0075`–`LA_0078`) | Disjoint speaker partitions |
| **Bona Fide Utterances** | 280 (40.0%) | 168 (40.0%) | 112 (40.0%) | Ground truth genuine human speech |
| **Spoof Utterances** | 420 (60.0%) | 252 (60.0%) | 168 (60.0%) | Deepfake speech across `A01`–`A06` |
| **Equal Error Rate (EER)** | **0.36%** | **0.60%** | **0.00%** | Primary ASVspoof discrimination metric |
| **ROC-AUC** | **0.9957** | **0.9928** | **1.0000** | 1.0 = Perfect separation |
| **EER Operating Threshold** | `+2.695514` | `+2.695514` | `+11.562364` | Threshold where FAR == FRR |
| **Operating Threshold** | `-1.062501` | `-1.062501` | `-1.062501` | Fixed official model cutoff |
| **Accuracy @ Cutoff** | **63.43%** | **61.43%** | **66.43%** | Classification accuracy at fixed cutoff |
| **False Alarm Rate (FAR)** | **60.71%** | **63.89%** | **55.95%** | Spoofs incorrectly accepted as genuine |
| **False Rejection Rate (FRR)** | **0.36%** | **0.60%** | **0.00%** | Genuine human voices incorrectly rejected |

---

## 2. Score Distribution Statistics (Raw Frozen Logits)

### Overall Cohort (700 Samples)
| Cohort | Utterances | Mean ± Std | Median | 25th – 75th Percentile | Range [Min, Max] |
|---|---|---|---|---|---|
| **Bona Fide (Genuine)** | 280 | `+13.2266 ± 1.7248` | `+13.5637` | `[+13.3662, +13.6579]` | `[-3.6785, +13.8706]` |
| **Spoof (Deepfake)** | 420 | `-0.9959 ± 0.5368` | `-0.9403` | `[-1.2836, -0.6081]` | `[-4.1430, +0.0413]` |
| **Overall Dataset** | 700 | `+4.6931 ± 7.0647` | `-0.4941` | `[-1.0382, +13.4653]` | `[-4.1430, +13.8706]` |

**Separation Margin (Min Bona Fide − Max Spoof):** `-3.7198` logits

### Calibration Fit Split (`cal_fit`, 420 Samples)
| Cohort | Utterances | Mean ± Std | Median | 25th – 75th Percentile | Range [Min, Max] |
|---|---|---|---|---|---|
| **Bona Fide (Genuine)** | 168 | `+13.0132 ± 2.1862` | `+13.4931` | `[+13.3059, +13.6123]` | `[-3.6785, +13.8578]` |
| **Spoof (Deepfake)** | 252 | `-0.9841 ± 0.5469` | `-0.9260` | `[-1.2515, -0.6030]` | `[-4.1430, +0.0413]` |
| **Overall Dataset** | 420 | `+4.6148 ± 7.0081` | `-0.4944` | `[-1.0182, +13.3947]` | `[-4.1430, +13.8578]` |

**Separation Margin (Min Bona Fide − Max Spoof):** `-3.7198` logits

### Held-Out Validation Split (`cal_val`, 280 Samples)
| Cohort | Utterances | Mean ± Std | Median | 25th – 75th Percentile | Range [Min, Max] |
|---|---|---|---|---|---|
| **Bona Fide (Genuine)** | 112 | `+13.5465 ± 0.3125` | `+13.6310` | `[+13.4758, +13.7156]` | `[+11.5624, +13.8706]` |
| **Spoof (Deepfake)** | 168 | `-1.0135 ± 0.5207` | `-0.9698` | `[-1.3379, -0.6167]` | `[-2.9804, -0.1709]` |
| **Overall Dataset** | 280 | `+4.8105 ± 7.1471` | `-0.4860` | `[-1.0940, +13.5608]` | `[-2.9804, +13.8706]` |

**Separation Margin (Min Bona Fide − Max Spoof):** `+11.7333` logits

---

## 3. Confusion Matrices (at Fixed Official Cutoff `-1.062501`)

### A. Overall Cohort (700 Samples)
| Actual \ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **165** | FP = **255** |
| **Actual Bona Fide (1)** | FN = **1** | TP = **279** |

### B. Calibration Fit Split (420 Samples)
| Actual \ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **91** | FP = **161** |
| **Actual Bona Fide (1)** | FN = **1** | TP = **167** |

### C. Held-Out Validation Split (280 Samples)
| Actual \ Predicted | Predicted Spoof (0) | Predicted Bona Fide (1) |
|---|---|---|
| **Actual Spoof (0)** | TN = **74** | FP = **94** |
| **Actual Bona Fide (1)** | FN = **0** | TP = **112** |

---

## 4. Per-Attack Family Breakdown (`A01`–`A06`)

### Overall Cohort Breakdown
| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 280 | 279 | 1 | **99.6%** | `+13.2266` | `[-3.6785, +13.8706]` |
| **A01** | Spoof Attack | 70 | 34 | 36 | **48.6%** | `-1.0949` | `[-2.5051, -0.5226]` |
| **A02** | Spoof Attack | 70 | 1 | 69 | **1.4%** | `-0.3926` | `[-1.3589, -0.1516]` |
| **A03** | Spoof Attack | 70 | 12 | 58 | **17.1%** | `-0.8268` | `[-1.8889, -0.3771]` |
| **A04** | Spoof Attack | 70 | 33 | 37 | **47.1%** | `-1.0611` | `[-2.5627, +0.0413]` |
| **A05** | Spoof Attack | 70 | 40 | 30 | **57.1%** | `-1.2366` | `[-4.1430, -0.3202]` |
| **A06** | Spoof Attack | 70 | 45 | 25 | **64.3%** | `-1.3633` | `[-2.4796, -0.3545]` |

### Calibration Fit Breakdown
| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 168 | 167 | 1 | **99.4%** | `+13.0132` | `[-3.6785, +13.8578]` |
| **A01** | Spoof Attack | 42 | 18 | 24 | **42.9%** | `-1.0449` | `[-1.6721, -0.5226]` |
| **A02** | Spoof Attack | 42 | 1 | 41 | **2.4%** | `-0.4206` | `[-1.3589, -0.1516]` |
| **A03** | Spoof Attack | 42 | 6 | 36 | **14.3%** | `-0.8118` | `[-1.8484, -0.3771]` |
| **A04** | Spoof Attack | 42 | 17 | 25 | **40.5%** | `-1.0186` | `[-2.5627, +0.0413]` |
| **A05** | Spoof Attack | 42 | 22 | 20 | **52.4%** | `-1.2468` | `[-4.1430, -0.3202]` |
| **A06** | Spoof Attack | 42 | 27 | 15 | **64.3%** | `-1.3622` | `[-2.4545, -0.3545]` |

### Held-Out Validation Breakdown
| Attack / Source Category | Type | Samples | Correct / Detected | Errors / Missed | Rate (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|---|
| **- (Genuine)** | Bona Fide | 112 | 112 | 0 | **100.0%** | `+13.5465` | `[+11.5624, +13.8706]` |
| **A01** | Spoof Attack | 28 | 16 | 12 | **57.1%** | `-1.1700` | `[-2.5051, -0.5967]` |
| **A02** | Spoof Attack | 28 | 0 | 28 | **0.0%** | `-0.3507` | `[-0.6177, -0.1709]` |
| **A03** | Spoof Attack | 28 | 6 | 22 | **21.4%** | `-0.8493` | `[-1.8889, -0.4133]` |
| **A04** | Spoof Attack | 28 | 16 | 12 | **57.1%** | `-1.1249` | `[-2.0319, -0.4126]` |
| **A05** | Spoof Attack | 28 | 18 | 10 | **64.3%** | `-1.2212` | `[-2.9804, -0.3891]` |
| **A06** | Spoof Attack | 28 | 18 | 10 | **64.3%** | `-1.3650` | `[-2.4796, -0.5482]` |

---

## 5. Per-Speaker Breakdown (All 10 Core Speakers)

| Speaker ID | Samples | Bona Fide (TP / Total) | Spoof Detected (TN / Total) | Accuracy (%) | Mean Score | Score Range [Min, Max] |
|---|---|---|---|---|---|---|
| **LA_0069** | 70 | 28 / 28 | 14 / 42 | **60.0%** | `+4.8713` | `[-2.3036, +13.6863]` |
| **LA_0070** | 70 | 28 / 28 | 16 / 42 | **62.9%** | `+4.6697` | `[-2.4545, +13.7267]` |
| **LA_0071** | 70 | 27 / 28 | 19 / 42 | **65.7%** | `+3.8450` | `[-4.1430, +13.6929]` |
| **LA_0072** | 70 | 28 / 28 | 14 / 42 | **60.0%** | `+4.8522` | `[-2.2370, +13.8578]` |
| **LA_0073** | 70 | 28 / 28 | 16 / 42 | **62.9%** | `+4.5288` | `[-2.7650, +13.7093]` |
| **LA_0074** | 70 | 28 / 28 | 12 / 42 | **57.1%** | `+4.9219` | `[-1.7408, +13.7792]` |
| **LA_0075** | 70 | 28 / 28 | 15 / 42 | **61.4%** | `+4.9088` | `[-2.4796, +13.8706]` |
| **LA_0076** | 70 | 28 / 28 | 23 / 42 | **72.9%** | `+4.7383` | `[-2.5051, +13.7471]` |
| **LA_0077** | 70 | 28 / 28 | 21 / 42 | **70.0%** | `+4.7699` | `[-2.4039, +13.8175]` |
| **LA_0078** | 70 | 28 / 28 | 15 / 42 | **61.4%** | `+4.8250` | `[-2.9804, +13.8569]` |

---

## 6. Throughput & Execution Verification

- **Total Samples Evaluated:** 700
- **Total Inference Time:** 561.47 seconds (9.36 minutes)
- **Average Inference Latency:** 802.10 ms / sample
- **Processing Throughput:** 1.25 utterances / sec
- **Hardware Execution:** CPU (`torch.cuda.is_available() == False`)

> [!IMPORTANT]
> **Calibration Parameters ($w, b$) Status:** Untrained and uncalibrated during this phase. Only frozen raw logits were extracted.

> [!IMPORTANT]
> **Quarantine Verification:** The ASVspoof 2019 LA evaluation archive (`LA_eval.zip`) was **not** accessed or downloaded.