# Calibration & Evaluation Dataset Audit
**Project:** VERA SIH 26104 MVP  
**Target Model:** `lab260/Spectra-AASIST3` (Fixed Pretrained Baseline)  
**Audit Date:** September 2026  

---

## 1. Executive Summary: Why Were Only 4 Validation & 4 Test Samples Used?

A rigorous audit of the workspace reveals the exact structural root cause of the previous 4+4 sample split:

1. **Origin of Audio Files:** The dataset in `evaluation/benchmark_data/audio/` is an initial **synthetic smoke-test / prototype corpus** consisting of 12 audio files across 3 synthetic acoustic speakers (`SPK_001`, `SPK_002`, `SPK_003`).
2. **Artificial 3-Way Partitioning:** The initial split logic assigned data using a standard 60/20/20 ratio across the 3 speakers: 1 speaker to `train` (`SPK_002`), 1 to `val` (`SPK_001`), and 1 to `test` (`SPK_003`).
3. **Idle Training Partition:** Because **zero model fine-tuning or weight training was occurring** (weights are fixed), the 4 samples in `train` sat completely idle and unused, leaving only 4 samples in validation and 4 in test.
4. **Unindexed Robustness Audio:** An additional cohort of 9 labeled spoof files created during robustness testing (`evaluation/benchmark_data/unseen_audio/`) was not included in the original `benchmark_manifest.csv`.

> [!IMPORTANT]
> **Conclusion of Root-Cause Analysis:** The previous 4+4 split was an **initial smoke-test artifact** combined with an inefficient 3-way split allocation. The local workspace genuinely contains only **21 labeled audio files** across 3 synthetic speakers. No large external corpus (such as ASVspoof 2019/2021) exists locally. Therefore, this calibration remains an **initial prototype / smoke-test calibration**, and claims of 'high general accuracy' cannot be supported until large-scale external benchmark datasets are ingested.

---

## 2. Inventory of Locally Available Labeled Data

- **Total Labeled Audio Files:** **21**
- **Unique Speakers:** **3** (`SPK_001, SPK_002, SPK_003`)
- **Bona Fide (Genuine) Samples:** **6** (28.6%)
- **Spoof (Synthetic / Deepfake) Samples:** **15** (71.4%)
- **Attack Types Represented (6):** `-, Neural-Vocoder-HiFiGAN, TTS-ChatTTS-Style, TTS-VITS, VC-DiffVC, VC-FreeVC-Style`
- **Languages:** English (`en`)

### Per-Speaker Breakdown:

| Speaker ID | Total Samples | Bona Fide | Spoof | Attack Types Present |
|---|---|---|---|---|
| `SPK_001` | **7** | 2 | 5 | `-, Neural-Vocoder-HiFiGAN, TTS-ChatTTS-Style, TTS-VITS, VC-DiffVC, VC-FreeVC-Style` |
| `SPK_002` | **7** | 2 | 5 | `-, Neural-Vocoder-HiFiGAN, TTS-ChatTTS-Style, TTS-VITS, VC-DiffVC, VC-FreeVC-Style` |
| `SPK_003` | **7** | 2 | 5 | `-, Neural-Vocoder-HiFiGAN, TTS-ChatTTS-Style, TTS-VITS, VC-DiffVC, VC-FreeVC-Style` |

---

## 3. Revised Speaker-Disjoint Split Architecture

Since no training is conducted on model weights, reserving samples for an unused `train` split is suboptimal. The 21 samples are reorganized into a strict 2-way speaker-disjoint partition:

| Partition | Speaker Count | Speaker IDs | Samples (Bona Fide / Spoof) | Pipeline Function |
|---|---|---|---|---|
| **Validation Split** | 2 | `SPK_001`, `SPK_002` | **14 samples** (4 real, 10 spoof) | **Fit candidate calibrators & select parameters** |
| **Held-Out Test Split** | 1 | `SPK_003` | **7 samples** (2 real, 5 spoof) | **Strict out-of-sample evaluation** |

### Speaker Disjointness Verification:
$$\text{speakers}(\text{validation}) \cap \text{speakers}(\text{test}) = \{\text{SPK\_001}, \text{SPK\_002}\} \cap \{\text{SPK\_003}\} = \emptyset$$
- **Speaker Leakage:** 0 speakers leaked.
- **Test Set Contamination:** Zero test labels or test logits were seen by any calibration algorithm.

---

## 4. Calibration Candidate Comparison (Validation Split)

Evaluated on the expanded validation partition (14 samples):

| Method | Parameter Count | Brier Score | ECE | Validation Log-Loss | Feasibility & Robustness on N=14 | Selection Decision |
|---|---|---|---|---|---|---|
| **PlattScaling** | 2 (w, b) | `0.1818` | `0.0319` | `0.5148` | High (Regularized 2-parameter logistic curve avoids overfitting) | ✅ **Selected Method** |
| **TemperatureScaling** | 1 (T) | `0.2209` | `0.2141` | `0.6289` | Low (Fixed intercept b=0 cannot handle non-zero logit threshold) | Rejected |
| **IsotonicRegression** | Non-parametric | `0.1429` | `0.0000` | `0.3961` | Poor (Non-parametric step function overfits on small N) | Rejected |

### Method Justification:
1. **Isotonic Regression Rejection:** With only 14 validation samples, Isotonic Regression produces degenerate piecewise-constant steps with zero-gradient plateaus, failing to generalize to unseen logits.
2. **Temperature Scaling Rejection:** Temperature scaling assumes the logit zero-point corresponds to P=0.5. In `Spectra-AASIST3`, the operational threshold is negative (-1.0625), necessitating a non-zero intercept parameter.
3. **Platt Scaling Selection:** The 2-parameter logistic model provides a smooth, monotonic probability curve (w = +0.3623, b = -1.1366), preserving ROC-AUC and EER ranking while minimizing validation Brier loss.

---

## 5. Held-Out Test Evaluation Summary

- **Held-Out Test Set:** 7 samples (Speaker `SPK_003`, 2 bona fide, 5 spoof across 5 attack types)
- **Test Brier Score:** `0.2070`
- **Test Expected Calibration Error (ECE):** `0.1088`
- **Test Classification Accuracy:** `71.4%` (5 / 7 correct)
- **Test EER:** `20.00%`
- **Test ROC-AUC:** `0.6000`

---

## 6. Honest Limitations & Roadmap to Production

1. **Prototype-Scale Sample Size:** With N=21 total samples across 3 synthetic acoustic speakers, this calibration demonstrates mathematical and software validity, but **cannot be considered production-grade**.
2. **No Real Human Acoustic Diversity:** Current samples are synthetic sinusoidal/formant signals without conversational speech variability, background acoustic diversity, or regional accents.
3. **Recommended Next Step:** Ingest the official **ASVspoof 2019 Logical Access (LA)** evaluation corpus (25,380 evaluation trials across 48 speakers) to establish a statistically powered benchmark.