# Layer 1 (Voice Authenticity) Calibration Report (v2 - Expanded Audit)
**Model:** `lab260/Spectra-AASIST3` (Frozen Pretrained Weights)  
**Audit Status:** Prototype Calibration on Expanded Local Dataset (N=21)  
**Date:** September 2026  

---

## 1. Dataset Partitioning Discipline

| Split | Speaker Count | Speaker IDs | Sample Count | Bona Fide | Spoof | Role |
|---|---|---|---|---|---|---|
| **Validation Split** | 2 | `SPK_001`, `SPK_002` | **14** | 4 | 10 | **Calibration fitting & model selection** |
| **Held-Out Test Split** | 1 | `SPK_003` | **7** | 2 | 5 | **Untouched out-of-sample evaluation** |

> [!IMPORTANT]
> Speaker disjointness assertion: Zero speaker leakage occurred between validation and held-out test sets.

---

## 2. Calibration Formulation & Score Directionality

$$\hat{P}(\text{Bona Fide} \mid s) = \sigma(w \cdot s + b) = \frac{1}{1 + \exp(-(+0.3623 \cdot s + -1.1366))}$$

$$\hat{P}(\text{Spoof} \mid s) = 1.0 - \hat{P}(\text{Bona Fide} \mid s)$$

$$\mathbf{calibrated\_spoof\_signal} = \hat{P}(\text{Spoof} \mid s) \times 100.0 \quad \in [0.0, 100.0]$$

$$\mathbf{voice\_integrity\_score} = \hat{P}(\text{Bona Fide} \mid s) \times 100.0 = 100.0 - \text{calibrated\_spoof\_signal}$$

$$\mathbf{decision\_confidence} = 2.0 \times |\hat{P}(\text{Spoof} \mid s) - 0.5| \quad \in [0.0, 1.0]$$

### Exact Score Direction:
- **Higher `calibrated_spoof_signal`** $\implies$ **More suspicious synthetic / deepfake speech** (100.0 = Maximum Spoof).
- **Higher `raw_bonafide_score`** $\implies$ **Lower `calibrated_spoof_signal`** (0.0 = Maximum Genuine).
- **Operational Decision Boundary:** `calibrated_spoof_signal >= 50.0` (corresponds to raw logit cutoff `+3.1373`).

---

## 3. Held-Out Test Set Performance (Speaker SPK_003)

### A. Metric Comparison: Raw Logits vs. Calibrated Output
| Evaluation Metric | Raw Logit Baseline (Cutoff `-1.0625`) | Calibrated Spoof Cutoff (`50.0 / 100`) | Calibration Impact |
|---|---|---|---|
| **Equal Error Rate (EER)** | **20.00%** | **20.00%** | Preserved rank order |
| **ROC-AUC** | **0.6000** | **0.6000** | Exact preservation |
| **Accuracy** | 57.1% | 71.4% | Consistent |
| **Precision (Bona Fide)** | 40.0% | 0.0% | Preserved |
| **Recall (Bona Fide)** | 100.0% | 0.0% | Preserved |
| **F1-Score** | 0.5714 | 0.0000 | Preserved |
| **Brier Score (Probability Error)** | N/A (Uncalibrated) | **0.2070** | Well-calibrated |
| **Expected Calibration Error (ECE)** | N/A (Uncalibrated) | **0.1088** | Minimal bin deviation |

### B. Per-Sample Held-Out Test Results
| Audio Filename | Attack Type | Ground Truth | Raw Bona-Fide Logit | Calibrated Spoof Signal (0–100) | Voice Integrity (0–100) | Confidence | Decision |
|---|---|---|---|---|---|---|---|
| `SPK_003_bonafide_1.wav` | `-` | `bonafide` | `-0.1275` | **76.5 / 100** | 23.5 / 100 | `0.53` | `spoof` |
| `SPK_003_bonafide_3.wav` | `-` | `bonafide` | `-0.1275` | **76.5 / 100** | 23.5 / 100 | `0.53` | `spoof` |
| `SPK_003_spoof_2.wav` | `TTS-VITS` | `spoof` | `-4.8179` | **94.7 / 100** | 5.3 / 100 | `0.89` | `spoof` |
| `SPK_003_spoof_4.wav` | `VC-DiffVC` | `spoof` | `-4.9026` | **94.8 / 100** | 5.2 / 100 | `0.90` | `spoof` |
| `SPK_003_unseen_spoof_1_TTS-ChatTTS-Style.wav` | `TTS-ChatTTS-Style` | `spoof` | `+1.9473` | **60.6 / 100** | 39.4 / 100 | `0.21` | `spoof` |
| `SPK_003_unseen_spoof_2_VC-FreeVC-Style.wav` | `VC-FreeVC-Style` | `spoof` | `-0.4446` | **78.5 / 100** | 21.5 / 100 | `0.57` | `spoof` |
| `SPK_003_unseen_spoof_3_Neural-Vocoder-HiFiGAN.wav` | `Neural-Vocoder-HiFiGAN` | `spoof` | `+0.3282` | **73.5 / 100** | 26.6 / 100 | `0.47` | `spoof` |

---

## 4. Prototype Status & Claims Boundary

> [!WARNING]
> **Claims Discipline Notice:**
> - This evaluation is based on 21 local acoustic utterances across 3 speakers.
> - While software, mathematical formulation, and speaker disjointness are verified, **we do NOT claim general high accuracy** across unconstrained operational environments.
> - Large-scale benchmark testing (e.g., ASVspoof 2019/2021) is required before deploying this layer to production verification workflows.