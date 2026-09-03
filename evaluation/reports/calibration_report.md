# Layer 1 (Voice Authenticity) Model Output Calibration Report
**Target Model:** `lab260/Spectra-AASIST3` (Pretrained Official Checkpoint — Frozen & Unchanged)  
**Benchmark Source:** ASVspoof 2019 LA Development Partition  
**Calibration Method:** PlattScaling (Sigmoid / Logistic Scaling)  
**Date:** September 2026  

---

## 1. Executive Summary & Calibration Objectives

The raw output of `lab260/Spectra-AASIST3` is an unnormalized real-valued logit where **higher means more bona fide (genuine)**. While raw logits are optimal for ROC/EER thresholding, downstream fusion in the VERA pipeline requires a bounded, calibrated authenticity score on a standard **0–100 scale** without distorting the model's discriminative power.

### Key Principles Enforced:
1. **Strict Split Discipline:** Calibration parameters ($w, b$) were fitted **EXCLUSIVELY** on the calibration fit partition (`cal_fit`, 420 samples).
2. **Zero Test Contamination:** The held-out validation set (`cal_val`, 280 samples across 4 completely unseen speakers) was **never accessed during parameter learning**.
3. **Zero Evaluation Leakage:** The official ASVspoof 2019 evaluation set (`LA_eval.zip`) was **not downloaded or accessed**.
4. **Preservation of Raw Logits:** Raw model logits are preserved in all predictions and reported alongside calibrated probabilities.
5. **Terminology Compliance:** The calibrated 0–100 value is defined as the **Voice Integrity Score** (calibrated posterior $\hat{P}(\text{Bona Fide}) \times 100$). It is strictly **NEVER referred to as accuracy**.

---

## 2. Partitioning Discipline & Speaker Separation

| Partition | Sample Count | Speaker Count | Speaker IDs | Composition (Bona Fide / Spoof) | Role in Calibration Pipeline |
|---|---|---|---|---|---|
| **Calibration Fit (`cal_fit`)** | 420 | 6 | `LA_0069, LA_0070, LA_0071, LA_0072, LA_0073, LA_0074` | 168 / 252 | **Exclusively used to learn parameters ($w, b$)** |
| **Held-Out Validation (`cal_val`)** | 280 | 4 | `LA_0075, LA_0076, LA_0077, LA_0078` | 112 / 168 | **Untouched benchmark for out-of-sample evaluation** |

> [!IMPORTANT]
> **Speaker Disjointness Guarantee:**  
> $\text{speakers}(\text{cal\_fit}) \cap \text{speakers}(\text{cal\_val}) = \emptyset$.  
> There is zero speaker identity leakage between calibration learning and calibration testing.

---

## 3. Candidate Calibration Methods Comparison (Fitted on `cal_fit`)

Three candidate calibrators were fitted and compared strictly on the `cal_fit` partition:

| Calibration Approach | Method Type | Brier Score (lower is better) | Expected Calibration Error (ECE) | Log-Loss | Selection Decision |
|---|---|---|---|---|---|
| **PlattScaling** | Parametric Logistic | `0.005800` | `0.004939` | `0.038889` | ✅ **Selected Method** |
| **TemperatureScaling** | Single Temperature | `0.006085` | `0.002007` | `0.053663` | Rejected |
| **IsotonicRegression** | Non-parametric Step | `0.004705` | `0.000000` | `0.026802` | Rejected |

### Rationale for Selecting Platt Scaling:
1. **Strict Monotonicity:** The sigmoid transform preserves the exact rank-ordering of raw logits, guaranteeing that ROC-AUC and EER remain completely intact.
2. **Smooth Probabilistic Calibration:** Platt scaling provides smooth, well-conditioned probability estimates for unseen extreme logits without the plateau/clipping artifacts of Isotonic Regression.
3. **Affine Flexibility ($w, b$):** Temperature scaling lacks an intercept ($b=0$), assuming symmetry around logit $0.0$. Because Spectra-AASIST3 logits are shifted positively, the intercept ($b = -3.388$) is mathematically required to center the decision boundary.

---

## 4. Calibrated Mapping Function & Learned Parameters

### Mathematical Formulation:
Given raw bona fide logit $s = \text{logits}[:, 1]$ from `Spectra-AASIST3`:

$$\hat{P}(\text{Bona Fide} \mid s) = \sigma(w \cdot s + b) = \frac{1}{1 + e^{-(w \cdot s + b)}}$$

$$\hat{P}(\text{Spoof} \mid s) = 1.0 - \hat{P}(\text{Bona Fide} \mid s)$$

$$\text{calibrated\_spoof\_signal} = \hat{P}(\text{Spoof} \mid s) \times 100.0 \quad \in [0.0, 100.0]$$

$$\text{voice\_integrity\_score} = \hat{P}(\text{Bona Fide} \mid s) \times 100.0 = 100.0 - \text{calibrated\_spoof\_signal}$$

$$\text{decision\_confidence} = 2.0 \times |\hat{P}(\text{Bona Fide} \mid s) - 0.5| \quad \in [0.0, 1.0]$$

### Saved Parameters (`calibration/config.json`):
- **Slope Parameter ($w$):** `+1.218015`
- **Intercept Parameter ($b$):** `-3.388055`
- **Operating Cutoff at $P=0.50$:** `s = -b / w = +2.781620` logits (corresponds to `calibrated_spoof_signal = 50.0`)

---

## 5. Comprehensive Performance Comparison: `cal_fit` vs. `cal_val`

| Metric | Calibration Fit (`cal_fit`, 420) | Held-Out Validation (`cal_val`, 280) | Reference / Interpretation |
|---|---|---|---|
| **ROC-AUC** | **0.9928** | **1.0000** | Area under ROC curve (rank-preserved) |
| **Equal Error Rate (EER)** | **0.60%** | **0.00%** | Primary ASVspoof discrimination metric |
| **EER Operating Threshold** | `+2.695514` | `+11.562364` | Threshold where FAR == FRR |
| **Brier Score** | `0.005800` | `0.000103` | Mean squared probability error (lower is better) |
| **Expected Calibration Error (ECE)** | `0.004939` | `0.006912` | Calibration error across probability bins |
| **Log-Loss** | `0.038889` | `0.006964` | Cross-entropy loss |
| **Accuracy @ Calibrated Cutoff** | **99.29%** | **100.00%** | Classification accuracy at $P=0.50$ |
| **False Rejection Rate (FRR)** | **1.79%** | **0.00%** | Genuine voices incorrectly flagged as spoof |
| **False Alarm Rate (FAR)** | **0.00%** | **0.00%** | Spoofs incorrectly accepted as genuine |
| **Precision (Bona Fide)** | **100.00%** | **100.00%** | True bona fide / total classified bona fide |
| **Recall (Bona Fide)** | **98.21%** | **100.00%** | True bona fide / total actual bona fide |
| **F1-Score** | **0.9910** | **1.0000** | Harmonic mean of Precision and Recall |
| **Mean Calibrated P(BonaFide) - Genuine** | `0.9937` | `0.9999` | Average posterior probability for true human speech |
| **Mean Calibrated P(Spoof) - Spoofs** | `0.9859` | `0.9877` | Average posterior probability for deepfake speech |

---

## 6. Score Distribution & Calibration Visualizations

The generated calibration diagnostic plot is saved to `evaluation/reports/calibration_curve.png`:
- **Panel A (Reliability Diagram):** Calibration curve demonstrating near-perfect alignment with the ideal diagonal (ECE < 0.7%).
- **Panel B (Sigmoid Function Curve):** Smooth parametric mapping with decision cutoff at $+2.7816$ logits dividing bona-fide and spoof clusters.
- **Panel C (Score Separation):** Bimodal separation of Voice Integrity Scores on held-out validation data (bona fide clustered at 99–100, spoofs clustered at 0–5).

---

## 7. Score Interpretation Guide for Downstream VERA Pipeline

| Voice Integrity Score Band | Spoof Risk Band | Qualitative Assessment | Recommended Action in VERA Pipeline |
|---|---|---|---|
| **80.0 – 100.0** | 0.0 – 20.0 | **Strong Bona Fide** | Authenticated genuine human voice; Layer 1 pass |
| **50.0 – 79.9** | 20.1 – 50.0 | **Moderate Bona Fide** | Likely genuine voice; low anomaly signal |
| **20.0 – 49.9** | 50.1 – 80.0 | **Suspicious / Probable Spoof** | Flagged for multi-layer multimodal inspection |
| **0.0 – 19.9** | 80.1 – 100.0 | **Confirmed Deepfake / Synthetic** | High-confidence synthetic voice detected |

> [!IMPORTANT]
> **Governance & Verification Checks:**  
> 1. `cal_val` was strictly **held-out** and never used during parameter fitting.  
> 2. `LA_eval` remains **completely untouched and quarantined**.  
> 3. `lab260/Spectra-AASIST3` weights and inference code were **not modified**.  
> 4. All 49 project unit tests pass.