# Layer 1 (Voice Authenticity) Model Implementation Audit
**Project:** VERA SIH 26104 MVP  
**Model Target:** `lab260/Spectra-AASIST3`  
**Model Card Reference:** [Hugging Face: lab260/Spectra-AASIST3](https://huggingface.co/lab260/Spectra-AASIST3)  
**Date of Audit:** September 2026  
**Status:** **VERIFIED & COMPLIANT**

---

## 1. How the Model Is Currently Loaded

- **Mechanism:** The model is loaded using `PyTorchModelHubMixin` via `SpectraAASIST3.from_pretrained("lab260/Spectra-AASIST3")`.
- **Initialization Sequence:**
  1. `SpectraAASIST3.__init__` instantiates the three core architectural blocks:
     - `Wav2Vec2Encoder`: Loads the SSL transformer backbone architecture (`facebook/wav2vec2-xls-r-300m`) via Hugging Face `transformers.Wav2Vec2Model`.
     - `MLPBridge`: Single-layer `Linear(1024 -> 128)` projection with SELU activation and 0.1 dropout.
     - `KANAASIST`: Graph attention network (GAT-S and GAT-T) using Kolmogorov-Arnold (`KANLinear`) B-spline layers and multi-branch heterogeneous pooling.
  2. `from_pretrained` downloads and loads `model.safetensors` (~1.27 GB self-contained weights) from the repository, overwriting all parameters across both the front-end encoder and the back-end classifier.
  3. The model is set to evaluation mode (`.eval()`) and transferred to the target device (`cuda` or `cpu`).

---

## 2. Vendored / Copied Official Files

| File in Project | Source in Hub Repo | Status & Integrity |
|---|---|---|
| `models/spectra_aasist3_net.py` | `model.py` (`lab260/Spectra-AASIST3`) | **Exact Byte-for-Byte Copy** (0 architectural modifications) |

---

## 3. Preprocessing Implementation

The preprocessing pipeline in `infer_spectra_aasist3.py` follows the official model card eval specifications:

1. **Audio Input:** 16,000 Hz single-channel mono `float32` waveform.
2. **Preemphasis:** First-order high-pass preemphasis filter applied across the full waveform:
   $$y[n] = x[n] - 0.97 \cdot x[n-1]$$
   (`PREEMPH_COEF = 0.97`)
3. **Deterministic Windowing:** Fixed length of **64,600 samples** (~4.04 seconds):
   - **Shorter utterances:** Tile-repeated (`np.tile`) until length $\ge 64,600$, then sliced to `[:64600]`.
   - **Longer utterances:** Deterministically truncated to the first `64,600` samples (`[:64600]`).
   - **No random cropping / No random padding.**

---

## 4. How the Output Score Is Produced

- **Forward Output:** The model outputs a raw 2-class logits tensor: $\text{logits} \in \mathbb{R}^{B \times 2}$.
- **Class Indexing:**
  - Class 0: Spoof / Synthetic / Deepfake
  - Class 1: Bona Fide (Genuine human speech)
- **Score Extraction:** The inference function returns `logits[:, 1]`.
- **Interpretation:** **Higher = More Bona Fide (Genuine).** Lower / Negative = More likely deepfake / synthetic.

---

## 5. Official Model Card Compliance Matrix

| Requirement | Official Model Card Specification | Current Implementation | Match |
|---|---|---|:---:|
| **Model Repository** | `lab260/Spectra-AASIST3` | `lab260/Spectra-AASIST3` | Yes |
| **Backbone Front-End** | `facebook/wav2vec2-xls-r-300m` | `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")` | Yes |
| **Bridge Layer** | Linear(1024 -> 128), SELU, Dropout 0.1 | `MLPBridge(1024, 128, hidden_dim=128, dropout=0.1, activation=nn.SELU(), n_layers=1)` | Yes |
| **Back-End Classifier** | KAN-AASIST with 4 HSGAL branches | `KANAASIST(filts=[128, [1, 32], ...], layer_type="KANLinear")` | Yes |
| **Sampling Rate** | 16 kHz mono | 16 kHz mono validated | Yes |
| **Preemphasis** | 0.97 coefficient | 0.97 coefficient | Yes |
| **Sample Window** | 64,600 samples, tile-repeat if shorter | 64,600 samples, tile-repeat if shorter | Yes |
| **Output Logits** | Index 1 = Bona Fide score | Index 1 extracted (`logits[0, 1]`) | Yes |
| **No Inventions** | No synthetic post-processing / invented thresholds | Pure raw logits preserved | Yes |

---

## 6. Audit Summary: Correctness, Gaps & Next Steps

### What Is Correct:
1. **Zero Mismatches:** All architecture parameters, weights, preprocessing filters, and output indexing strictly follow the official Hugging Face model repository.
2. **Verified Test Execution:** Tested locally against `samples/test_sample.wav`, yielding the verified output:
   $$\text{Bona fide score} = 2.382637$$

### What Is Missing (For VERA Layer 1 Integration):
1. **Modular Service Layer (`ml/voice_detector/`):** A clean Python class (e.g. `VoiceAuthenticityDetector`) with batch support and standard response formatting for integration into the VERA MVP pipeline and FastAPI backend.
2. **Batch Inference Support:** `score_batch(audios, srs)` utility for processing multi-segment audio streams.
3. **Threshold Context Documentation:** The official `model.py` specifies an EER operational cutoff of `-1.0625009` (`classify()` method), which should be documented for downstream decision logic while maintaining raw logit output.

### Recommended Actions:
- Keep `infer_spectra_aasist3.py` intact as the reference standalone CLI script.
- Package Layer 1 components cleanly in `ml/voice_detector/` for consumption by the VERA SIH 26104 pipeline.
