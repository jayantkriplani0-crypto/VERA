# Layer 1 Voice Authenticity Evaluation Pipeline
**Model:** `lab260/Spectra-AASIST3`  
**Framework:** VERA SIH 26104 MVP  

---

## 1. Overview & Objectives

This evaluation pipeline quantifies the discrimination performance of the fixed pretrained **`lab260/Spectra-AASIST3`** model in separating bona-fide (genuine human) speech from spoofed / deepfake synthetic speech **without modifying or fine-tuning the model weights**.

### Key Pipeline Features:
1. **ASVspoof Protocol Support:** Ingests standard ASVspoof protocol TXT files, CSV manifests, and JSON formats.
2. **Speaker-Disjoint Partitions:** Enforces strict speaker isolation across `train`, `val`, and `test` splits ($\text{speakers}(A) \cap \text{speakers}(B) = \emptyset$) to prevent data leakage.
3. **Official Metric Standards:** Computes Equal Error Rate (EER), ROC-AUC, Confusion Matrix, Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR), inference latency, and throughput.
4. **No Inventions:** Uses raw bona fide logits ($logits[:, 1]$) with the official model EER threshold (`-1.0625009`). Does not invent fake probabilities or percentage claims.

---

## 2. Directory Structure

```
evaluation/
├── run_evaluation.py     # Main evaluation execution CLI & engine
├── metrics.py            # EER, ROC-AUC, confusion matrix, and throughput metrics
├── dataset.py            # Multi-format dataset loader & speaker-disjoint split controls
├── README.md             # This guide
└── reports/              # Automatically generated evaluation outputs
    ├── predictions_*.csv # Per-sample predictions, metadata, scores, and latencies
    ├── predictions_*.json# JSON export of per-sample results
    ├── metrics_*.json    # Structured JSON summary of all computed metrics
    └── report_*.md       # Comprehensive Markdown evaluation summary table
```

---

## 3. Dataset Formats Supported

### Option A: Standard CSV Manifest
A CSV file containing audio paths and metadata:
```csv
file_path,label,speaker_id,attack_type,language,split
samples/user1_real.wav,bonafide,SPK_001,-,en,test
samples/user1_tts.wav,spoof,SPK_001,TTS-FastSpeech2,en,test
```

### Option B: ASVspoof Protocol File (`.txt`)
Standard ASVspoof format:
```text
SPEAKER_ID AUDIO_FILENAME SYSTEM_ID ATTACK_TYPE KEY
LA_0079 LA_E_2833633 - - bonafide
LA_0079 LA_E_8877456 - A07 spoof
```

### Option C: JSON Manifest
```json
[
  {
    "file_path": "samples/audio1.wav",
    "label": "bonafide",
    "speaker_id": "SPK_001",
    "attack_type": "-",
    "language": "en",
    "split": "test"
  }
]
```

---

## 4. Speaker-Disjoint Splitting Guarantee

To prevent identity leakage during evaluation:
- Datasets are partitioned strictly by `speaker_id`.
- The function `ASVDataset.verify_speaker_disjointness(splits)` asserts that zero speaker IDs overlap across subsets:
$$\text{speakers}(\text{train}) \cap \text{speakers}(\text{val}) = \emptyset$$
$$\text{speakers}(\text{train}) \cap \text{speakers}(\text{test}) = \emptyset$$
$$\text{speakers}(\text{val}) \cap \text{speakers}(\text{test}) = \emptyset$$
- If any overlap is detected, a `SpeakerLeakageError` is immediately raised.

---

## 5. Usage & CLI Commands

### Run Evaluation on an Existing Manifest
```bash
python evaluation/run_evaluation.py --manifest path/to/dataset.csv
```

### Run Evaluation on an ASVspoof Protocol Directory
```bash
python evaluation/run_evaluation.py --manifest protocols/ASVspoof2019.LA.cm.eval.trl.txt --audio-dir path/to/wavs/
```

### Run Evaluation with Automatic Speaker-Disjoint Splitting
```bash
python evaluation/run_evaluation.py --manifest data/full_manifest.csv --split-speakers --split test
```

### Run Synthetic Benchmark Verification (Out-of-the-box)
```bash
python evaluation/run_evaluation.py --generate-benchmark --num-benchmark-speakers 6 --device auto
```

---

## 6. Output Metrics & Report Structure

Every evaluation run creates four timestamped artifacts in `evaluation/reports/`:
1. `predictions_<split>_<timestamp>.csv` — Full sample-level table (file path, ground truth, logit score, speaker ID, attack type, correct/miss flag, latency).
2. `predictions_<split>_<timestamp>.json` — Structured JSON array of all predictions.
3. `metrics_<split>_<timestamp>.json` — Complete numerical summary (EER, AUC, precision, recall, F1, latency, throughput).
4. `report_<split>_<timestamp>.md` — Human-readable Markdown report table with confusion matrix.
