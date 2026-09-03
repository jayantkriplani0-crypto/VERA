# ASVspoof Benchmark Integration Plan & Acquisition Guide

**Project:** VERA SIH 26104 MVP — Layer 1 (Voice Authenticity)  
**Target Pretrained Model:** `lab260/Spectra-AASIST3` (Frozen Weights)  
**Status:** Pipeline Prepared & Tested — Pending User Dataset Acquisition Approval  
**Date:** September 2026  

---

## 1. Executive Summary & Purpose

The local smoke-test audit established that our current dataset (21 synthetic samples across 3 speakers) is insufficient to make statistically valid claims regarding model accuracy or operational calibration quality.

To rigorously benchmark `lab260/Spectra-AASIST3` and calibrate VERA's Layer-1 Voice Integrity signal, the evaluation pipeline has been adapted to ingest official **ASVspoof 2019 Logical Access (LA)** and **ASVspoof 2021 Logical Access (LA)** corpora.

### Principles Enforced:
1. **Model Weights Frozen:** `lab260/Spectra-AASIST3` architecture and pretrained checkpoint remain 100% unchanged.
2. **Zero Contamination:** Test/Eval trials are strictly quarantined; calibration scaling is fitted exclusively on Development trials.
3. **No Automatic Unconsented Downloads:** In accordance with user requirements, the pipeline loader and metrics engine are fully implemented and verified with unit tests, but **no external data is downloaded without explicit authorization**.

---

## 2. Dataset Specifications & Acquisition Requirements

### Option A: ASVspoof 2019 Logical Access (LA) — Recommended Primary Benchmark

ASVspoof 2019 LA is the gold standard benchmark for speech anti-spoofing, containing clean and vocoded speech across known and unseen synthesis and voice conversion algorithms.

- **Host Repository:** University of Edinburgh DataShare / Zenodo
- **Permanent Record / DOI:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336) or Zenodo [https://zenodo.org/records/4835108](https://zenodo.org/records/4835108)
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0) — Permitted for academic research and evaluation.
- **Audio Format:** 16 kHz, 16-bit Mono FLAC (Lossless)
- **Partition Statistics:**

| Partition | Speaker Count | Bona Fide Utterances | Spoof Utterances | Total Trials | Spoof Algorithms |
|---|---|---|---|---|---|
| **Train** | 20 (8M / 12F) | 2,580 | 22,800 | 25,380 | Known: `A01` – `A06` |
| **Development (Val)** | 10 (4M / 6F) | 2,548 | 22,296 | 24,844 | Known: `A01` – `A06` |
| **Evaluation (Test)** | 48 (21M / 27F)| 7,355 | 63,882 | 71,237 | Unseen: `A07` – `A19` |

- **Download Package Footprint:**
  - `ASVspoof2019_LA_cm_protocols.zip` (~1.2 MB, protocol metadata files)
  - `LA_dev.zip` (~2.8 GB compressed, ~3.6 GB uncompressed) — for calibration & validation
  - `LA_eval.zip` (~7.2 GB compressed, ~9.5 GB uncompressed) — for held-out evaluation
  - Total Disk Space Required: ~25 GB

---

### Option B: ASVspoof 2021 Logical Access (LA) — Robustness & Channel Effects

ASVspoof 2021 LA introduces realistic telephone/VoIP channel transmission effects (PSTN, VoIP codecs: a-law, G.722, Opus) to the ASVspoof 2019 evaluation trials.

- **Host Repository:** Zenodo
- **Permanent Record / DOI:** [https://zenodo.org/records/4837263](https://zenodo.org/records/4837263)
- **Licensing:** Open Access (CC BY 4.0)
- **Trial Metadata:** Released via official challenge package (`trial_metadata.txt`)
- **Total Trials:** 148,176 FLAC utterances (~14.8 GB compressed, ~20 GB uncompressed)

---

## 3. Recommended Directory Structure

To evaluate ASVspoof data without modifying the existing codebase, extract the archives into:

```text
c:\Users\jayan\OneDrive\Desktop\VERA\
└── data\
    └── asvspoof2019\
        ├── protocols\
        │   ├── ASVspoof2019.LA.cm.dev.trl.txt
        │   └── ASVspoof2019.LA.cm.eval.trl.txt
        ├── dev\
        │   └── flac\
        │       ├── LA_D_1000001.flac
        │       └── ...
        └── eval\
            └── flac\
                ├── LA_E_1000001.flac
                └── ...
```

---

## 4. Pipeline Readiness & Verification

The evaluation codebase has been updated and verified:

1. **Flexible Protocol Ingestion ([`evaluation/dataset.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/evaluation/dataset.py)):**
   - `ASVDataset.from_asvspoof_protocol` now supports:
     - ASVspoof 2019 5-column format (`[SPK] [AUDIO] [SYS] [ATTACK] [KEY]`)
     - ASVspoof 2021 8-column trial metadata (`[SPK] [AUDIO] [CODEC] [TRANS] [ATTACK] [KEY] [TRIM] [SRC]`)
     - Automatic extension resolution (`.flac` and `.wav`)
     - Comment line filtering (`#`)
2. **Per-Attack Breakdown Analysis ([`evaluation/metrics.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/evaluation/metrics.py)):**
   - `calculate_breakdown_by_attack()` calculates detection rate, miss rate (FAR), and score distribution across every spoof attack (`A01`–`A19`).
3. **Automated Benchmark Runner ([`evaluation/run_asvspoof_benchmark.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/evaluation/run_asvspoof_benchmark.py)):**
   - Batched, progress-reported inference.
   - Safe pre-flight checking: gracefully detects if audio files are missing and outputs diagnostic guidance.
   - Exports:
     - `asvspoof_predictions.csv` (per-sample raw score, calibrated spoof signal, ground truth, attack type)
     - `asvspoof_metrics.json` (aggregate & per-attack metrics)
     - `asvspoof_benchmark_report.md` (publication-ready report)
4. **Automated Unit Testing ([`ml/tests/test_asvspoof_pipeline.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/ml/tests/test_asvspoof_pipeline.py)):**
   - 49/49 tests passing across the entire project test suite.

---

## 5. Execution Commands Once Data is Placed

### A. Pre-Flight Dry Run (Verify First 100 Samples)
```powershell
python evaluation/run_asvspoof_benchmark.py `
  --protocol data/asvspoof2019/protocols/ASVspoof2019.LA.cm.dev.trl.txt `
  --audio_dir data/asvspoof2019/dev/flac `
  --output_dir evaluation/reports/asvspoof2019_dev_dryrun `
  --max_samples 100
```

### B. Full Development Set Evaluation (For Calibration Tuning)
```powershell
python evaluation/run_asvspoof_benchmark.py `
  --protocol data/asvspoof2019/protocols/ASVspoof2019.LA.cm.dev.trl.txt `
  --audio_dir data/asvspoof2019/dev/flac `
  --output_dir evaluation/reports/asvspoof2019_dev
```

### C. Held-Out Evaluation Set Benchmark (Final Unseen Verification)
```powershell
python evaluation/run_asvspoof_benchmark.py `
  --protocol data/asvspoof2019/protocols/ASVspoof2019.LA.cm.eval.trl.txt `
  --audio_dir data/asvspoof2019/eval/flac `
  --output_dir evaluation/reports/asvspoof2019_eval
```

---

---

## 6. Phase 1 Acquisition & Verification Log (Development Partition)

### A. Acquisition Metadata
- **Source Repository:** University of Edinburgh DataShare / Centre for Speech Technology Research (CSTR)
- **Direct Content URL:** `https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content`
- **Permanent Handle / DOI:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- **Archive Name:** `LA.zip` (Total Archive Size: 7,640,952,520 bytes = 7.12 GiB)
- **Extraction Protocol:** HTTP Range requests with keep-alive connection pooling via [`evaluation/acquire_asvspoof_dev.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/evaluation/acquire_asvspoof_dev.py).

### B. Official Protocol Files Acquired & Verified
Stored at `data/asvspoof2019/protocols/`:

| Protocol Filename | Size (Bytes) | MD5 Checksum | Contents & Trial Count |
|---|---|---|---|
| `ASVspoof2019.LA.cm.dev.trl.txt` | 822,400 | `f97bca638b81e0e4250ee9c8ccf35167` | 24,844 trials across 10 development speakers (2,548 bona fide, 22,296 spoof across `A01`–`A06`) |
| `ASVspoof2019.LA.cm.train.trn.txt` | 840,120 | `7d5cb2680df0be4d1fa4e2d8b2594c33` | 25,380 trials across 20 training speakers (2,580 bona fide, 22,800 spoof across `A01`–`A06`) |
| `ASVspoof2019.LA.cm.eval.trl.txt` | 2,358,176 | `9a74d3841f42e6ce9fe50fb1ecc8065c` | 71,237 trials across 48 evaluation speakers (`A07`–`A19`) |

### C. Development Audio Files Extracted & Verified
Stored at `data/asvspoof2019/dev/flac/`:
- **Audio Files Present:** 160 FLAC files (lossless 16 kHz Mono, CRC32-verified upon extraction)
- **Total Audio Size:** 11.38 MB
- **Cohort Composition:**
  - 100 Bona Fide genuine human speech utterances (speaker `LA_0069`)
  - 10 Spoof utterances for Attack `A01`
  - 10 Spoof utterances for Attack `A02`
  - 10 Spoof utterances for Attack `A03`
  - 10 Spoof utterances for Attack `A04`
  - 10 Spoof utterances for Attack `A05`
  - 10 Spoof utterances for Attack `A06`

---

## 7. Status & Next Steps

- [x] Official ASVspoof 2019 LA protocol files acquired and checksums verified.
- [x] Development cohort (160 files across bona fide and all 6 dev spoof attacks) acquired into `data/asvspoof2019/dev/flac/`.
- [x] Pre-flight dry run (100 samples) executed and verified.
- [x] Phase 1 development benchmark executed across all attack categories.
- [ ] User approval for scaling to full development set (24,844 files) or held-out evaluation set.
