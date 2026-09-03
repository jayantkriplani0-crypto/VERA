# VERA Layer 1: Machine Learning Handoff Contract & Integration Guide

**Deliverable:** Person 1 (AI/ML / Voice Authenticity) Final Handoff Specification  
**Target Recipient:** Person 2 (Backend / Platform Integration Engineer)  
**Status:** **FROZEN & CERTIFIED FOR INTEGRATION**  
**Date:** September 2026  

---

## 1. Scope & Governance Guarantees

This document constitutes the official contract and integration guide for **VERA Layer 1 (Voice Authenticity Detection)**.

### Strict Governance Assurances:
1. **Model Weights Frozen:** `lab260/Spectra-AASIST3` architecture, checkpoint weights, and tensor execution are strictly frozen. No fine-tuning or retraining was performed.
2. **Calibration Parameters Frozen:** Platt logistic parameters ($w = +1.218015, b = -3.388055$) and calibrated cutoff ($s^* = +2.781620$, Spoof Signal $= 50.0$) are frozen in [`calibration/config.json`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/calibration/config.json).
3. **Reproducibility Verified:** Identical audio produces **0.0000000000** maximum discrepancy across multiple runs and $< 10^{-5}$ discrepancy across Direct Engine vs. REST API.
4. **Test Suite Green:** **87 / 87 automated tests pass** (`python -m pytest ml/tests/ api/tests/ -v`).
5. **No Evaluation Contamination:** `LA_eval` was never used for model training or calibration parameter fitting.

---

## 2. Distinction: Voice Authenticity vs. Fraud Risk

> [!CAUTION]
> **CRITICAL ARCHITECTURAL BOUNDARY (MANDATORY FOR PERSON 2 / BACKEND):**  
> **Layer 1 measures Acoustic & Spectral Voice Authenticity, NOT Fraud Probability.**  
> - $\hat{P}(\text{Spoof})$ is the probability that the physical audio signal was generated or converted by a synthetic algorithm / vocoder.
> - **$\hat{P}(\text{Spoof}) \ne \hat{P}(\text{Fraud})$.**  
> 
> **Why this distinction matters:**
> 1. A legitimate user calling from an aggressive noise-cancelling Bluetooth headset or lossy VoIP codec may trigger acoustic degradation without being a fraudster.
> 2. A real human authorized fraudster (e.g. social engineering / authorized push payment scam) produces authentic human voice ($\hat{P}(\text{BonaFide}) \approx 1.0$), yet the call is fraudulent.
> 3. An authorized synthetic accessibility tool or corporate IVR produces synthetic speech ($\hat{P}(\text{Spoof}) \approx 1.0$) without fraudulent intent.
> 
> **Person 2 / Backend Responsibility:**  
> The backend Risk Engine must fuse Layer 1's `spoof_signal` and `state` with Layer 2 (Liveness / Interactive challenge), Layer 3 (Speaker Verification / Biometric voiceprint), and Layer 4 (Call metadata, behavioral telemetry, transaction amount). **Never take irreversible step-up or account-blocking actions based solely on Layer 1.**

---

## 3. Layer-1 Input & Output Contracts

### A. Input Contract

Layer 1 accepts two primary input modes:

#### 1. Single-Audio File (REST API or Direct Engine)
- **Container Format:** WAV (`.wav`), FLAC (`.flac`), or Ogg (`.ogg`).
- **Channels:** Mono (Multi-channel audio is automatically averaged to mono).
- **Target Sampling Rate:** $16,000$ Hz (Non-16kHz audio is automatically resampled via polyphase filtering).
- **Size Limit:** Maximum 25 MB per request.

#### 2. Streaming Audio Chunks (WebSocket API or Rolling Buffer)
- **Encoding:** Raw 16-bit signed integer PCM bytes or 32-bit float array.
- **Sampling Rate:** Exactly $16,000$ Hz mono.
- **Chunk Ingestion Interval:** Arbitrary packet sizes (Recommended: 100 ms to 500 ms packets, corresponding to 1,600 to 8,000 samples).
- **Session Duration Limit:** Maximum 300 seconds (5 minutes) per continuous stream.

---

### B. Output Contract

Every evaluation window emits the following strongly typed fields:

| Field Name | Type | Range / Format | Semantic Description |
|---|---|---|---|
| `model_version` | `str` | `"lab260/Spectra-AASIST3"` | Fixed model identifier. |
| `request_id` | `str` | UUID string | Unique request or stream session ID. |
| `window_id` | `int` | $\ge 0$ | Monotonic sequence index of the sliding window. |
| `timestamp_start` | `float` | Seconds ($\ge 0.0$) | Elapsed stream time at window start. |
| `timestamp_end` | `float` | Seconds ($> \text{start}$) | Elapsed stream time at window end. |
| `raw_bona_fide_logit` | `float` | $(-\infty, +\infty)$ | Unnormalized logit from transformer ($> 2.781620 \implies$ bona fide). |
| `p_bonafide` | `float` | $[0.0, 1.0]$ | Calibrated posterior probability $\hat{P}(\text{Bona Fide} \mid s) = \sigma(ws + b)$. |
| `p_spoof` | `float` | $[0.0, 1.0]$ | Calibrated posterior probability $\hat{P}(\text{Spoof} \mid s) = 1.0 - \hat{P}(\text{Bona Fide})$. |
| `voice_integrity_score`| `float` | $[0.0, 100.0]$ | Scaled integrity metric: $100.0 \times \hat{P}(\text{Bona Fide})$ (100 = authentic). |
| `spoof_signal` | `float` | $[0.0, 100.0]$ | Scaled deepfake signal: $100.0 \times \hat{P}(\text{Spoof})$ (100 = synthetic). |
| `smoothed_spoof_signal`| `float`| $[0.0, 100.0]$ | EMA-filtered spoof signal ($\alpha = 0.35$). |
| `decision_confidence` | `float` | $[0.0, 1.0]$ | Distance from threshold: $2.0 \times |\hat{P}(\text{Bona Fide}) - 0.5|$. |
| `state` | `str` | Enum | `"BONAFIDE"`, `"SUSPICIOUS"`, or `"SPOOF"`. |
| `is_alert` | `bool` | `true` / `false` | High-priority flag: `true` if `state == "SPOOF"`. |
| `latency_ms` | `float` | Milliseconds | End-to-end forward compute latency for this window. |

#### Operational State Mapping:
- **`BONAFIDE`:** Smoothed Spoof Signal $< 35.0$ (High confidence human speech).
- **`SUSPICIOUS`:** $35.0 \le \text{Smoothed Spoof Signal} < 65.0$ (Borderline audio, acoustic degradation, or transitional state).
- **`SPOOF`:** Smoothed Spoof Signal $\ge 65.0$ (High confidence deepfake / cloned speech; requires 2 consecutive windows to commit).

---

## 4. How to Import & Integrate Layer 1 in Code

### Option A: Via HTTP / WebSocket Microservice (Recommended)
Person 2 launches the API container and consumes standard REST / WebSocket interfaces:

```python
import httpx

# Single audio analysis
async with httpx.AsyncClient() as client:
    with open("call_audio.wav", "rb") as f:
        response = await client.post(
            "http://localhost:8000/api/v1/voice/analyze",
            files={"file": ("call_audio.wav", f, "audio/wav")},
            timeout=10.0,
        )
    result = response.json()
    print(f"Spoof Signal: {result['spoof_signal']}, Decision: {result['classification']}")
```

### Option B: In-Process Python Import (Direct Service Layer)
If Person 2 wishes to embed Layer 1 directly inside the backend service process without network hops:

```python
from pathlib import Path
from api.services.voice_service import VoiceService

# 1. Initialize singleton once at backend startup
voice_service = VoiceService(device="cpu", num_threads=6)

# 2. Analyze audio file/bytes directly
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

analysis = await voice_service.analyze_audio_bytes(audio_bytes)
print(f"Classification: {analysis.classification}, Integrity: {analysis.voice_integrity_score}")

# 3. Handle live streaming WebSocket / WebRTC session
session = voice_service.create_streaming_session()
try:
    async for chunk_bytes in audio_stream:
        events = await session.push_chunk(chunk_bytes)
        for ev in events:
            if ev.is_alert:
                await trigger_liveness_challenge(ev)
finally:
    session.release()
```

---

## 5. Failure Modes & Operational Limitations

Do NOT treat Layer 1 as an infallible oracle. The following failure modes have been empirically observed and documented:

| Failure Mode | Underlying Root Cause | Observed Behavior | Recommended Backend Mitigation |
|---|---|---|---|
| **Acoustic Noise / Reverberation** | Heavy environmental background noise, room reverb, or crowd chatter | Can lower bona-fide logit toward the boundary (e.g. from `+13` to `+1.3`) | EMA smoothing absorbs momentary spikes; rely on `SUSPICIOUS` buffer before alerting. |
| **Severe VoIP Compression** | High-compression codecs (e.g. G.729, AMR 4.75kbps, GSM-FR) discarding spectral detail | Artifacts resembling vocoders can cause false spoof suspicion | Use Layer 4 call metadata to recognize low-bitrate carrier codecs and widen hysteresis bands. |
| **Very Short Audio ($< 1.5$s)** | Insufficient phonetic variation for transformer temporal self-attention | Model window tiles audio to fill 64,600 samples; slight boundary artifact | Accumulate at least 3.0–4.0 seconds of speech before making final fraud determinations. |
| **Silent / Low-Energy Audio** | Pure silence, microphone muting, or background hiss | Produces ambiguous logits around `0.0` to `+2.0` | Implement Voice Activity Detection (VAD) before sending chunks to Layer 1. |
| **Acoustic Replay Attacks** | High-fidelity recording played through physical speakers into a mic | May exhibit genuine vocal tracts but acoustic channel coloration | Layer 1 detects vocoders/synthesis; physical replay requires Layer 2 challenge-response. |
| **Unseen Zero-Day Vocoders** | Novel generative synthesis architectures not represented in training | Possible false acceptance of synthetic speech | The 13 ASVspoof 2019 evaluation attacks (`A07`–`A19`) all generalized at 100%, but future zero-day models warrant periodic benchmark re-testing. |
| **CPU Processing Latency** | Sequential forward passes take ~780–830 ms per 4-second window | Real-time factor is 0.63 (faster than real time), but latency is not zero | Run sliding window hops of 1.0s or 2.0s so processing stays strictly ahead of audio arrival. |

---

## 6. Performance & Operational Baseline

The accepted, frozen engineering baseline on standard CPU (Intel, 6 threads):
- **Model Load Time:** `5.93 seconds` (One-time startup penalty).
- **First Inference Latency:** `803.64 ms` cold start ($0.88$s from stream start).
- **Per-Window Forward Latency:**
  - Minimum: `783.73 ms`
  - Median: `827.11 ms`
  - Mean: `850.69 ms`
  - 95th Percentile: `975.54 ms`
  - 99th Percentile: `998.44 ms`
- **Real-Time Factor (RTF):** `0.6382` ($\implies$ **1.57x–1.60x faster than real-time**).
- **Throughput:** `1.18–1.20 windows/second` (`25,070 samples/sec`).
- **Peak Process RAM (RSS):** `1,814.9 MB` ($< 2.0$ GB).

---

## 7. Deployment & Local Startup Instructions

### Installation & Environment:
```bash
# Verify Python version >= 3.10
python --version

# Run all 87 verification tests
python -m pytest ml/tests/ api/tests/ -v
```

### Starting the Layer 1 API:
```bash
# Start API on port 8000 with 1 worker (internal thread pool handles concurrency)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Validating via Interactive Client:
```bash
# 1. Health check
python scripts/demo_client.py --mode health

# 2. Analyze audio file
python scripts/demo_client.py --mode analyze --file path/to/sample.wav

# 3. Simulate real-time streaming WebSocket
python scripts/demo_client.py --mode stream --file path/to/sample.wav
```

### Interactive API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc UI: `http://localhost:8000/redoc`

---

## 8. Exact Files for Person 2 Integration

Person 2 needs only the following clean interfaces:

1. **FastAPI Application:** [`api/app.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/api/app.py) (`from api.app import app`).
2. **Service Layer Singleton:** [`api/services/voice_service.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/api/services/voice_service.py) (`VoiceService`).
3. **Data Models & Schemas:** [`api/schemas.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/api/schemas.py) (`VoiceAnalysisResponse`, `StreamingVoiceEvent`).
4. **Frozen Model Manifest:** [`docs/layer1_freeze_manifest.json`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/docs/layer1_freeze_manifest.json).
5. **Interactive Client Demo:** [`scripts/demo_client.py`](file:///c:/Users/jayan/OneDrive/Desktop/VERA/scripts/demo_client.py).
