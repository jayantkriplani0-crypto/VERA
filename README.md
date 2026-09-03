# VERA - Person 2: Speaker Consistency & Conversation Intelligence Pipeline

This repository contains the complete implementation for **Person 2** in the **VERA (Voice Scam & Social Engineering Detection)** architecture.

---

## 🗺️ Architecture & Flowchart

```text
                           [ INCOMING AUDIO (16kHz WAV) ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ BRANCH A: WHO? ]                             [ BRANCH B: WHAT & WHY? ]
      Speaker Consistency (ECAPA)                   Conversation Intelligence (Whisper + LLM)
                  │                                               │
    ┌─────────────┴─────────────┐                   ┌─────────────┴─────────────┐
    ▼                           ▼                   ▼                           ▼
[Reference Enrollment]    [Live Embedding]    [faster-whisper]         [Transcript Chunks]
    │                           │                   │                           │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  ▼                                               ▼
         [Cosine Similarity]                             [Gemini Structured AI]
                  │                                               │
                  ▼                                               ▼
       {speaker_match_score}                         {intent, urgency, secrecy, ...}
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                               [ PERSON 2 JSON CONTRACT ]
                                          │
                                          ▼
                            [ PERSON 3: RISK ENGINE ]
```

---

## 📁 Project Directory Layout

```text
vera_person2/
├── config.py                 # Central configurations, model paths, thresholds
├── requirements.txt          # Python package requirements
├── speaker/                  # Branch A: Speaker consistency
│   ├── __init__.py
│   ├── enroll.py             # Reference voice enrollment & embedding caching (.npy/.pt)
│   └── verifier.py           # Live embedding extraction & cosine similarity
├── asr/                      # Branch B: Automatic speech recognition
│   ├── __init__.py
│   └── transcriber.py        # faster-whisper local worker & chunk processor
├── conversation/             # Branch B: Conversation intelligence
│   ├── __init__.py
│   ├── schemas.py            # Pydantic schemas for JSON contract
│   └── analyzer.py           # Gemini Structured Output & multilingual rule fallback
├── pipeline/                 # Core orchestration & Person 3 interface
│   ├── __init__.py
│   └── engine.py             # Orchestrates parallel/sequential execution
├── test_data/                # Sample reference voices & test call audio
│   ├── generate_samples.py   # Utility to generate realistic mock audio WAVs
│   └── conversation_cases.py # Ground-truth transcripts (Hindi, Hinglish, English)
├── main.py                   # Complete CLI & demonstration runner
└── README.md                 # Team documentation & explanation guide
```

---

## 🚀 Quickstart & Usage

### 1. Run Complete Simulation Demo
```bash
python main.py
```

### 2. Enroll a New Reference Speaker Profile
```bash
python main.py --enroll CEO_Rahul test_data/ceo_reference.wav
```

### 3. Process Live Audio Chunk Against Enrolled Speaker
```bash
python main.py --process test_data/scam_call_imposter.wav CEO_Rahul
```

---

## 📄 Output Contract (Person 2 -> Person 3)

```json
{
  "speaker": {
    "match_score": 0.471,
    "confidence": 0.68,
    "speaker_id": "CEO_Rahul"
  },
  "conversation": {
    "transcript": "Attention sir, this is Senior Officer Sharma from Cyber Crime Branch. Your account is linked to money laundering. Share your OTP immediately or police will be dispatched. Do not inform anyone.",
    "intent": {
      "label": "credential_request",
      "action": "share_otp"
    },
    "interaction": {
      "urgency": true,
      "secrecy": false,
      "authority_pressure": true,
      "threat": false,
      "credential_request": true
    },
    "emotion": {
      "label": "urgency",
      "confidence": 0.85
    },
    "pressure": "high"
  },
  "timestamp": "2026-09-02T11:25:51.114902Z",
  "latency_ms": 36.45
}
```
