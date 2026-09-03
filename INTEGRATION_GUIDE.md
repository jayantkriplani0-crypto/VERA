# Person 2 (Speaker & Conversation AI) - Team Integration Guide

## 📦 What Person 2 Delivers
1. **Speaker Consistency (WHO?)**: Uses pretrained `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) to extract 192-dim speaker embeddings and compute cosine similarity against enrolled profiles.
2. **ASR (WHAT?)**: Uses local `faster-whisper` for low-latency, privacy-preserving speech-to-text chunking.
3. **Conversation Intelligence (WHAT ARE THEY ASKING?)**: Uses Gemini Structured AI / Contextual NLP to extract intent, urgency, secrecy, authority pressure, threats, and OTP/credential requests.

---

## 🚀 How Person 3 / Pipeline Leader Calls Person 2 in Python

```python
from pipeline.engine import Person2Pipeline

# 1. Initialize Pipeline (Loads ECAPA + faster-whisper + Gemini)
pipeline = Person2Pipeline()

# 2. Process an audio chunk (WAV 16kHz) against an enrolled reference profile
result = pipeline.process_chunk(
    audio_path="incoming_call_chunk.wav",
    reference_speaker_id="CEO_Rahul" # Optional: pass None if no reference voice enrolled
)

# 3. Get standard Person 3 JSON Contract
person3_payload = result.to_person3_contract()
print(person3_payload)
```

---

## 📄 Exact Output JSON Contract (Delivered to Person 3)

```json
{
  "speaker_match": 0.23,
  "transcript": "Attention Sir, this is Senior Officer Sharma from Cybercrime Branch. Your account is linked to money laundering. Share your OTP immediately or police will be dispatched. Do not inform anyone.",
  "intent": "credential_request",
  "interaction": {
    "urgency": true,
    "secrecy": false,
    "authority_pressure": true
  },
  "emotion": {
    "label": "urgency",
    "confidence": 0.85
  }
}
```

---

## 🛠️ Installation for Team Members
```bash
pip install -r requirements.txt
python main.py
```
