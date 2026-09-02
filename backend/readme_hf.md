---
license: apache-2.0
language:
  - en
tags:
  - audio-classification
  - deepfake-detection
  - audio-forensics
  - voice-security
  - asvspoof
  - wavlm
  - transformers
datasets:
  - Bisher/ASVspoof_2019_LA
metrics:
  - accuracy
  - f1
base_model: microsoft/wavlm-base
pipeline_tag: audio-classification
widget:
  - example_title: Test Audio
    src: https://huggingface.co/spaces/0xmola/audio-forensics-deepfake-detector
---

# WavLM Deepfake Audio Forensics

Fine-tuned [WavLM-base](https://huggingface.co/microsoft/wavlm-base) for real-time audio deepfake detection.

## Model Description

This model detects AI-cloned/synthetic voices by analyzing raw audio waveforms through a CNN-Transformer hybrid architecture. It identifies synthetic artifacts that human ears miss: unnatural pitch consistency, GAN-generated frequency smoothness, and missing microtremors.

### Architecture
- **Base:** WavLM-base (94.6M params)
- **Task Head:** Sequence Classification (2 classes: bonafide, spoof)
- **Training:** CNN feature extractor frozen, only transformer layers fine-tuned

### Training Recipe
Based on [WavLM Model Ensemble for Audio Deepfake Detection](https://arxiv.org/abs/2408.07414):
- **Dataset:** ASVspoof 2019 LA (25,380 samples)
- **Epochs:** 5
- **Effective Batch Size:** 64 (16 × 4 gradient accumulation)
- **Learning Rate:** 3e-5 with linear warmup (10%)
- **Audio Length:** 4 seconds (64,000 samples at 16kHz)

## Usage

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch, librosa

model_id = "0xmola/wavlm-deepfake-audio-forensics"
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)
model.eval()

# Load audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Inference
inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    
# Risk score (0-100, higher = more likely fake)
spoof_idx = model.config.label2id["spoof"]
risk_score = int(probs[0, spoof_idx].item() * 100)
print(f"Risk Score: {risk_score}/100")
print("⚠️ HIGH RISK" if risk_score >= 60 else "✅ LOW RISK")
```

## Demo

Try the live demo: [Audio Forensics Deepfake Detector](https://huggingface.co/spaces/0xmola/audio-forensics-deepfake-detector)

## References

1. [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
2. [WavLM Model Ensemble for Audio Deepfake Detection](https://arxiv.org/abs/2408.07414)
3. [ASVspoof 2019 Challenge](https://www.asvspoof.org/)
