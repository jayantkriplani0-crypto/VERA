import numpy as np

_extractor = None
_model = None
_device = None

def get_model():
    global _extractor, _model, _device
    if _model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "MelodyMachine/Deepfake-Audio-Detection-V2"
        
        _extractor = AutoFeatureExtractor.from_pretrained(model_id)
        _model = AutoModelForAudioClassification.from_pretrained(model_id)
        _model.to(_device)
        _model.eval()
    return _extractor, _model, _device

def analyze_voice(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    if len(audio_array) == 0:
        return {
            "voice_integrity_score": 0.5,
            "label": "genuine",
            "model_name": "MelodyMachine/Deepfake-Audio-Detection-V2",
            "confidence": 0.0
        }
        
    # Safely handle very short audio
    if len(audio_array) < 400:
        audio_array = np.pad(audio_array, (0, 400 - len(audio_array)), 'constant')
        
    extractor, model, device = get_model()
    import torch

    inputs = extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        
    # 0 = fake, 1 = real
    fake_prob = float(probs[0, 0].item())
    real_prob = float(probs[0, 1].item())
    
    is_fake = fake_prob >= 0.5
    label = "synthetic" if is_fake else "genuine"
    confidence = fake_prob if is_fake else real_prob
    
    return {
        "voice_integrity_score": fake_prob,
        "label": label,
        "model_name": "MelodyMachine/Deepfake-Audio-Detection-V2",
        "confidence": confidence
    }
