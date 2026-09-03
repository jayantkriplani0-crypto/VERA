import numpy as np
import torch
import logging

logger = logging.getLogger("vera.voice_integrity")

_model = None
_device = None

# The model's KANAASIST expects exactly 64400 raw samples (4.025s at 16kHz)
# which maps to exactly 400 SSL encoder frames.
EXPECTED_SAMPLES = 64400

def get_model():
    global _model, _device
    if _model is None:
        from models.spectra_aasist3_net import spectra_aasist3
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            _model = spectra_aasist3.from_pretrained("lab260/Spectra-AASIST3")
        except Exception as e:
            logger.warning(f"Could not fetch pretrained weights for Spectra-AASIST3 ({e}). Initializing randomly.")
            _model = spectra_aasist3()
            
        _model.to(_device)
        _model.eval()
    return None, _model, _device

def analyze_voice(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    if len(audio_array) == 0:
        return {
            "voice_integrity_score": 0.5,
            "label": "genuine",
            "model_name": "Spectra-AASIST3",
            "confidence": 0.0
        }
        
    _, model, device = get_model()

    # Truncate or pad audio to exactly EXPECTED_SAMPLES (64400) so the SSL
    # encoder produces exactly 400 frames to match the KANAASIST positional encoding.
    if len(audio_array) > EXPECTED_SAMPLES:
        audio_array = audio_array[:EXPECTED_SAMPLES]
    elif len(audio_array) < EXPECTED_SAMPLES:
        audio_array = np.pad(audio_array, (0, EXPECTED_SAMPLES - len(audio_array)), 'constant')

    waveform = torch.from_numpy(audio_array).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)
    
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=-1)
        
    # Class 0 = bonafide (genuine), class 1 = spoof (fake)
    real_prob = float(probs[0, 0].item())
    fake_prob = float(probs[0, 1].item())
    
    is_fake = fake_prob >= 0.5
    label = "synthetic" if is_fake else "genuine"
    confidence = fake_prob if is_fake else real_prob
    
    return {
        "voice_integrity_score": fake_prob,
        "label": label,
        "model_name": "Spectra-AASIST3",
        "confidence": confidence
    }
