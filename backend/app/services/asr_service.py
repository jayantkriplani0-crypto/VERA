import numpy as np

_processor = None
_model = None
_device = None

def get_model():
    global _processor, _model, _device
    if _model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "openai/whisper-tiny"
        
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        _model.to(_device)
        _model.eval()
    return _processor, _model, _device

def transcribe_audio(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    if len(audio_array) == 0:
        return {
            "transcript": "",
            "language": "unknown",
            "model_name": "openai/whisper-tiny",
            "confidence": 0.0
        }
        
    processor, model, device = get_model()
    import torch

    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return {
        "transcript": transcript,
        "language": "auto", 
        "model_name": "openai/whisper-tiny",
        "confidence": None 
    }
