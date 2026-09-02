import numpy as np

_extractor = None
_model = None
_device = None

def get_model():
    global _extractor, _model, _device
    if _model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "anton-l/wav2vec2-base-superb-sv"
        
        _extractor = AutoFeatureExtractor.from_pretrained(model_id)
        _model = AutoModelForAudioXVector.from_pretrained(model_id)
        _model.to(_device)
        _model.eval()
    return _extractor, _model, _device

def extract_embedding(audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    if len(audio_array) < 400:
        audio_array = np.pad(audio_array, (0, 400 - len(audio_array)), 'constant')
        
    extractor, model, device = get_model()
    import torch
    
    inputs = extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        emb = model(**inputs).embeddings
        
    return emb.cpu().numpy()

def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    import torch
    import torch.nn.functional as F
    
    t1 = torch.from_numpy(emb1)
    t2 = torch.from_numpy(emb2)
    
    similarity = F.cosine_similarity(t1, t2).item()
    score = float((similarity + 1.0) / 2.0)
    
    threshold = 0.85
    is_match = score >= threshold
    confidence = float(min(1.0, abs(score - threshold) * 5 + 0.5))
    
    return {
        "speaker_similarity_score": score,
        "match": is_match,
        "model_name": "anton-l/wav2vec2-base-superb-sv",
        "confidence": confidence
    }

def verify_speaker(audio_array1: np.ndarray, audio_array2: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Compares two audio arrays and returns a speaker similarity score.
    """
    if len(audio_array1) == 0 or len(audio_array2) == 0:
        return {
            "speaker_similarity_score": 0.0,
            "match": False,
            "model_name": "anton-l/wav2vec2-base-superb-sv",
            "confidence": 0.0
        }
        
    emb1 = extract_embedding(audio_array1, sample_rate)
    emb2 = extract_embedding(audio_array2, sample_rate)
    
    return compare_embeddings(emb1, emb2)
