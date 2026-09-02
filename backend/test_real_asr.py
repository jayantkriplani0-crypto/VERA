import numpy as np
import time
import urllib.request
import tempfile
import os
import librosa
from app.services.asr_service import transcribe_audio

print("Downloading short speech sample...")
url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
    urllib.request.urlretrieve(url, f.name)
    audio_path = f.name

print("Loading audio and normalizing to 16 kHz mono...")
y, sr = librosa.load(audio_path, sr=16000, mono=True)
os.remove(audio_path)

print(f"Audio shape: {y.shape}, Length: {len(y)/sr:.2f} seconds")

print("\nLoading model and running ASR inference...")
start_time = time.time()
result = transcribe_audio(y, sr)
end_time = time.time()

print("\n--- INFERENCE RESULTS ---")
print("model loaded successfully:  Yes")
print(f"inference_time:             {end_time - start_time:.3f} seconds")
print(f"transcript:                 {result['transcript']}")
print(f"language:                   {result['language']}")
print(f"confidence:                 {result['confidence']}")
print(f"model_name:                 {result['model_name']}")
print("-------------------------")
