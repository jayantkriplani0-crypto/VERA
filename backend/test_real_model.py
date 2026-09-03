import numpy as np
import time
from app.services.voice_integrity_service import analyze_voice

print("Loading model and running inference...")

sample_rate = 16000
audio_array = np.zeros(sample_rate, dtype=np.float32)

start_time = time.time()
result = analyze_voice(audio_array, sample_rate)
end_time = time.time()

print("\n--- INFERENCE RESULTS ---")
print(f"model loaded successfully")
print(f"inference_time:        {end_time - start_time:.3f} seconds")
print(f"voice_integrity_score: {result['voice_integrity_score']}")
print(f"label:                 {result['label']}")
print(f"confidence:            {result['confidence']}")
print(f"model_name:            {result['model_name']}")
print("-------------------------")
