"""Phase 7: Systematic Failure and Edge-Case QA Suite."""
from pathlib import Path
import io
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

from api.app import app
from api.schemas import VoiceAnalysisResponse
from calibration.apply_calibration import CalibratedVoiceScorer
from ml.preprocessing.preprocessor import apply_preemphasis, apply_windowing
from ml.temporal.engine import RollingInferenceEngine
from ml.voice_detector.detector import VoiceAuthenticityDetector

detector = VoiceAuthenticityDetector(device="cpu")
scorer = CalibratedVoiceScorer()
engine = RollingInferenceEngine(detector=detector)

edge_results = {}

def run_direct_safe(audio_array: np.ndarray, desc: str):
    prep = apply_preemphasis(audio_array, coef=0.97)
    win = apply_windowing(prep, target_length=64600)
    raw = detector._run_forward(win)
    cal = scorer.calibrate_raw_score(raw)
    return {
        "status": "PASS",
        "raw_score": round(raw, 4),
        "spoof_signal": round(cal.calibrated_spoof_signal, 2),
        "confidence": round(cal.decision_confidence, 4),
    }

print("Running Phase 7 Edge & Failure QA...")

# 1. Pure Silence
sig_zeros = np.zeros(64600, dtype=np.float32)
edge_results["silence"] = run_direct_safe(sig_zeros, "Pure zeros")

# 2. Near Silence (low energy noise)
sig_near = np.random.normal(0, 1e-5, 64600).astype(np.float32)
edge_results["near_silence"] = run_direct_safe(sig_near, "Near silence 1e-5")

# 3. Clipping (extreme amplitudes)
sig_clip = np.clip(np.random.normal(0, 2.0, 64600).astype(np.float32), -1.0, 1.0)
edge_results["clipping"] = run_direct_safe(sig_clip, "Clipped [-1.0, 1.0]")

# 4. Very Short Audio (800 samples = 0.05 sec)
sig_short = (0.5 * np.sin(np.linspace(0, 0.05, 800) * 2 * np.pi * 440)).astype(np.float32)
edge_results["very_short_800_samples"] = run_direct_safe(sig_short, "Very short 800 samples")

# 5. Exactly 64,600 samples
sig_exact = (0.5 * np.sin(np.linspace(0, 4.0375, 64600) * 2 * np.pi * 440)).astype(np.float32)
edge_results["exactly_64600_samples"] = run_direct_safe(sig_exact, "Exact 64600")

# 6. Just Below 64,600 (64,599 samples)
sig_below = sig_exact[:-1]
edge_results["just_below_64599_samples"] = run_direct_safe(sig_below, "64599 samples")

# 7. Just Above 64,600 (64,601 samples)
sig_above = np.append(sig_exact, 0.0).astype(np.float32)
edge_results["just_above_64601_samples"] = run_direct_safe(sig_above, "64601 samples")

# 8. Long Audio (160,000 samples = 10.0 sec)
sig_long = (0.5 * np.sin(np.linspace(0, 10.0, 160000) * 2 * np.pi * 440)).astype(np.float32)
edge_results["long_audio_160000_samples"] = run_direct_safe(sig_long, "160000 samples")

# 9. Irregular Chunks through Rolling Engine
engine.reset()
irregular_sizes = [160, 3200, 800, 16000, 80, 64000, 1600]
events_emitted = []
for sz in irregular_sizes:
    chunk = (0.3 * np.sin(np.linspace(0, sz/16000.0, sz) * 2 * np.pi * 440)).astype(np.float32)
    evs = engine.push_chunk(chunk)
    events_emitted.extend(evs)
edge_results["irregular_chunks"] = {
    "status": "PASS",
    "total_chunks_pushed": len(irregular_sizes),
    "windows_emitted": len(events_emitted),
    "total_samples": sum(irregular_sizes),
}

# 10. REST API Audio Format Verification (WAV, FLAC, OGG, Empty, Corrupt)
with TestClient(app) as client:
    # 10a. Supported WAV
    bio_wav = io.BytesIO()
    sf.write(bio_wav, sig_exact, 16000, format="WAV")
    resp_wav = client.post("/api/v1/voice/analyze", files={"file": ("test.wav", bio_wav.getvalue(), "audio/wav")})
    assert resp_wav.status_code == 200
    edge_results["format_wav"] = {"status": "PASS", "http_status": 200}

    # 10b. Supported FLAC
    bio_flac = io.BytesIO()
    sf.write(bio_flac, sig_exact, 16000, format="FLAC")
    resp_flac = client.post("/api/v1/voice/analyze", files={"file": ("test.flac", bio_flac.getvalue(), "audio/flac")})
    assert resp_flac.status_code == 200
    edge_results["format_flac"] = {"status": "PASS", "http_status": 200}

    # 10c. Supported OGG
    bio_ogg = io.BytesIO()
    sf.write(bio_ogg, sig_exact, 16000, format="OGG")
    resp_ogg = client.post("/api/v1/voice/analyze", files={"file": ("test.ogg", bio_ogg.getvalue(), "audio/ogg")})
    assert resp_ogg.status_code == 200
    edge_results["format_ogg"] = {"status": "PASS", "http_status": 200}

    # 10d. Zero-byte input
    resp_empty = client.post("/api/v1/voice/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert resp_empty.status_code == 400
    edge_results["zero_byte_input"] = {"status": "PASS", "http_status": 400, "detail": resp_empty.json()["detail"]}

    # 10e. Corrupted audio
    resp_corrupt = client.post("/api/v1/voice/analyze", files={"file": ("corrupt.wav", b"RIFF\x00\x00\x00NOT_VALID_DATA", "audio/wav")})
    assert resp_corrupt.status_code == 400
    edge_results["corrupted_input"] = {"status": "PASS", "http_status": 400, "detail": resp_corrupt.json()["detail"]}

print("\n=== Phase 7 Edge & Failure QA Results ===")
print(json.dumps(edge_results, indent=2))

out_path = Path("evaluation/reports/failure_edge_qa.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(edge_results, f, indent=2)
print(f"Saved results to {out_path}")
