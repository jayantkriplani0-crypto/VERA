"""Phase 6: Determinism QA across 5 repeated runs of Direct, REST, and Streaming."""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from starlette.testclient import TestClient

from api.app import app
from api.schemas import StreamingVoiceEvent, VoiceAnalysisResponse
from calibration.apply_calibration import CalibratedVoiceScorer
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.preprocessor import apply_preemphasis, apply_windowing
from ml.voice_detector.detector import VoiceAuthenticityDetector

sample_path = Path("data/asvspoof2019/eval/flac/LA_E_5849185.flac")
waveform, _ = load_audio(sample_path)
with open(sample_path, "rb") as f:
    audio_bytes = f.read()

detector = VoiceAuthenticityDetector(device="cpu")
scorer = CalibratedVoiceScorer()

# Preprocess for direct inference
prep = apply_preemphasis(waveform, coef=0.97)
win = apply_windowing(prep, target_length=64600)

pcm_bytes = (waveform * 32767.0).astype(np.int16).tobytes()

direct_runs = []
rest_runs = []
stream_runs = []

with TestClient(app) as client:
    for i in range(5):
        # 1. Direct inference
        raw = detector._run_forward(win)
        cal = scorer.calibrate_raw_score(raw)
        direct_runs.append({
            "raw": raw,
            "p_bonafide": cal.calibrated_probability_bonafide,
            "spoof_signal": cal.calibrated_spoof_signal,
        })

        # 2. REST inference
        files = {"file": ("test.flac", audio_bytes, "audio/flac")}
        resp = client.post("/api/v1/voice/analyze", files=files)
        data = VoiceAnalysisResponse(**resp.json())
        rest_runs.append({
            "raw": data.raw_bona_fide_logit,
            "p_bonafide": data.calibrated_bona_fide_probability,
            "spoof_signal": data.spoof_signal,
            "classification": data.classification,
        })

        # 3. Streaming simulation
        with client.websocket_connect("/api/v1/voice/stream") as ws:
            ws.send_bytes(pcm_bytes)
            ws.send_text('{"action": "flush"}')
            ev_raw = ws.receive_text()
            ev = StreamingVoiceEvent.model_validate_json(ev_raw)
            stream_runs.append({
                "raw": ev.raw_logit,
                "p_bonafide": ev.p_bonafide,
                "spoof_signal": ev.spoof_signal,
                "state": ev.state,
            })

# Compute maximum differences across the 5 runs
direct_raws = [r["raw"] for r in direct_runs]
direct_probs = [r["p_bonafide"] for r in direct_runs]
max_diff_direct_raw = max(direct_raws) - min(direct_raws)
max_diff_direct_prob = max(direct_probs) - min(direct_probs)

rest_raws = [r["raw"] for r in rest_runs]
rest_probs = [r["p_bonafide"] for r in rest_runs]
max_diff_rest_raw = max(rest_raws) - min(rest_raws)
max_diff_rest_prob = max(rest_probs) - min(rest_probs)
rest_classifications = set(r["classification"] for r in rest_runs)

stream_raws = [r["raw"] for r in stream_runs]
stream_probs = [r["p_bonafide"] for r in stream_runs]
max_diff_stream_raw = max(stream_raws) - min(stream_raws)
max_diff_stream_prob = max(stream_probs) - min(stream_probs)
stream_states = set(r["state"] for r in stream_runs)

report = {
    "direct_5_runs": {
        "max_logit_difference": max_diff_direct_raw,
        "max_prob_difference": max_diff_direct_prob,
        "is_deterministic": max_diff_direct_raw == 0.0,
    },
    "rest_5_runs": {
        "max_logit_difference": max_diff_rest_raw,
        "max_prob_difference": max_diff_rest_prob,
        "classifications": list(rest_classifications),
        "state_divergence": len(rest_classifications) > 1,
        "is_deterministic": max_diff_rest_raw == 0.0,
    },
    "stream_5_runs": {
        "max_logit_difference": max_diff_stream_raw,
        "max_prob_difference": max_diff_stream_prob,
        "states": list(stream_states),
        "state_divergence": len(stream_states) > 1,
        "is_deterministic": max_diff_stream_raw == 0.0,
    },
}

print("=== Phase 6 Determinism QA Report ===")
print(json.dumps(report, indent=2))

out_path = Path("evaluation/reports/determinism_qa.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Saved report to {out_path}")
