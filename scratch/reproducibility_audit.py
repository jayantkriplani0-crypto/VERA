from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.preprocessor import apply_preemphasis, apply_windowing
from ml.voice_detector.detector import VoiceAuthenticityDetector
from calibration.apply_calibration import CalibratedVoiceScorer
from ml.temporal.engine import RollingInferenceEngine

detector = VoiceAuthenticityDetector(device="cpu")
scorer = CalibratedVoiceScorer()

samples = [
    ("Genuine", "data/asvspoof2019/eval/flac/LA_E_5849185.flac"),
    ("Spoof",   "data/asvspoof2019/eval/flac/LA_E_2834763.flac"),
    ("Edge",    "data/asvspoof2019/eval/flac/LA_E_1721212.flac"),
]

results = {}

for category, rel_path in samples:
    path = Path(rel_path)
    runs = []
    for run_idx in range(3):
        # 1. Direct Model + Calibrator Run
        waveform, _ = load_audio(path)
        prep = apply_preemphasis(waveform, coef=0.97)
        win = apply_windowing(prep, target_length=64600)
        raw_score = detector._run_forward(win)
        cal = scorer.calibrate_raw_score(raw_score)

        # 2. Temporal Streaming Engine Run
        engine = RollingInferenceEngine(detector=detector)
        events = list(engine.process_file_stream(path, chunk_size_sec=0.5))
        first_ev = events[0]

        runs.append({
            "raw_score": raw_score,
            "p_bonafide": cal.calibrated_probability_bonafide,
            "p_spoof": cal.calibrated_probability_spoof,
            "voice_integrity": cal.voice_integrity_score,
            "spoof_signal": cal.calibrated_spoof_signal,
            "confidence": cal.decision_confidence,
            "stream_first_raw": first_ev.raw_score,
            "stream_first_state": first_ev.state.value,
        })
    results[category] = runs

# Verify reproducibility
audit_summary = {}
for category, runs in results.items():
    print(f"\n=== {category} Sample ===")
    r0 = runs[0]
    r1 = runs[1]
    r2 = runs[2]
    diff_raw = max(abs(r0["raw_score"] - r1["raw_score"]), abs(r0["raw_score"] - r2["raw_score"]))
    diff_p = max(abs(r0["p_bonafide"] - r1["p_bonafide"]), abs(r0["p_bonafide"] - r2["p_bonafide"]))
    diff_sig = max(abs(r0["spoof_signal"] - r1["spoof_signal"]), abs(r0["spoof_signal"] - r2["spoof_signal"]))
    diff_integ = max(abs(r0["voice_integrity"] - r1["voice_integrity"]), abs(r0["voice_integrity"] - r2["voice_integrity"]))
    diff_conf = max(abs(r0["confidence"] - r1["confidence"]), abs(r0["confidence"] - r2["confidence"]))
    states = [r0["stream_first_state"], r1["stream_first_state"], r2["stream_first_state"]]

    print(f"  Raw Logit:    {r0['raw_score']:+.6f} (max diff across 3 runs: {diff_raw:.10f})")
    print(f"  P(BonaFide):   {r0['p_bonafide']:.8f} (max diff: {diff_p:.10f})")
    print(f"  Spoof Signal:  {r0['spoof_signal']:.2f} (max diff: {diff_sig:.10f})")
    print(f"  Integrity:     {r0['voice_integrity']:.2f} (max diff: {diff_integ:.10f})")
    print(f"  Confidence:    {r0['confidence']:.4f} (max diff: {diff_conf:.10f})")
    print(f"  Temporal State across runs: {states} (deterministic: {len(set(states)) == 1})")

    assert diff_raw < 1e-6
    assert diff_p < 1e-6
    assert diff_sig == 0.0
    assert diff_integ == 0.0
    assert diff_conf == 0.0
    assert len(set(states)) == 1

    audit_summary[category] = {
        "raw_score": r0["raw_score"],
        "p_bonafide": r0["p_bonafide"],
        "p_spoof": r0["p_spoof"],
        "spoof_signal": r0["spoof_signal"],
        "voice_integrity": r0["voice_integrity"],
        "confidence": r0["confidence"],
        "temporal_state": states[0],
        "max_discrepancy": max(diff_raw, diff_p, diff_sig),
        "reproducible": True,
    }

print("\n" + "=" * 65)
print("  PHASE 2 REPRODUCIBILITY AUDIT: 100% VERIFIED IDENTICAL")
print("=" * 65)

out_file = Path("evaluation/reports/reproducibility_audit.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2)
print(f"Saved audit results to {out_file}")
