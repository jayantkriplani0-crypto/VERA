import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
import torch

from ml.preprocessing.preprocessor import preprocess_waveform, apply_preemphasis, apply_windowing
from ml.temporal.engine import RollingInferenceEngine
from ml.temporal.rolling_buffer import RollingAudioBuffer, WINDOW_SIZE_SAMPLES, DEFAULT_HOP_SIZE_SAMPLES
from ml.temporal.smoother import ExponentialMovingAverageSmoother
from ml.temporal.state_machine import HysteresisStateMachine
from ml.voice_detector.detector import VoiceAuthenticityDetector
from calibration.apply_calibration import CalibratedVoiceScorer


def profile_pipeline():
    print("=" * 65)
    print("  VERA LAYER 1: COMPONENT-BY-COMPONENT LATENCY PROFILING")
    print("=" * 65)

    # 1. Model Load Time & RAM
    p = psutil.Process(os.getpid())
    ram_before_load = p.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    detector = VoiceAuthenticityDetector(device="cpu")
    t_load = time.perf_counter() - t0
    ram_after_load = p.memory_info().rss / (1024 * 1024)
    print(f"1. Model Load Time: {t_load:.3f} s (RAM: {ram_before_load:.1f} -> {ram_after_load:.1f} MB, +{ram_after_load - ram_before_load:.1f} MB)")

    # 2. Calibrator Load Time
    t0 = time.perf_counter()
    scorer = CalibratedVoiceScorer()
    t_cal_load = (time.perf_counter() - t0) * 1000.0
    print(f"2. Calibrator Load Time: {t_cal_load:.3f} ms")

    # 3. First Inference Latency (Cold Start / JIT / Warmup)
    dummy_audio = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32)
    t0 = time.perf_counter()
    preprocessed_dummy = preprocess_waveform(dummy_audio)
    score_cold = detector._run_forward(preprocessed_dummy)
    t_cold = (time.perf_counter() - t0) * 1000.0
    print(f"3. First Inference Latency (Cold): {t_cold:.2f} ms (Raw score: {score_cold:+.4f})")

    # 4. Detailed Component Timings over N iterations
    N = 10
    print(f"\nRunning {N} iterations for component breakdown:")

    # Component A: Preprocessing Breakdown
    times_preemph = []
    times_windowing = []
    times_total_prep = []

    for _ in range(N):
        raw_w = np.random.randn(WINDOW_SIZE_SAMPLES).astype(np.float32)

        t_start = time.perf_counter()
        t0 = time.perf_counter()
        pe = apply_preemphasis(raw_w, coef=0.97)
        times_preemph.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        w = apply_windowing(pe, target_length=WINDOW_SIZE_SAMPLES)
        times_windowing.append((time.perf_counter() - t0) * 1000.0)
        times_total_prep.append((time.perf_counter() - t_start) * 1000.0)

    print(f"  A. Preprocessing:")
    print(f"     - apply_preemphasis: {np.mean(times_preemph):.3f} ms ± {np.std(times_preemph):.3f} ms")
    print(f"     - enforce_window_len: {np.mean(times_windowing):.3f} ms ± {np.std(times_windowing):.3f} ms")
    print(f"     - total prep:        {np.mean(times_total_prep):.3f} ms")

    # Component B: Forward Pass (Neural Network on CPU)
    times_forward = []
    for _ in range(N):
        t0 = time.perf_counter()
        s = detector._run_forward(w)
        times_forward.append((time.perf_counter() - t0) * 1000.0)
    print(f"  B. Model Forward Pass: {np.mean(times_forward):.2f} ms ± {np.std(times_forward):.2f} ms (p50: {np.median(times_forward):.2f} ms)")

    # Component C: Calibration
    times_cal = []
    for _ in range(N):
        t0 = time.perf_counter()
        res = scorer.calibrate_raw_score(s)
        times_cal.append((time.perf_counter() - t0) * 1000.0)
    print(f"  C. Platt Calibration:  {np.mean(times_cal):.4f} ms ± {np.std(times_cal):.4f} ms")

    # Component D: Smoother (EMA)
    smoother = ExponentialMovingAverageSmoother(alpha=0.35)
    times_smooth = []
    for _ in range(N):
        t0 = time.perf_counter()
        sm_s = smoother.update(res.calibrated_spoof_signal)
        times_smooth.append((time.perf_counter() - t0) * 1000.0)
    print(f"  D. EMA Smoothing:      {np.mean(times_smooth):.4f} ms ± {np.std(times_smooth):.4f} ms")

    # Component E: Hysteresis State Machine
    state_machine = HysteresisStateMachine()
    times_state = []
    for _ in range(N):
        t0 = time.perf_counter()
        st, chg = state_machine.update(sm_s)
        times_state.append((time.perf_counter() - t0) * 1000.0)
    print(f"  E. Hysteresis Machine: {np.mean(times_state):.4f} ms ± {np.std(times_state):.4f} ms")

    # Component F: Rolling Buffer Append / Slicing
    buf = RollingAudioBuffer()
    chunk_500ms = np.random.randn(8000).astype(np.float32)
    times_buf = []
    for _ in range(N * 4):
        t0 = time.perf_counter()
        wins = buf.append(chunk_500ms)
        times_buf.append((time.perf_counter() - t0) * 1000.0)
    print(f"  F. Rolling Buffer:     {np.mean(times_buf):.4f} ms ± {np.std(times_buf):.4f} ms")

    # Threading exploration
    current_threads = torch.get_num_threads()
    print(f"\nTorch current CPU threads: {current_threads}")
    for num_t in [1, 2, 4, 6, 8, current_threads]:
        if num_t > os.cpu_count():
            continue
        torch.set_num_threads(num_t)
        t_threads = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = detector._run_forward(w)
            t_threads.append((time.perf_counter() - t0) * 1000.0)
        print(f"  -> Threads={num_t}: {np.mean(t_threads):.2f} ms ± {np.std(t_threads):.2f} ms")

    torch.set_num_threads(current_threads)


if __name__ == "__main__":
    profile_pipeline()
