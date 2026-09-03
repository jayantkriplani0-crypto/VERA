"""Interactive demo client for the VERA Layer 1 Voice Authenticity API.

Demonstrates:
  1. Service health check: GET /api/v1/voice/health
  2. Single-file audio analysis: POST /api/v1/voice/analyze
  3. Continuous real-time streaming: WebSocket /api/v1/voice/stream
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
import sys
import time
import httpx
import numpy as np
import soundfile as sf

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_synthetic_wav(duration_sec: float = 5.0, sr: int = 16000) -> bytes:
    """Generate in-memory mono WAV containing synthetic speech-like harmonics."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), dtype=np.float32)
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t) +
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.1 * np.sin(2 * np.pi * 880 * t)
    )
    bio = io.BytesIO()
    sf.write(bio, signal, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def check_health(base_url: str) -> None:
    """Query health endpoint and display service status."""
    url = f"{base_url.rstrip('/')}/api/v1/voice/health"
    print(f"\n[DEMO] Querying Health Endpoint: {url}...")
    try:
        with httpx.Client() as client:
            resp = client.get(url, timeout=10.0)
            if resp.status_code == 200:
                print("✅ Service is Healthy!")
                print(json.dumps(resp.json(), indent=2))
            else:
                print(f"❌ Service returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")


def analyze_audio_file(base_url: str, file_path: Optional[Path] = None) -> None:
    """Upload audio file to /api/v1/voice/analyze and print structured result."""
    url = f"{base_url.rstrip('/')}/api/v1/voice/analyze"

    if file_path and file_path.is_file():
        print(f"\n[DEMO] Uploading '{file_path}' to {url}...")
        with open(file_path, "rb") as f:
            content = f.read()
        filename = file_path.name
    else:
        print(f"\n[DEMO] Generating 5.0s synthetic test audio and uploading to {url}...")
        content = create_synthetic_wav(duration_sec=5.0)
        filename = "synthetic_sample.wav"

    files = {"file": (filename, content, "audio/wav")}

    try:
        t0 = time.perf_counter()
        with httpx.Client() as client:
            resp = client.post(url, files=files, timeout=30.0)
        round_trip_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            res = resp.json()
            print("=" * 65)
            print("  VERA LAYER 1: VOICE AUTHENTICITY ANALYSIS RESULT")
            print("=" * 65)
            print(f"Request ID       : {res['request_id']}")
            print(f"Model            : {res['model']}")
            print(f"Classification   : {res['classification']}")
            print(f"Voice Integrity  : {res['voice_integrity_score']:.2f} / 100.0 (higher = genuine)")
            print(f"Spoof Signal     : {res['spoof_signal']:.2f} / 100.0 (higher = synthetic)")
            print(f"P(Bona Fide)     : {res['calibrated_bona_fide_probability']:.6f}")
            print(f"P(Spoof)         : {res['calibrated_spoof_probability']:.6f}")
            print(f"Raw Model Logit  : {res['raw_bona_fide_logit']:+.4f}")
            print(f"Decision Conf.   : {res['decision_confidence']:.4f}")
            print(f"Audio Duration   : {res['duration_seconds']:.2f} seconds")
            print(f"Server Compute   : {res['processing_latency_ms']:.1f} ms")
            print(f"Client RTT       : {round_trip_ms:.1f} ms")
            print("=" * 65)
        else:
            print(f"❌ Analysis failed (status {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")


async def stream_audio_websocket(ws_url: str, file_path: Optional[Path] = None, chunk_duration_sec: float = 0.5) -> None:
    """Stream audio chunks over WebSocket and print rolling inference events."""
    try:
        import websockets
    except ImportError:
        print("Optional package 'websockets' not found. Installing or using httpx...")
        return

    print(f"\n[DEMO] Connecting to WebSocket: {ws_url}...")

    # Load or synthesize audio
    if file_path and file_path.is_file():
        data, sr = sf.read(str(file_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    else:
        print("[DEMO] Synthesizing 10.0s multi-tone audio stream...")
        sr = 16000
        t = np.linspace(0, 10.0, int(10.0 * sr), dtype=np.float32)
        data = 0.5 * np.sin(2 * np.pi * 300 * t)

    chunk_samples = int(chunk_duration_sec * sr)
    num_chunks = int(np.ceil(len(data) / float(chunk_samples)))

    async with websockets.connect(ws_url) as ws:
        print(f"[DEMO] Connected! Streaming {num_chunks} chunks ({chunk_duration_sec*1000:.0f} ms each)...")

        async def sender():
            for i in range(num_chunks):
                chunk = data[i * chunk_samples : (i + 1) * chunk_samples]
                # Convert float32 to 16-bit PCM bytes
                pcm_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
                await ws.send(pcm_bytes)
                await asyncio.sleep(chunk_duration_sec)

            # Send flush control frame
            await ws.send(json.dumps({"action": "flush"}))

        async def receiver():
            try:
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    if "event" in event and event["event"] == "voice_authenticity":
                        print(
                            f"  -> Window #{event['window_id']} ({event['timestamp_start']:.2f}s-{event['timestamp_end']:.2f}s): "
                            f"State={event['state']} | SpoofSignal={event['spoof_signal']:.1f} | "
                            f"Smoothed={event['smoothed_spoof_signal']:.1f} | Conf={event['confidence']:.2f} | Latency={event['latency_ms']:.1f}ms"
                        )
            except Exception:
                pass

        await asyncio.gather(sender(), receiver())


def main():
    parser = argparse.ArgumentParser(description="VERA Layer 1 Voice Authenticity API Demo Client")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000", help="Base API URL")
    parser.add_argument("--mode", choices=["health", "analyze", "stream", "all"], default="all", help="Demo mode")
    parser.add_argument("--file", type=str, default=None, help="Path to audio file (optional)")
    args = parser.parse_args()

    audio_file = Path(args.file) if args.file else None

    if args.mode in ("health", "all"):
        check_health(args.base_url)

    if args.mode in ("analyze", "all"):
        analyze_audio_file(args.base_url, audio_file)

    if args.mode in ("stream", "all"):
        ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/api/v1/voice/stream"
        try:
            asyncio.run(stream_audio_websocket(ws_url, audio_file))
        except Exception as e:
            print(f"[DEMO] Note: WebSocket demo skipped or finished ({e})")


if __name__ == "__main__":
    main()
