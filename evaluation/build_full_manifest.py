"""Consolidate all 21 local labeled audio samples into a comprehensive manifest."""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_manifest() -> Path:
    audio_base = PROJECT_ROOT / "evaluation" / "benchmark_data" / "audio"
    audio_unseen = PROJECT_ROOT / "evaluation" / "benchmark_data" / "unseen_audio"
    output_manifest = PROJECT_ROOT / "evaluation" / "benchmark_data" / "all_local_manifest.csv"

    rows = []

    # 1. Base files (12 files)
    for p in sorted(audio_base.glob("*.wav")):
        name = p.stem
        parts = name.split("_")
        spk = parts[0] + "_" + parts[1]  # e.g. SPK_001
        label = parts[2]                 # bonafide or spoof
        idx = parts[3]
        attack = "-" if label == "bonafide" else ("TTS-VITS" if idx == "2" else "VC-DiffVC")
        rows.append({
            "file_path": str(p.resolve()),
            "label": label,
            "speaker_id": spk,
            "attack_type": attack,
            "language": "en",
            "split": "unassigned"
        })

    # 2. Unseen files (9 files)
    for p in sorted(audio_unseen.glob("*.wav")):
        name = p.stem
        # e.g. SPK_001_unseen_spoof_1_TTS-ChatTTS-Style.wav
        parts = name.split("_")
        spk = parts[0] + "_" + parts[1]
        label = "spoof"
        attack = "_".join(parts[5:]) if len(parts) > 5 else parts[-1]
        rows.append({
            "file_path": str(p.resolve()),
            "label": label,
            "speaker_id": spk,
            "attack_type": attack,
            "language": "en",
            "split": "unassigned"
        })

    with open(output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "label", "speaker_id", "attack_type", "language", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to {output_manifest} with {len(rows)} samples.")
    return output_manifest

if __name__ == "__main__":
    build_manifest()
