"""Build and acquire the 700-sample ASVspoof 2019 LA development calibration cohort.

Features:
  - Exact balanced selection across all 10 core development speakers.
  - Strict speaker-disjoint partition: cal_fit (6 speakers, 420 samples) vs cal_val (4 speakers, 280 samples).
  - Exact attack distribution: 28 bona fide, 7 of each A01-A06 per speaker.
  - Downloads any missing audio files using RangeZipReader with keep-alive connection pooling.
  - Saves reproducible CSV manifests for the combined cohort and individual splits.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.acquire_asvspoof_dev import RangeZipReader, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE

FIT_SPEAKERS = ["LA_0069", "LA_0070", "LA_0071", "LA_0072", "LA_0073", "LA_0074"]
VAL_SPEAKERS = ["LA_0075", "LA_0076", "LA_0077", "LA_0078"]
ALL_SPEAKERS = FIT_SPEAKERS + VAL_SPEAKERS

BONAFIDE_PER_SPEAKER = 28
PER_ATTACK_PER_SPEAKER = 7  # 7 * 6 = 42 spoof per speaker -> total 70 per speaker


def select_cohort_trials(proto_path: Path) -> List[Dict[str, Any]]:
    """Parse protocol and select the exact balanced 700 trials."""
    with open(proto_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

    selected_trials: List[Dict[str, Any]] = []

    for spk in ALL_SPEAKERS:
        spk_lines = [l for l in lines if l[0] == spk]
        split_role = "cal_fit" if spk in FIT_SPEAKERS else "cal_val"

        # 1. Select bona fide
        bf_lines = [l for l in spk_lines if l[4] == "bonafide"][:BONAFIDE_PER_SPEAKER]
        assert len(bf_lines) == BONAFIDE_PER_SPEAKER, f"Not enough bona fide for {spk}: {len(bf_lines)}"
        for l in bf_lines:
            selected_trials.append({
                "speaker_id": l[0],
                "audio_stem": l[1],
                "attack_type": l[3],
                "label": l[4],
                "split": split_role,
            })

        # 2. Select spoof for each attack A01-A06
        for att in [f"A{i:02d}" for i in range(1, 7)]:
            att_lines = [l for l in spk_lines if l[3] == att][:PER_ATTACK_PER_SPEAKER]
            assert len(att_lines) == PER_ATTACK_PER_SPEAKER, f"Not enough {att} for {spk}: {len(att_lines)}"
            for l in att_lines:
                selected_trials.append({
                    "speaker_id": l[0],
                    "audio_stem": l[1],
                    "attack_type": l[3],
                    "label": l[4],
                    "split": split_role,
                })

    return selected_trials


def acquire_missing_audio(trials: List[Dict[str, Any]], audio_dir: Path) -> int:
    """Download any trials that do not already have local FLAC files."""
    missing = [t for t in trials if not (audio_dir / f"{t['audio_stem']}.flac").is_file()]
    print(f"[Cohort Builder] Total cohort trials: {len(trials)}")
    print(f"[Cohort Builder] Already cached locally: {len(trials) - len(missing)}")
    print(f"[Cohort Builder] Need to download: {len(missing)}")

    if not missing:
        print("[Cohort Builder] All 700 audio files are already present on disk!")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})
    reader = RangeZipReader(session, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE)

    t0 = time.perf_counter()
    downloaded = 0
    total_bytes = 0

    for idx, t in enumerate(missing, 1):
        flac_name = f"{t['audio_stem']}.flac"
        target_path = audio_dir / flac_name
        zinfo = reader.zf.getinfo(f"LA/ASVspoof2019_LA_dev/flac/{flac_name}")
        data = reader.extract_member(zinfo)
        target_path.write_bytes(data)
        total_bytes += len(data)
        downloaded += 1

        if idx % 50 == 0 or idx == len(missing):
            elapsed = time.perf_counter() - t0
            rate = idx / max(1e-6, elapsed)
            print(f"  Downloaded [{idx}/{len(missing)}] ({idx/len(missing)*100:.1f}%) - {rate:.1f} files/sec")

    print(f"[Cohort Builder] Downloaded {downloaded} files ({total_bytes / (1024*1024):.2f} MB) in {time.perf_counter() - t0:.2f}s.")
    return downloaded


def save_manifests(trials: List[Dict[str, Any]], audio_dir: Path, manifests_dir: Path) -> None:
    """Save combined and split CSV manifests with resolved file paths."""
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for t in trials:
        file_path = (audio_dir / f"{t['audio_stem']}.flac").resolve()
        assert file_path.is_file(), f"Audio file not found: {file_path}"
        records.append({
            "file_path": str(file_path),
            "filename": f"{t['audio_stem']}.flac",
            "speaker_id": t["speaker_id"],
            "attack_type": t["attack_type"],
            "label": t["label"],
            "ground_truth_binary": 1 if t["label"] == "bonafide" else 0,
            "split": t["split"],
        })

    # Assert strict speaker disjointness
    fit_spks = set(r["speaker_id"] for r in records if r["split"] == "cal_fit")
    val_spks = set(r["speaker_id"] for r in records if r["split"] == "cal_val")
    assert fit_spks.isdisjoint(val_spks), f"Speaker leakage detected! Overlap: {fit_spks & val_spks}"
    print(f"[Cohort Builder] Speaker disjointness verified! Fit speakers: {fit_spks}, Val speakers: {val_spks}")

    fieldnames = ["file_path", "filename", "speaker_id", "attack_type", "label", "ground_truth_binary", "split"]

    # 1. Combined manifest
    comb_path = manifests_dir / "calibration_cohort_manifest.csv"
    with open(comb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[Cohort Builder] Saved combined manifest ({len(records)} rows) to: {comb_path}")

    # 2. Calibration Fit manifest
    fit_records = [r for r in records if r["split"] == "cal_fit"]
    fit_path = manifests_dir / "cal_fit_manifest.csv"
    with open(fit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fit_records)
    print(f"[Cohort Builder] Saved cal_fit manifest ({len(fit_records)} rows) to: {fit_path}")

    # 3. Calibration Val manifest
    val_records = [r for r in records if r["split"] == "cal_val"]
    val_path = manifests_dir / "cal_val_manifest.csv"
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_records)
    print(f"[Cohort Builder] Saved cal_val manifest ({len(val_records)} rows) to: {val_path}")


def main() -> None:
    base_dir = PROJECT_ROOT / "data" / "asvspoof2019"
    proto_path = base_dir / "protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    audio_dir = base_dir / "dev" / "flac"
    manifests_dir = base_dir / "manifests"

    print("=" * 65)
    print("  ASVSPOOF 2019 LA: CALIBRATION COHORT BUILDER (700 SAMPLES)")
    print("=" * 65)

    trials = select_cohort_trials(proto_path)
    assert len(trials) == 700, f"Expected 700 trials, got {len(trials)}"
    print(f"[Cohort Builder] Selected {len(trials)} trials across {len(ALL_SPEAKERS)} speakers.")

    acquire_missing_audio(trials, audio_dir)
    save_manifests(trials, audio_dir, manifests_dir)
    print("\n[Cohort Builder] Cohort preparation complete and fully verified!")


if __name__ == "__main__":
    main()
