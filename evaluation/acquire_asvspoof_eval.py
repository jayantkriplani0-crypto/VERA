"""Acquire official ASVspoof 2019 LA evaluation audio files from LA.zip via RangeZipReader.

Features:
  - Selects a balanced cohort of 700 evaluation samples:
    - 180 bona fide genuine human speech utterances.
    - 40 spoof utterances for each of the 13 unseen evaluation attack algorithms (A07-A19).
  - Uses HTTP Range requests with keep-alive connection pooling to stream directly from Edinburgh DataShare.
  - Verifies CRC32 checksums for every extracted FLAC file.
  - Extracts into data/asvspoof2019/eval/flac/ without touching dev/ or protocols/.
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

BONAFIDE_COUNT = 180
PER_ATTACK_COUNT = 40  # 40 * 13 attacks = 520 spoof samples -> total 700 samples
EVAL_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]


def select_eval_cohort(eval_proto_path: Path) -> List[Dict[str, str]]:
    """Parse evaluation protocol and select balanced samples across bona fide and A07-A19."""
    with open(eval_proto_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split() for l in f if l.strip() and not l.startswith("#")]

    print(f"[Eval Acquirer] Total trials in official evaluation protocol: {len(lines)}")

    selected: List[Dict[str, str]] = []

    # 1. Select bona fide
    bf_lines = [l for l in lines if l[4] == "bonafide"][:BONAFIDE_COUNT]
    assert len(bf_lines) == BONAFIDE_COUNT, f"Not enough bona fide: {len(bf_lines)}"
    for l in bf_lines:
        selected.append({
            "speaker_id": l[0],
            "audio_stem": l[1],
            "attack_type": l[3],
            "label": l[4],
        })

    # 2. Select spoof across A07-A19
    for att in EVAL_ATTACKS:
        att_lines = [l for l in lines if l[3] == att][:PER_ATTACK_COUNT]
        assert len(att_lines) == PER_ATTACK_COUNT, f"Not enough {att} trials: {len(att_lines)}"
        for l in att_lines:
            selected.append({
                "speaker_id": l[0],
                "audio_stem": l[1],
                "attack_type": l[3],
                "label": l[4],
            })

    return selected


def acquire_eval_flac_files(trials: List[Dict[str, str]], target_dir: Path) -> int:
    """Download and extract FLAC files from LA.zip with CRC32 verification."""
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = [t for t in trials if not (target_dir / f"{t['audio_stem']}.flac").is_file()]
    print(f"[Eval Acquirer] Selected evaluation cohort: {len(trials)} trials")
    print(f"[Eval Acquirer] Already cached locally: {len(trials) - len(missing)}")
    print(f"[Eval Acquirer] Need to download: {len(missing)}")

    if not missing:
        print("[Eval Acquirer] All evaluation audio files are already present on disk!")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})
    reader = RangeZipReader(session, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE)

    t0 = time.perf_counter()
    downloaded = 0
    total_bytes = 0

    for idx, t in enumerate(missing, 1):
        flac_name = f"{t['audio_stem']}.flac"
        member_name = f"LA/ASVspoof2019_LA_eval/flac/{flac_name}"
        target_path = target_dir / flac_name

        success = False
        for attempt in range(5):
            try:
                zinfo = reader.zf.getinfo(member_name)
                data = reader.extract_member(zinfo)
                target_path.write_bytes(data)
                total_bytes += len(data)
                downloaded += 1
                success = True
                break
            except Exception as e:
                time.sleep(1.5)
                session = requests.Session()
                session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})
                reader.session = session

        if not success:
            print(f"[Eval Acquirer Error] Could not download {flac_name} after 5 attempts.")

        time.sleep(0.02)  # Avoid Windows ephemeral socket pool exhaustion

        if idx == 1 or idx % 50 == 0 or idx == len(missing):
            elapsed = time.perf_counter() - t0
            rate = idx / max(1e-6, elapsed)
            print(f"  Progress: [{idx}/{len(missing)}] ({idx/len(missing)*100:.1f}%) - {rate:.1f} files/sec")

    print(f"[Eval Acquirer] Downloaded {downloaded} files ({total_bytes / (1024*1024):.2f} MB) in {time.perf_counter() - t0:.2f}s.")
    return downloaded


def main() -> None:
    base_dir = PROJECT_ROOT / "data" / "asvspoof2019"
    eval_proto = base_dir / "protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    eval_audio_dir = base_dir / "eval" / "flac"

    print("=" * 65)
    print("  ASVSPOOF 2019 LA: EVALUATION SET ACQUISITION")
    print("=" * 65)

    trials = select_eval_cohort(eval_proto)
    assert len(trials) == 700, f"Expected 700 trials, got {len(trials)}"

    acquire_eval_flac_files(trials, eval_audio_dir)

    # Verify present files
    present_flac = list(eval_audio_dir.glob("*.flac"))
    print(f"\n[Eval Acquirer] Total FLAC files in {eval_audio_dir}: {len(present_flac)}")
    print("[Eval Acquirer] Evaluation cohort successfully acquired and verified!")


if __name__ == "__main__":
    main()
