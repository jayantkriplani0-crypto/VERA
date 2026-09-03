from pathlib import Path
import sys
import time
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.acquire_asvspoof_dev import RangeZipReader, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE
base_dir = PROJECT_ROOT / "data" / "asvspoof2019"
proto_dir = base_dir / "protocols"
dev_flac_dir = base_dir / "dev" / "flac"

dev_proto_file = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
with open(dev_proto_file, "r", encoding="utf-8") as f:
    dev_lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]

# Core speakers: LA_0069 through LA_0078
core_speakers = [f"LA_{i:04d}" for i in range(69, 79)]
target_stems = []

for spk in core_speakers:
    spk_lines = [l for l in dev_lines if l[0] == spk]
    bf_stems = [l[1] for l in spk_lines if l[4] == "bonafide"][:10]
    spf_stems = [l[1] for l in spk_lines if l[4] == "spoof"][:10]
    for stem in bf_stems + spf_stems:
        flac_name = f"{stem}.flac"
        if not (dev_flac_dir / flac_name).is_file():
            target_stems.append((spk, stem))

print(f"Targeting {len(target_stems)} new audio files across core development speakers...")

if target_stems:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})
    reader = RangeZipReader(session, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE)

    t0 = time.perf_counter()
    for idx, (spk, stem) in enumerate(target_stems, 1):
        flac_name = f"{stem}.flac"
        target_path = dev_flac_dir / flac_name
        zinfo = reader.zf.getinfo(f"LA/ASVspoof2019_LA_dev/flac/{flac_name}")
        data = reader.extract_member(zinfo)
        target_path.write_bytes(data)
        if idx % 15 == 0 or idx == len(target_stems):
            print(f"  Acquired [{idx}/{len(target_stems)}] for speaker {spk}...")

    print(f"Acquired all {len(target_stems)} files in {time.perf_counter() - t0:.2f}s!")

total_flac = len(list(dev_flac_dir.glob("*.flac")))
print(f"Total FLAC files now in {dev_flac_dir}: {total_flac}")
