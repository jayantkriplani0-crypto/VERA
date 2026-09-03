"""ASVspoof 2019 LA Acquisition Module.

Acquires official ASVspoof 2019 Logical Access (LA) protocol files and development
audio data directly from the official Edinburgh DataShare repository.
Uses HTTP Range requests and connection pooling for verified lossless extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import struct
import sys
import time
import zipfile
import zlib

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIRECT_CONTENT_URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
LA_ZIP_TOTAL_SIZE = 7640952520


class RangeZipReader:
    """Reads zip central directory and extracts files via HTTP Range requests with keep-alive."""

    def __init__(self, session: requests.Session, url: str = DIRECT_CONTENT_URL, total_size: int = LA_ZIP_TOTAL_SIZE):
        self.session = session
        self.url = url
        self.total_size = total_size
        self._init_central_directory()

    def _init_central_directory(self) -> None:
        fetch_size = 14 * 1024 * 1024
        start_byte = self.total_size - fetch_size
        cache_path = PROJECT_ROOT / "data" / "asvspoof2019" / ".la_zip_tail_14mb.bin"

        if cache_path.is_file() and cache_path.stat().st_size == fetch_size:
            print(f"[RangeZipReader] Loading cached central directory ({fetch_size / (1024*1024):.1f} MB) from disk...")
            tail_data = cache_path.read_bytes()
        else:
            print(f"[RangeZipReader] Fetching central directory ({fetch_size / (1024*1024):.1f} MB) from remote...")
            headers = {"Range": f"bytes={start_byte}-{self.total_size - 1}"}
            tail_data = b""
            for attempt in range(5):
                try:
                    resp = self.session.get(self.url, headers=headers, timeout=90)
                    resp.raise_for_status()
                    tail_data = resp.content
                    break
                except Exception as e:
                    print(f"[RangeZipReader] Central directory fetch attempt {attempt+1}/5 failed: {e}. Retrying in 3s...")
                    time.sleep(3)

            if len(tail_data) != fetch_size:
                raise RuntimeError(f"Failed to fetch {fetch_size} bytes of central directory after 5 attempts.")

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(tail_data)
            print(f"[RangeZipReader] Cached central directory to {cache_path}")

        class StreamWrapper(io.RawIOBase):
            def __init__(self, session, url, size, tail_offset, tail_bytes):
                self.session = session
                self.url = url
                self.size = size
                self.pos = 0
                self.tail_offset = tail_offset
                self.tail_bytes = tail_bytes

            def seekable(self):
                return True

            def seek(self, offset, whence=io.SEEK_SET):
                if whence == io.SEEK_SET:
                    self.pos = offset
                elif whence == io.SEEK_CUR:
                    self.pos += offset
                elif whence == io.SEEK_END:
                    self.pos = self.size + offset
                return self.pos

            def tell(self):
                return self.pos

            def read(self, size=-1):
                if size == -1 or size is None:
                    size = self.size - self.pos
                size = min(size, self.size - self.pos)
                if size <= 0:
                    return b""
                if self.pos >= self.tail_offset:
                    rel = self.pos - self.tail_offset
                    chunk = self.tail_bytes[rel : rel + size]
                    self.pos += len(chunk)
                    return chunk
                end = self.pos + size - 1
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
                r.raise_for_status()
                data = r.content
                self.pos += len(data)
                return data

        stream = StreamWrapper(self.session, self.url, self.total_size, start_byte, tail_data)
        self.zf = zipfile.ZipFile(stream)

    def extract_member(self, zinfo: zipfile.ZipInfo, max_retries: int = 3) -> bytes:
        """Fetch single member in 1 range request using the known header offset."""
        read_len = 300 + zinfo.compress_size
        range_header = f"bytes={zinfo.header_offset}-{zinfo.header_offset + read_len - 1}"

        for attempt in range(max_retries):
            try:
                resp = self.session.get(self.url, headers={"Range": range_header}, timeout=60)
                resp.raise_for_status()
                raw_chunk = resp.content

                sig, ver, flags, comp_method, mtime, mdate, crc32, comp_size, uncomp_size, name_len, extra_len = (
                    struct.unpack("<IHHHHHIIIHH", raw_chunk[:30])
                )
                assert sig == 0x04034B50, f"Invalid local header signature: {hex(sig)}"
                data_start = 30 + name_len + extra_len
                comp_bytes = raw_chunk[data_start : data_start + zinfo.compress_size]

                if comp_method == zipfile.ZIP_STORED:
                    uncomp = comp_bytes
                elif comp_method == zipfile.ZIP_DEFLATED:
                    uncomp = zlib.decompress(comp_bytes, -15)
                else:
                    raise ValueError(f"Unsupported compression method: {comp_method}")

                assert zlib.crc32(uncomp) == zinfo.CRC, f"CRC mismatch in {zinfo.filename}"
                return uncomp
            except Exception as err:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0)


def acquire_dev_data(
    num_samples: int = 100,
    base_dir: Path = PROJECT_ROOT / "data" / "asvspoof2019",
    balanced: bool = False,
    per_attack: int = 10,
) -> dict[str, Any]:
    """Acquire protocol files and first N development samples."""
    proto_dir = base_dir / "protocols"
    dev_flac_dir = base_dir / "dev" / "flac"
    proto_dir.mkdir(parents=True, exist_ok=True)
    dev_flac_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ASVSPOOF 2019 LA: OFFICIAL ACQUISITION PIPELINE (PHASE 1)")
    print("=" * 65)
    print(f"Source URL     : {DIRECT_CONTENT_URL}")
    print(f"Destination Dir: {base_dir}")
    print(f"Target Samples : {num_samples} development utterances")
    print("=" * 65)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})

    print("\n[Step 1] Connecting to Edinburgh DataShare & indexing central directory...")
    reader = RangeZipReader(session)
    print(f"  -> Successfully indexed {len(reader.zf.filelist)} archive members.")

    # 1. Acquire Protocols
    print("\n[Step 2] Verifying and acquiring official protocol files...")
    protocols = [
        "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    ]
    protocol_stats = []
    for p in protocols:
        fname = Path(p).name
        target = proto_dir / fname
        zinfo = reader.zf.getinfo(p)

        if target.is_file() and target.stat().st_size == zinfo.file_size:
            data = target.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            print(f"  [Cached] {fname} ({len(data):,} bytes, MD5: {md5})")
        else:
            print(f"  [Downloading] {fname} ({zinfo.file_size:,} bytes)...")
            data = reader.extract_member(zinfo)
            target.write_bytes(data)
            md5 = hashlib.md5(data).hexdigest()
            print(f"  -> Saved {fname} ({len(data):,} bytes, MD5: {md5})")

        protocol_stats.append({
            "name": fname,
            "size_bytes": len(data),
            "md5": md5,
            "path": str(target),
        })

    # 2. Parse Development Protocol
    dev_proto_file = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    with open(dev_proto_file, "r", encoding="utf-8") as f:
        dev_lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]

    total_dev_trials = len(dev_lines)
    print(f"\n[Step 3] Parsing development protocol: {total_dev_trials} total trials found.")

    if balanced:
        # Group by attack type to ensure representation of bonafide and all spoof types
        by_attack: dict[str, list[str]] = {}
        for parts in dev_lines:
            att = parts[3]
            by_attack.setdefault(att, []).append(parts[1])

        target_stems = []
        for att in sorted(by_attack.keys()):
            selected = by_attack[att][:per_attack]
            target_stems.extend(selected)
            print(f"  Selected {len(selected)} samples for category '{att}'")
    else:
        target_stems = [parts[1] for parts in dev_lines[:num_samples]]

    # 3. Acquire Audio Files
    print(f"\n[Step 4] Acquiring {len(target_stems)} development audio files into '{dev_flac_dir}'...")
    audio_stats = []
    total_audio_bytes = 0
    t0 = time.perf_counter()

    for idx, stem in enumerate(target_stems, 1):
        flac_name = f"{stem}.flac"
        target_path = dev_flac_dir / flac_name
        zinfo = reader.zf.getinfo(f"LA/ASVspoof2019_LA_dev/flac/{flac_name}")

        if target_path.is_file() and target_path.stat().st_size == zinfo.file_size:
            data = target_path.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
        else:
            data = reader.extract_member(zinfo)
            target_path.write_bytes(data)
            md5 = hashlib.md5(data).hexdigest()

        total_audio_bytes += len(data)
        audio_stats.append({
            "filename": flac_name,
            "size_bytes": len(data),
            "md5": md5,
        })

        if idx == 1 or idx % 25 == 0 or idx == len(target_stems):
            elapsed = time.perf_counter() - t0
            rate = idx / max(1e-6, elapsed)
            print(f"  Acquired [{idx}/{len(target_stems)}] ({idx/len(target_stems)*100:.1f}%) - {rate:.1f} files/sec")

    elapsed_total = time.perf_counter() - t0
    print(f"\nSUCCESS: Acquired {len(audio_stats)} audio files ({total_audio_bytes / (1024*1024):.2f} MB) in {elapsed_total:.2f}s.")

    return {
        "status": "SUCCESS",
        "protocols": protocol_stats,
        "audio_files_count": len(audio_stats),
        "total_audio_bytes": total_audio_bytes,
        "dev_flac_dir": str(dev_flac_dir),
        "proto_dir": str(proto_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire ASVspoof 2019 LA protocols and development audio.")
    parser.add_argument("--samples", type=int, default=100, help="Number of development samples to acquire (default: 100)")
    parser.add_argument("--balanced", action="store_true", help="Acquire a balanced distribution across all attack types")
    parser.add_argument("--per_attack", type=int, default=10, help="Samples per attack type when --balanced is set")
    args = parser.parse_args()
    acquire_dev_data(num_samples=args.samples, balanced=args.balanced, per_attack=args.per_attack)


if __name__ == "__main__":
    main()
