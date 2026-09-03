import io
from pathlib import Path
import urllib.request
import zipfile
import hashlib

url = 'https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download'
total_size = 7640952520
fetch_size = 8 * 1024 * 1024
start_byte = total_size - fetch_size

headers = {'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start_byte}-{total_size - 1}'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    tail_data = resp.read()

class RangeSeekableStream(io.RawIOBase):
    def __init__(self, url, size, tail_offset, tail_bytes):
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
            return b''
        if self.pos >= self.tail_offset:
            rel = self.pos - self.tail_offset
            chunk = self.tail_bytes[rel:rel+size]
            self.pos += len(chunk)
            return chunk
        end = self.pos + size - 1
        req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

stream = RangeSeekableStream(url, total_size, start_byte, tail_data)
zf = zipfile.ZipFile(stream)

# Target directory
base_dir = Path("data/asvspoof2019")
proto_dir = base_dir / "protocols"
dev_flac_dir = base_dir / "dev" / "flac"
proto_dir.mkdir(parents=True, exist_ok=True)
dev_flac_dir.mkdir(parents=True, exist_ok=True)

# 1. Extract protocol files
protocol_names = [
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
]

print("Extracting protocol files...")
for p in protocol_names:
    target_name = Path(p).name
    target_path = proto_dir / target_name
    print(f"  Extracting {target_name}...")
    data = zf.read(p)
    target_path.write_bytes(data)
    md5 = hashlib.md5(data).hexdigest()
    print(f"  -> Saved {target_name} ({len(data)} bytes, MD5: {md5})")

# 2. Extract first 100 dev FLAC files from the protocol list
dev_proto_path = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
with open(dev_proto_path, "r", encoding="utf-8") as f:
    lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]

print(f"\nTotal trials in dev protocol: {len(lines)}")
# Collect first 100 trial audio stems
sample_100_stems = [parts[1] for parts in lines[:100]]

print(f"Extracting first 100 development audio samples for pre-flight dry run...")
extracted_count = 0
for idx, stem in enumerate(sample_100_stems, 1):
    flac_name = f"{stem}.flac"
    zip_path = f"LA/ASVspoof2019_LA_dev/flac/{flac_name}"
    target_path = dev_flac_dir / flac_name
    if not target_path.exists():
        data = zf.read(zip_path)  # Note: zipfile automatically verifies CRC32
        target_path.write_bytes(data)
    extracted_count += 1
    if idx % 20 == 0 or idx == 100:
        print(f"  Extracted [{idx}/100] dev audio files...")

print(f"\nSuccessfully extracted {extracted_count} dev FLAC files into '{dev_flac_dir}'!")
