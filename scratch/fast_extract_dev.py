import io
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib
import hashlib

url = 'https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download'
total_size = 7640952520
fetch_size = 8 * 1024 * 1024
start_byte = total_size - fetch_size

print(f"Fetching central directory (last {fetch_size // (1024*1024)} MB)...")
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

def extract_member_by_range(zinfo: zipfile.ZipInfo) -> bytes:
    """Fetch single zip member using 1 HTTP range request."""
    # Local header is at zinfo.header_offset: 30 bytes + name_len + extra_len
    # Max header size is ~300 bytes.
    read_len = 300 + zinfo.compress_size
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={zinfo.header_offset}-{zinfo.header_offset + read_len - 1}'}
    )
    with urllib.request.urlopen(req) as r:
        raw_chunk = r.read()

    # Parse local file header
    sig, ver, flags, comp_method, mtime, mdate, crc32, comp_size, uncomp_size, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", raw_chunk[:30]
    )
    assert sig == 0x04034b50, f"Invalid local header signature: {hex(sig)}"
    data_start = 30 + name_len + extra_len
    comp_bytes = raw_chunk[data_start:data_start + zinfo.compress_size]

    if comp_method == zipfile.ZIP_STORED:
        uncomp_bytes = comp_bytes
    elif comp_method == zipfile.ZIP_DEFLATED:
        uncomp_bytes = zlib.decompress(comp_bytes, -15)
    else:
        raise ValueError(f"Unsupported compression method: {comp_method}")

    assert zlib.crc32(uncomp_bytes) == zinfo.CRC, "CRC mismatch!"
    return uncomp_bytes

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

print("\n[1/3] Extracting official ASVspoof 2019 protocol files...")
for p in protocol_names:
    zinfo = zf.getinfo(p)
    target_name = Path(p).name
    target_path = proto_dir / target_name
    print(f"  Fetching {target_name} ({zinfo.file_size} bytes)...")
    data = extract_member_by_range(zinfo)
    target_path.write_bytes(data)
    md5 = hashlib.md5(data).hexdigest()
    print(f"  -> Successfully extracted {target_name} ({len(data)} bytes, MD5: {md5})")

# 2. Parse first 100 dev samples from dev protocol
dev_proto_path = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
with open(dev_proto_path, "r", encoding="utf-8") as f:
    dev_lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]

print(f"\n[2/3] Total trials in dev protocol: {len(dev_lines)}")
first_100_stems = [parts[1] for parts in dev_lines[:100]]
first_100_zinfos = [zf.getinfo(f"LA/ASVspoof2019_LA_dev/flac/{s}.flac") for s in first_100_stems]

# Determine byte range for these 100 files
min_offset = min(z.header_offset for z in first_100_zinfos)
max_offset = max(z.header_offset + 300 + z.compress_size for z in first_100_zinfos)
chunk_bytes = max_offset - min_offset
print(f"Byte range for first 100 dev samples: {min_offset} - {max_offset} ({chunk_bytes / (1024*1024):.2f} MB)")

# Fetch chunk in ONE range request!
print(f"Downloading contiguous {chunk_bytes / (1024*1024):.2f} MB chunk for all 100 samples in ONE request...")
req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={min_offset}-{max_offset - 1}'}
)
with urllib.request.urlopen(req) as resp:
    chunk_data = resp.read()
print(f"Received {len(chunk_data)} bytes ({len(chunk_data)/(1024*1024):.2f} MB)!")

# Extract files from the downloaded chunk
print("\n[3/3] Extracting and verifying 100 dev FLAC audio files...")
extracted_count = 0
for idx, zinfo in enumerate(first_100_zinfos, 1):
    flac_name = Path(zinfo.filename).name
    target_path = dev_flac_dir / flac_name
    rel_offset = zinfo.header_offset - min_offset

    # Parse local file header
    sig, ver, flags, comp_method, mtime, mdate, crc32, comp_size, uncomp_size, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", chunk_data[rel_offset:rel_offset+30]
    )
    assert sig == 0x04034b50, f"Bad header for {flac_name}"
    data_start = rel_offset + 30 + name_len + extra_len
    comp_bytes = chunk_data[data_start:data_start + zinfo.compress_size]

    if comp_method == zipfile.ZIP_STORED:
        uncomp_bytes = comp_bytes
    elif comp_method == zipfile.ZIP_DEFLATED:
        uncomp_bytes = zlib.decompress(comp_bytes, -15)
    else:
        raise ValueError(f"Unsupported method: {comp_method}")

    # Verify CRC32
    assert zlib.crc32(uncomp_bytes) == zinfo.CRC, f"CRC mismatch in {flac_name}!"
    target_path.write_bytes(uncomp_bytes)
    extracted_count += 1

print(f"\nSUCCESS: Extracted {extracted_count} verified FLAC audio files into '{dev_flac_dir.resolve()}'!")
