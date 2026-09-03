import io
import urllib.request
import zipfile

url = 'https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download'
total_size = 7640952520

# Central directory is at the end of the file. Let's read the last 8 MB.
fetch_size = 8 * 1024 * 1024
start_byte = total_size - fetch_size
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Range': f'bytes={start_byte}-{total_size - 1}'
}

print(f"Fetching last {fetch_size // (1024*1024)} MB of LA.zip to parse central directory...")
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    tail_data = resp.read()

print(f"Received {len(tail_data)} bytes.")

# Custom RemoteZip or find EOCD
# End of Central Directory signature: b'PK\x05\x06'
eocd_pos = tail_data.rfind(b'PK\x05\x06')
if eocd_pos == -1:
    print("EOCD not found in last 8 MB, might need larger fetch.")
else:
    print(f"Found EOCD at offset {eocd_pos} in tail data.")
    # Zip64 check
    eocd64_pos = tail_data.rfind(b'PK\x06\x06')
    print("Zip64 EOCD:", eocd64_pos != -1)

    # Let's see if zipfile can read with a seeking wrapper
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

            # Check if read fits inside cached tail
            if self.pos >= self.tail_offset:
                rel = self.pos - self.tail_offset
                chunk = self.tail_bytes[rel:rel+size]
                self.pos += len(chunk)
                return chunk

            # Range request
            end = self.pos + size - 1
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={self.pos}-{end}'})
            with urllib.request.urlopen(req) as r:
                data = r.read()
            self.pos += len(data)
            return data

    stream = RangeSeekableStream(url, total_size, start_byte, tail_data)
    zf = zipfile.ZipFile(stream)
    namelist = zf.namelist()
    print(f"Total entries in LA.zip: {len(namelist)}")
    print("First 20 entries:")
    for name in namelist[:20]:
        print(" ", name)

    # Search for protocol files
    protocols = [n for n in namelist if 'protocol' in n.lower() or n.endswith('.txt')]
    print("\nProtocol / Text files:")
    for p in protocols:
        print(" ", p)

    # Count dev and train files
    dev_files = [n for n in namelist if 'dev' in n.lower() and n.endswith('.flac')]
    train_files = [n for n in namelist if 'train' in n.lower() and n.endswith('.flac')]
    print(f"\nDev FLAC files count: {len(dev_files)}")
    print(f"Train FLAC files count: {len(train_files)}")
