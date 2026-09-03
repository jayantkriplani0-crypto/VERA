import io
import urllib.request
import zipfile

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

dev_infos = [z for z in zf.infolist() if 'dev/flac' in z.filename and z.filename.endswith('.flac')]
print(f"Dev FLAC count: {len(dev_infos)}")
if dev_infos:
    first = dev_infos[0]
    last = dev_infos[-1]
    print(f"First dev file: {first.filename}, offset: {first.header_offset}, size: {first.file_size}")
    print(f"Last dev file: {last.filename}, offset: {last.header_offset}, size: {last.file_size}")
    # Check if contiguous
    sorted_by_offset = sorted(dev_infos, key=lambda x: x.header_offset)
    min_offset = sorted_by_offset[0].header_offset
    max_offset = sorted_by_offset[-1].header_offset + sorted_by_offset[-1].compress_size
    span_bytes = max_offset - min_offset
    print(f"Byte range for dev flac files: {min_offset} to {max_offset} ({span_bytes / (1024*1024):.2f} MB)")
