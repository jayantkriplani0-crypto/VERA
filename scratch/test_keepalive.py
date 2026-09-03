import http.client
import struct
import zlib
import time
import zipfile
from pathlib import Path

# Open persistent HTTPS connection
conn = http.client.HTTPSConnection("datashare.ed.ac.uk", timeout=30)
headers = {'User-Agent': 'Mozilla/5.0'}

# Test fetching a small range
conn.request("GET", "/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download", headers={**headers, 'Range': 'bytes=0-10'})
resp = conn.getresponse()
print("Keep-Alive test status:", resp.status, "Length:", len(resp.read()))
conn.close()
