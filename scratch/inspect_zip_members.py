import sys
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.acquire_asvspoof_dev import RangeZipReader, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERA/1.0"})
reader = RangeZipReader(session, DIRECT_CONTENT_URL, LA_ZIP_TOTAL_SIZE)
names = reader.zf.namelist()
print(f"Total members in LA.zip: {len(names)}")

second_dirs = set('/'.join(n.split('/')[:2]) for n in names if len(n.split('/')) > 1)
print(f"Second-level directories: {sorted(list(second_dirs))}")

eval_files = [n for n in names if "eval" in n.lower()]
print(f"Eval-related members: {len(eval_files)}")
if eval_files:
    print(f"Sample eval members: {eval_files[:5]}")
