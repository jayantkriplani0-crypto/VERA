import requests
import json

r = requests.get("https://datashare.ed.ac.uk/server/api/core/items/handle/10283/3336")
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    bundles_url = data["_links"]["bundles"]["href"]
    r_b = requests.get(bundles_url)
    bundles = r_b.json()["_embedded"]["bundles"]
    for b in bundles:
        if b["name"] == "ORIGINAL":
            b_bitstreams_url = b["_links"]["bitstreams"]["href"]
            r_bs = requests.get(b_bitstreams_url)
            for bs in r_bs.json()["_embedded"]["bitstreams"]:
                print(f"Bitstream: {bs['name']}, Size: {bs['sizeBytes']:,} bytes, ID: {bs['id']}")
