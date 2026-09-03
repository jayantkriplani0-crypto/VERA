import urllib.request

url = 'https://datashare.ed.ac.uk/bitstreams/765597d6-633f-418a-96b0-39274a8d56c9/download'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = resp.read()

with open('scratch/ASVspoof2019_README.txt', 'wb') as f:
    f.write(data)

with open('scratch/ASVspoof2019_README.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for line in lines[:45]:
    print(line.rstrip())
