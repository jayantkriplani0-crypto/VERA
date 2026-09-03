from collections import defaultdict
from pathlib import Path

eval_proto = Path("data/asvspoof2019/protocols/ASVspoof2019.LA.cm.eval.trl.txt")
with open(eval_proto, "r", encoding="utf-8") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith('#')]

print(f"Total evaluation protocol trials: {len(lines)}")
spks = set(l[0] for l in lines)
print(f"Unique speakers: {len(spks)}")
attacks = defaultdict(int)
labels = defaultdict(int)
for l in lines:
    attacks[l[3]] += 1
    labels[l[4]] += 1

print(f"Labels: {dict(labels)}")
print("Attacks:")
for att in sorted(attacks.keys()):
    print(f"  {att}: {attacks[att]}")
