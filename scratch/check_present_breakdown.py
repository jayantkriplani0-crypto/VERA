from collections import defaultdict
from pathlib import Path

audio_dir = Path("data/asvspoof2019/dev/flac")
with open("data/asvspoof2019/protocols/ASVspoof2019.LA.cm.dev.trl.txt") as f:
    lines = [l.strip().split() for l in f if l.strip() and not l.startswith('#')]

present = [l for l in lines if (audio_dir / f"{l[1]}.flac").is_file()]
spk_stats = defaultdict(lambda: defaultdict(int))
for spk, stem, sys_id, att, key in present:
    spk_stats[spk][key] += 1
    spk_stats[spk][att] += 1
    spk_stats[spk]['total'] += 1

print(f"Total currently present on disk: {len(present)}")
print(f"{'Speaker':<10} {'Total':<7} {'BonaFide':<10} {'Spoof':<8} {'A01':<5} {'A02':<5} {'A03':<5} {'A04':<5} {'A05':<5} {'A06':<5}")
print("-" * 75)
for spk in sorted(spk_stats.keys()):
    d = spk_stats[spk]
    print(f"{spk:<10} {d['total']:<7} {d['bonafide']:<10} {d['spoof']:<8} {d['A01']:<5} {d['A02']:<5} {d['A03']:<5} {d['A04']:<5} {d['A05']:<5} {d['A06']:<5}")
