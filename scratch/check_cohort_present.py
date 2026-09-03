from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_calibration_cohort import select_cohort_trials

base_dir = PROJECT_ROOT / "data" / "asvspoof2019"
trials = select_cohort_trials(base_dir / "protocols" / "ASVspoof2019.LA.cm.dev.trl.txt")
audio_dir = base_dir / "dev" / "flac"
present = [t for t in trials if (audio_dir / f"{t['audio_stem']}.flac").is_file()]
print(f"Cohort trials present on disk: {len(present)} / {len(trials)}")
