"""Dataset loading and speaker-disjoint splitting module for Layer 1 Voice Authenticity.

Features:
  - Strongly typed AudioSample metadata: file_path, label (bonafide/spoof), speaker_id,
    attack_type, language, split.
  - Multi-format loader: CSV, JSON, ASVspoof protocol text files.
  - Strict speaker-disjoint train / validation / test partitioning (zero speaker leakage).
  - Validation assertions ensuring no speaker overlap across splits.
  - Synthetic benchmark dataset generator for offline reproducibility and verification.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf


class SpeakerLeakageError(Exception):
    """Raised when the same speaker ID is found across multiple dataset splits."""
    pass


@dataclass(frozen=True)
class AudioSample:
    """Represents a single labeled audio sample."""
    file_path: str
    label: str               # "bonafide" or "spoof"
    speaker_id: str
    attack_type: str         # "-" / "bonafide" for genuine; "A01".."A19", "TTS", "VC" etc. for spoof
    language: str = "en"
    split: str = "test"      # "train", "val", "test", "eval"
    codec: str = "-"         # ASVspoof 2021 codec (e.g. "alaw", "g722", "opus", "-")
    transmission: str = "-"  # ASVspoof 2021 channel (e.g. "pstn", "voip", "clean", "-")
    source: str = "-"        # System source or corpus identifier

    @property
    def is_bonafide(self) -> bool:
        """Returns True if the sample is bona fide (genuine human speech)."""
        return self.label.strip().lower() in ("bonafide", "bona_fide", "genuine", "real", "1")

    @property
    def ground_truth_binary(self) -> int:
        """Binary label: 1 = bona fide, 0 = spoof."""
        return 1 if self.is_bonafide else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ASVDataset:
    """Container for audio samples with multi-format parsing and speaker-disjoint split controls."""

    def __init__(self, samples: Optional[List[AudioSample]] = None) -> None:
        self.samples: List[AudioSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self.samples)

    @property
    def speakers(self) -> Set[str]:
        """Return the set of all unique speaker IDs in the dataset."""
        return {s.speaker_id for s in self.samples}

    @property
    def bonafide_count(self) -> int:
        """Count of bona fide samples."""
        return sum(1 for s in self.samples if s.is_bonafide)

    @property
    def spoof_count(self) -> int:
        """Count of spoof samples."""
        return sum(1 for s in self.samples if not s.is_bonafide)

    @property
    def attack_types(self) -> Set[str]:
        """Set of all attack types present."""
        return {s.attack_type for s in self.samples}

    # -----------------------------------------------------------------------
    # Multi-format Loaders
    # -----------------------------------------------------------------------
    @classmethod
    def from_csv(cls, csv_path: str | Path, base_audio_dir: Optional[str | Path] = None) -> ASVDataset:
        """Load dataset from a CSV manifest.

        Expected columns (flexible naming):
          - file_path / audio_path / filename / path
          - label / key / class (values: bonafide/spoof or 1/0)
          - speaker_id / speaker / spk_id
          - attack_type / attack / spoof_type (optional, defaults to '-' for bonafide)
          - language / lang (optional, defaults to 'en')
          - split (optional, defaults to 'test')
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {path.resolve()}")

        base_dir = Path(base_audio_dir) if base_audio_dir else path.parent
        samples = []

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize keys
                r = {k.strip().lower(): v.strip() for k, v in row.items() if k}

                # Resolve file path
                raw_file = r.get("file_path") or r.get("audio_path") or r.get("filename") or r.get("path")
                if not raw_file:
                    continue
                file_p = Path(raw_file)
                if not file_p.is_absolute():
                    file_p = (base_dir / file_p).resolve()

                label = r.get("label") or r.get("key") or r.get("class") or "bonafide"
                spk = r.get("speaker_id") or r.get("speaker") or r.get("spk_id") or "SPK_UNKNOWN"
                attack = r.get("attack_type") or r.get("attack") or r.get("spoof_type") or ("-" if label.lower() == "bonafide" else "unknown_spoof")
                lang = r.get("language") or r.get("lang") or "en"
                split = r.get("split") or "test"

                samples.append(AudioSample(
                    file_path=str(file_p),
                    label=label,
                    speaker_id=spk,
                    attack_type=attack,
                    language=lang,
                    split=split,
                ))

        return cls(samples)

    @classmethod
    def from_json(cls, json_path: str | Path, base_audio_dir: Optional[str | Path] = None) -> ASVDataset:
        """Load dataset from a JSON list or dictionary manifest."""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest JSON not found: {path.resolve()}")

        base_dir = Path(base_audio_dir) if base_audio_dir else path.parent
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_list = data if isinstance(data, list) else data.get("samples", [])
        samples = []

        for item in raw_list:
            raw_file = item.get("file_path") or item.get("audio_path") or item.get("filename")
            if not raw_file:
                continue
            file_p = Path(raw_file)
            if not file_p.is_absolute():
                file_p = (base_dir / file_p).resolve()

            samples.append(AudioSample(
                file_path=str(file_p),
                label=item.get("label", "bonafide"),
                speaker_id=item.get("speaker_id", "SPK_UNKNOWN"),
                attack_type=item.get("attack_type", "-" if item.get("label") == "bonafide" else "spoof"),
                language=item.get("language", "en"),
                split=item.get("split", "test"),
            ))

        return cls(samples)

    @classmethod
    def from_asvspoof_protocol(
        cls,
        protocol_file: str | Path,
        audio_dir: str | Path,
        file_ext: Optional[str] = None,
        split_name: str = "test"
    ) -> ASVDataset:
        """Load dataset from standard ASVspoof protocol TXT file (supports 2019 and 2021 formats).

        Formats Supported:
          1. ASVspoof 2019 (5 columns):
             `[SPEAKER_ID] [AUDIO_FILE_NAME] [SYSTEM_ID/-] [ATTACK_TYPE/-] [bonafide/spoof]`
             Example: `LA_0079 LA_E_2833633 - - bonafide`
                      `LA_0079 LA_E_8877456 - A07 spoof`

          2. ASVspoof 2021 trial metadata (7 or 8 columns):
             `[SPEAKER_ID] [AUDIO_FILE_NAME] [CODEC] [TRANSMISSION] [ATTACK_TYPE] [KEY] [TRIM] [SRC]`
             Example: `LA_0023 LA_E_0000001 alaw pstn - bonafide notrim asvspoof`
                      `LA_0023 LA_E_0000002 g722 ip A07 spoof notrim asvspoof`
        """
        proto_path = Path(protocol_file)
        if not proto_path.is_file():
            raise FileNotFoundError(f"Protocol file not found: {proto_path.resolve()}")

        audio_root = Path(audio_dir)
        samples = []

        with open(proto_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                parts = line_str.split()
                if len(parts) < 5:
                    continue

                spk_id = parts[0]
                audio_stem = parts[1]

                # Distinguish 2019 (5 columns) vs 2021 (7+ columns)
                if len(parts) >= 7 and parts[5].lower() in ("bonafide", "spoof"):
                    # ASVspoof 2021 format
                    codec = parts[2]
                    transmission = parts[3]
                    attack_type = parts[4]
                    key = parts[5].lower()
                    source = parts[7] if len(parts) > 7 else parts[6]
                else:
                    # ASVspoof 2019 standard format
                    codec = "-"
                    transmission = "-"
                    attack_type = parts[3]
                    key = parts[4].lower()
                    source = parts[2] if len(parts) > 2 else "-"

                # Extension resolution:
                # 1. If audio_stem already contains extension, use it
                # 2. If file_ext specified, use it
                # 3. Otherwise check if .flac or .wav exists in audio_root (default to .flac if none exists)
                if Path(audio_stem).suffix:
                    filename = audio_stem
                elif file_ext is not None:
                    ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
                    filename = f"{audio_stem}{ext}"
                else:
                    cand_flac = audio_root / f"{audio_stem}.flac"
                    cand_wav = audio_root / f"{audio_stem}.wav"
                    if cand_flac.is_file():
                        filename = f"{audio_stem}.flac"
                    elif cand_wav.is_file():
                        filename = f"{audio_stem}.wav"
                    else:
                        filename = f"{audio_stem}.flac"  # Default official format

                audio_path = (audio_root / filename).resolve()

                samples.append(AudioSample(
                    file_path=str(audio_path),
                    label="bonafide" if key == "bonafide" else "spoof",
                    speaker_id=spk_id,
                    attack_type=attack_type,
                    language="en",
                    split=split_name,
                    codec=codec,
                    transmission=transmission,
                    source=source,
                ))

        return cls(samples)

    @classmethod
    def auto_load(cls, file_or_dir: str | Path, audio_dir: Optional[str | Path] = None) -> ASVDataset:
        """Auto-detect format (CSV, JSON, protocol TXT) and load."""
        path = Path(file_or_dir)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path, base_audio_dir=audio_dir)
        elif path.suffix.lower() in (".json", ".jsonl"):
            return cls.from_json(path, base_audio_dir=audio_dir)
        elif path.suffix.lower() in (".txt", ".proto", ""):
            return cls.from_asvspoof_protocol(path, audio_dir=audio_dir or path.parent)
        raise ValueError(f"Unsupported dataset format for '{path}'")

    # -----------------------------------------------------------------------
    # Speaker-Disjoint Partitioning
    # -----------------------------------------------------------------------
    def split_by_speaker(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Dict[str, ASVDataset]:
        """Split the dataset into train, validation, and test subsets with GUARANTEED speaker isolation.

        Rules:
          - No speaker appearing in train will appear in val or test.
          - No speaker appearing in val will appear in test.
          - Verified by `verify_speaker_disjointness`.

        Returns:
            Dictionary with keys 'train', 'val', 'test' mapping to ASVDataset instances.
        """
        total_r = train_ratio + val_ratio + test_ratio
        train_r = train_ratio / total_r
        val_r = val_ratio / total_r

        all_speakers = sorted(list(self.speakers))
        if len(all_speakers) < 2:
            raise ValueError(
                f"Need at least 2 distinct speakers for disjoint splitting; found {len(all_speakers)}."
            )

        rng = random.Random(seed)
        shuffled_speakers = list(all_speakers)
        rng.shuffle(shuffled_speakers)

        n_spk = len(shuffled_speakers)
        if n_spk >= 3:
            # Guarantee at least 1 speaker in each split (train, val, test)
            n_train = max(1, int(round(n_spk * train_r)))
            n_val = max(1, int(round(n_spk * val_r)))
            if n_train + n_val >= n_spk:
                n_train = max(1, n_spk - 2)
                n_val = 1
        elif n_spk == 2:
            # With 2 speakers, assign 1 to train and 1 to test (0 to val)
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0

        train_spks = set(shuffled_speakers[:n_train])
        val_spks = set(shuffled_speakers[n_train:n_train + n_val])
        test_spks = set(shuffled_speakers[n_train + n_val:])

        splits_dict: Dict[str, List[AudioSample]] = {"train": [], "val": [], "test": []}

        for sample in self.samples:
            spk = sample.speaker_id
            if spk in train_spks:
                target_split = "train"
            elif spk in val_spks:
                target_split = "val"
            else:
                target_split = "test"

            # Create new sample with updated split tag
            splits_dict[target_split].append(AudioSample(
                file_path=sample.file_path,
                label=sample.label,
                speaker_id=sample.speaker_id,
                attack_type=sample.attack_type,
                language=sample.language,
                split=target_split,
            ))

        result = {k: ASVDataset(v) for k, v in splits_dict.items()}

        # Enforce speaker disjointness validation
        self.verify_speaker_disjointness(result)
        return result

    @staticmethod
    def verify_speaker_disjointness(splits: Dict[str, ASVDataset]) -> None:
        """Verify that no speaker is shared across any two splits.

        Raises:
            SpeakerLeakageError: If any speaker ID is present in more than one split.
        """
        seen: Dict[str, str] = {}
        for split_name, ds in splits.items():
            for spk in ds.speakers:
                if spk in seen:
                    prev_split = seen[spk]
                    raise SpeakerLeakageError(
                        f"CRITICAL: Speaker '{spk}' leaked between '{prev_split}' and '{split_name}'!"
                    )
                seen[spk] = split_name

    def filter_by_split(self, split_name: str) -> ASVDataset:
        """Return a new ASVDataset containing only samples from the specified split."""
        filtered = [s for s in self.samples if s.split.lower() == split_name.lower()]
        return ASVDataset(filtered)

    def save_manifest_csv(self, output_path: str | Path) -> None:
        """Save dataset samples to a standard CSV manifest."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["file_path", "label", "speaker_id", "attack_type", "language", "split", "codec", "transmission", "source"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore"
            )
            writer.writeheader()
            for s in self.samples:
                writer.writerow(s.to_dict())

    # -----------------------------------------------------------------------
    # Synthetic Benchmark Generator
    # -----------------------------------------------------------------------
    @classmethod
    def generate_synthetic_benchmark(
        cls,
        output_dir: str | Path,
        num_speakers: int = 10,
        samples_per_speaker: int = 4,
        sample_rate: int = 16_000,
        duration_sec: float = 3.0,
        seed: int = 42
    ) -> Tuple[ASVDataset, Path]:
        """Generate a synthetic multi-speaker dataset with bona fide and spoof audio for benchmark validation."""
        root = Path(output_dir)
        audio_dir = root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        attack_types = ["TTS-FastSpeech2", "TTS-VITS", "VC-StarGAN", "VC-DiffVC", "Neural-Vocoder"]
        samples: List[AudioSample] = []

        for i in range(num_speakers):
            spk_id = f"SPK_{i+1:03d}"
            base_freq = 120.0 + (i * 25.0)  # Distinct pitch per speaker

            for j in range(samples_per_speaker):
                is_bonafide = (j % 2 == 0)
                label = "bonafide" if is_bonafide else "spoof"
                attack = "-" if is_bonafide else attack_types[j % len(attack_types)]

                filename = f"{spk_id}_{label}_{j+1}.wav"
                file_path = audio_dir / filename

                # Synthesize acoustic wave
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
                if is_bonafide:
                    # Clean harmonic acoustic signal simulating natural vocal formants
                    harmonics = (
                        0.6 * np.sin(2 * np.pi * base_freq * t) +
                        0.3 * np.sin(2 * np.pi * (base_freq * 2) * t) +
                        0.15 * np.sin(2 * np.pi * (base_freq * 3) * t)
                    )
                    # Natural amplitude modulation
                    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
                    wave = (harmonics * envelope * 0.8).astype(np.float32)
                else:
                    # Synthetic / vocoded distortion simulation
                    carrier = np.sin(2 * np.pi * (base_freq * 1.1) * t)
                    modulator = np.sin(2 * np.pi * 50.0 * t)
                    noise = 0.05 * np_rng.randn(len(t))
                    wave = ((carrier * modulator + noise) * 0.7).astype(np.float32)

                sf.write(str(file_path), wave, sample_rate)

                samples.append(AudioSample(
                    file_path=str(file_path.resolve()),
                    label=label,
                    speaker_id=spk_id,
                    attack_type=attack,
                    language="en",
                    split="test",
                ))

        ds = cls(samples)
        manifest_path = root / "benchmark_manifest.csv"
        ds.save_manifest_csv(manifest_path)
        return ds, manifest_path
