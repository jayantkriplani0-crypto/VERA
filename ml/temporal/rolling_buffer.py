"""Rolling audio buffer for near-real-time streaming inference.

Optimized with a pre-allocated contiguous storage buffer to eliminate
continuous memory allocations and garbage collection overhead.
Generates model-compatible windows of exactly 64,600 samples (~4.0375 seconds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Constant defined by frozen Spectra-AASIST3 preprocessor requirements
WINDOW_SIZE_SAMPLES = 64_600  # 4.0375 seconds at 16,000 Hz
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_HOP_SIZE_SAMPLES = 16_000  # 1.0000 second hop
DEFAULT_INITIAL_CAPACITY = 262_144  # ~16.4 seconds of 16kHz audio in preallocated buffer


@dataclass(frozen=True)
class AudioWindow:
    """A model-compatible audio window extracted from the rolling buffer."""
    window_index: int
    samples: np.ndarray
    sample_rate: int
    start_sample_idx: int
    end_sample_idx: int
    start_sec: float
    end_sec: float
    is_flushed: bool = False

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / float(self.sample_rate)


class RollingAudioBuffer:
    """Manages an incoming stream of audio samples using pre-allocated contiguous storage."""

    def __init__(
        self,
        window_size_samples: int = WINDOW_SIZE_SAMPLES,
        hop_size_samples: int = DEFAULT_HOP_SIZE_SAMPLES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        initial_capacity: int = DEFAULT_INITIAL_CAPACITY,
    ) -> None:
        if window_size_samples <= 0:
            raise ValueError("window_size_samples must be positive.")
        if hop_size_samples <= 0:
            raise ValueError("hop_size_samples must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self.window_size_samples = window_size_samples
        self.hop_size_samples = hop_size_samples
        self.sample_rate = sample_rate

        # Pre-allocated storage buffer (reused across all chunk arrivals)
        self._capacity = max(initial_capacity, window_size_samples * 4)
        self._storage: np.ndarray = np.zeros(self._capacity, dtype=np.float32)
        self._head: int = 0  # Read cursor
        self._tail: int = 0  # Write cursor

        self._total_samples_received: int = 0
        self._next_window_start_idx: int = 0
        self._window_counter: int = 0

    @property
    def buffered_samples_count(self) -> int:
        """Current number of unconsumed/retained samples in buffer."""
        return self._tail - self._head

    @property
    def total_samples_received(self) -> int:
        """Total cumulative samples ingested into the buffer since start/reset."""
        return self._total_samples_received

    @property
    def total_duration_received_sec(self) -> float:
        """Total cumulative seconds of audio received."""
        return self._total_samples_received / float(self.sample_rate)

    @property
    def next_window_start_sec(self) -> float:
        """Timestamp in seconds of the start of the next window."""
        return self._next_window_start_idx / float(self.sample_rate)

    def append(self, chunk: np.ndarray | bytes | list[float]) -> List[AudioWindow]:
        """Append incoming audio samples and return any newly ready overlapping windows."""
        if isinstance(chunk, bytes):
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(chunk, list):
            audio = np.asarray(chunk, dtype=np.float32)
        elif isinstance(chunk, np.ndarray):
            if chunk.ndim > 1:
                # Average multi-channel to mono
                audio = np.mean(chunk, axis=1, dtype=np.float32)
            else:
                audio = chunk.astype(np.float32)
        else:
            raise TypeError(f"Unsupported chunk type: {type(chunk)}")

        chunk_len = len(audio)
        if chunk_len == 0:
            return []

        # Scale audio safely: if unscaled int16 data (>256.0), divide by 32768; else clip to [-1.0, 1.0]
        max_val = float(np.max(np.abs(audio)))
        if max_val > 256.0:
            audio = np.clip(audio / 32768.0, -1.0, 1.0)
        elif max_val > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

        unconsumed = self._tail - self._head

        # Ensure storage has enough capacity for unconsumed + incoming chunk
        if self._tail + chunk_len > self._capacity:
            if unconsumed + chunk_len <= self._capacity:
                # Compact unconsumed samples to the beginning of storage
                if unconsumed > 0:
                    self._storage[:unconsumed] = self._storage[self._head : self._tail]
                self._head = 0
                self._tail = unconsumed
            else:
                # Expand capacity
                new_cap = max(self._capacity * 2, unconsumed + chunk_len + self.window_size_samples * 2)
                new_storage = np.zeros(new_cap, dtype=np.float32)
                if unconsumed > 0:
                    new_storage[:unconsumed] = self._storage[self._head : self._tail]
                self._storage = new_storage
                self._capacity = new_cap
                self._head = 0
                self._tail = unconsumed

        # Copy incoming samples into preallocated storage
        self._storage[self._tail : self._tail + chunk_len] = audio
        self._tail += chunk_len
        self._total_samples_received += chunk_len

        ready_windows: List[AudioWindow] = []

        # Extract windows while sufficient samples exist
        while (self._tail - self._head) >= self.window_size_samples:
            window_samples = self._storage[self._head : self._head + self.window_size_samples].copy()
            start_idx = self._next_window_start_idx
            end_idx = start_idx + self.window_size_samples

            window = AudioWindow(
                window_index=self._window_counter,
                samples=window_samples,
                sample_rate=self.sample_rate,
                start_sample_idx=start_idx,
                end_sample_idx=end_idx,
                start_sec=round(start_idx / float(self.sample_rate), 4),
                end_sec=round(end_idx / float(self.sample_rate), 4),
                is_flushed=False,
            )
            ready_windows.append(window)
            self._window_counter += 1
            self._next_window_start_idx += self.hop_size_samples

            # Advance read head by hop_size_samples
            self._head += self.hop_size_samples

        # If read head caught up with write head, reset to start of buffer
        if self._head == self._tail:
            self._head = 0
            self._tail = 0

        return ready_windows

    def flush(self, pad_mode: str = "tile") -> Optional[AudioWindow]:
        """Produce a final trailing window from any remaining buffered audio.

        If buffered audio is less than window_size_samples, it will be tiled or zero-padded
        to guarantee exactly window_size_samples (64,600).

        Args:
            pad_mode: 'tile' (repeat audio to fill window, standard for ASVspoof) or 'zero' (zero-pad).

        Returns:
            AudioWindow if at least some unconsumed audio remains, else None.
        """
        unconsumed = self._tail - self._head
        if unconsumed <= 0:
            return None

        buffered = self._storage[self._head : self._tail].copy()
        raw_len = len(buffered)

        if raw_len >= self.window_size_samples:
            window_samples = buffered[: self.window_size_samples]
        elif pad_mode == "tile":
            repeats = int(np.ceil(self.window_size_samples / float(raw_len)))
            window_samples = np.tile(buffered, repeats)[: self.window_size_samples]
        elif pad_mode == "zero":
            window_samples = np.pad(buffered, (0, self.window_size_samples - raw_len), mode="constant")
        else:
            raise ValueError(f"Unknown pad_mode: '{pad_mode}'. Must be 'tile' or 'zero'.")

        start_idx = self._next_window_start_idx
        end_idx = start_idx + raw_len

        window = AudioWindow(
            window_index=self._window_counter,
            samples=window_samples,
            sample_rate=self.sample_rate,
            start_sample_idx=start_idx,
            end_sample_idx=end_idx,
            start_sec=round(start_idx / float(self.sample_rate), 4),
            end_sec=round(end_idx / float(self.sample_rate), 4),
            is_flushed=True,
        )

        self._window_counter += 1
        self._head = 0
        self._tail = 0
        return window

    def reset(self) -> None:
        """Clear all buffer contents and reset counters."""
        self._head = 0
        self._tail = 0
        self._total_samples_received = 0
        self._next_window_start_idx = 0
        self._window_counter = 0
