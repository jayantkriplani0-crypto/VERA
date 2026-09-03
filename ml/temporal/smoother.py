"""Temporal smoothing and aggregation algorithms for streaming voice authenticity inference.

Mitigates transient acoustic noise, clipping, and momentary channel dropouts
from triggering unwarranted state changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional
import numpy as np


class TemporalSmoother(ABC):
    """Abstract base class for streaming score smoothers."""

    @abstractmethod
    def update(self, score: float) -> float:
        """Ingest a new raw/calibrated window score and return the smoothed score."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal historical state."""
        pass

    @property
    @abstractmethod
    def current_value(self) -> Optional[float]:
        """Current smoothed score, or None if no samples ingested."""
        pass

    @property
    @abstractmethod
    def history(self) -> List[float]:
        """Chronological sequence of ingested raw scores."""
        pass


class MovingAverageSmoother(TemporalSmoother):
    """Simple Moving Average (SMA) over a sliding FIFO queue of historical scores."""

    def __init__(self, window_len: int = 5) -> None:
        if window_len <= 0:
            raise ValueError("window_len must be a positive integer.")
        self.window_len = window_len
        self._history: Deque[float] = deque(maxlen=window_len)

    def update(self, score: float) -> float:
        self._history.append(float(score))
        return float(np.mean(self._history))

    def reset(self) -> None:
        self._history.clear()

    @property
    def current_value(self) -> Optional[float]:
        if not self._history:
            return None
        return float(np.mean(self._history))

    @property
    def history(self) -> List[float]:
        return list(self._history)


class ExponentialMovingAverageSmoother(TemporalSmoother):
    """Exponential Moving Average (EMA) with configurable smoothing factor alpha.

    Formula:
        S_0 = x_0
        S_t = alpha * x_t + (1 - alpha) * S_{t-1}
    """

    def __init__(self, alpha: float = 0.35) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        self.alpha = float(alpha)
        self._current_ema: Optional[float] = None
        self._history: List[float] = []

    def update(self, score: float) -> float:
        s = float(score)
        self._history.append(s)

        if self._current_ema is None:
            # Unbiased initialization on first observation
            self._current_ema = s
        else:
            self._current_ema = (self.alpha * s) + ((1.0 - self.alpha) * self._current_ema)

        return float(self._current_ema)

    def reset(self) -> None:
        self._current_ema = None
        self._history.clear()

    @property
    def current_value(self) -> Optional[float]:
        return self._current_ema

    @property
    def history(self) -> List[float]:
        return list(self._history)


def create_smoother(method: str = "ema", **kwargs) -> TemporalSmoother:
    """Factory helper to instantiate a configured smoother."""
    m = method.lower().strip()
    if m in ("ema", "exponential", "exponential_moving_average"):
        alpha = kwargs.get("alpha", 0.35)
        return ExponentialMovingAverageSmoother(alpha=alpha)
    elif m in ("sma", "moving_average", "ma"):
        window_len = kwargs.get("window_len", 5)
        return MovingAverageSmoother(window_len=window_len)
    else:
        raise ValueError(f"Unsupported smoothing method: '{method}'. Choose 'ema' or 'sma'.")
