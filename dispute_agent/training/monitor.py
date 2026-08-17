"""GRPO collapse monitoring.

The monitor pauses training when the ratio of zero-variance groups in a recent
window exceeds a configured threshold.
"""
from __future__ import annotations

from collections import deque
from statistics import pstdev


class CollapseMonitor:
    def __init__(self, window: int = 50, max_zero_variance_ratio: float = 0.30) -> None:
        self.window = window
        self.max_zero_variance_ratio = max_zero_variance_ratio
        self._recent: deque[bool] = deque(maxlen=window)
        self.should_pause = False
        self.reason: str | None = None

    def observe(self, group_rewards: list[float], valid_rollouts: int) -> None:
        if valid_rollouts <= 0:
            zero_variance = True
        else:
            zero_variance = pstdev(group_rewards) < 1e-9
        self._recent.append(zero_variance)
        if len(self._recent) >= self.window:
            ratio = sum(self._recent) / len(self._recent)
            if ratio > self.max_zero_variance_ratio:
                self.should_pause = True
                self.reason = "zero_variance_group_ratio"
