"""用于评估指标的固定种子分层 bootstrap。"""
from __future__ import annotations

import random
from statistics import mean
from typing import Callable


def bootstrap_metric(
    predictions: list[dict],
    metric_fn: Callable[[list[dict]], float],
    *,
    iterations: int = 1000,
    seed: int = 20260817,
    ci: float = 0.95,
) -> dict[str, float]:
    """返回某个指标的 ``{"lower": ..., "upper": ..., "mean": ...}``。"""
    rng = random.Random(seed)
    n = len(predictions)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
    samples = []
    for _ in range(iterations):
        resampled = [predictions[rng.randrange(n)] for _ in range(n)]
        samples.append(metric_fn(resampled))
    samples.sort()
    alpha = (1.0 - ci) / 2.0
    lower_idx = max(0, int(alpha * iterations))
    upper_idx = min(iterations - 1, int((1.0 - alpha) * iterations))
    return {
        "lower": samples[lower_idx],
        "upper": samples[upper_idx],
        "mean": mean(samples),
    }
