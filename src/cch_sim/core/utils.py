"""Small numerical utilities used across the codebase."""

from __future__ import annotations

import math

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic/sigmoid function.

    Clips inputs to avoid overflow/underflow in exponentials.
    """
    if isinstance(x, float):
        x = max(-50.0, min(50.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Stable log‑odds transform with clipping away from {0,1}."""
    if isinstance(p, float):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        return math.log(p / (1.0 - p))

    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def band_for_seconds(sec: float) -> str:
    """Map runtime seconds to a coarse time band: short/med/long.

    - short: ``sec < 120``
    - med: ``120 ≤ sec < 600``
    - long: ``sec ≥ 600``
    """
    if sec < 120.0:
        return "short"
    if sec < 600.0:
        return "med"
    return "long"
