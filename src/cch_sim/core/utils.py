from __future__ import annotations

import math
import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, float):
        x = max(-50.0, min(50.0, x))
        return 1.0 / (1.0 + math.exp(-x))
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    if isinstance(p, float):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        return math.log(p / (1.0 - p))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def band_for_seconds(sec: float) -> str:
    if sec < 120.0:
        return "short"
    if sec < 600.0:
        return "med"
    return "long"

