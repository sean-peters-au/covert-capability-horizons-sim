"""Apply detection models to simulated attempts (simplified, single‑threshold).

- Uses a single ultra‑low FPR target (``fpr_target``) to set a logit intercept.
- Detection probability increases monotonically with effective covert load (ECL).
- No exposure‑band calibration and no time‑dilation adjustments.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .utils import logit, sigmoid


def _b0_from_fpr(monitor: Dict) -> float:
    """Return the logit intercept from a single FPR target.

    Args:
        monitor: Dict with ``fpr_target``.

    Returns:
        ``logit(fpr_target)`` as float.
    """
    base = float(monitor.get("fpr_target", 1e-4))
    return float(logit(base))


def apply_detection_models(models_df: pd.DataFrame, monitor: Dict) -> pd.DataFrame:
    """Compute success flags for attempts under a monitor (single threshold).

    Args:
        models_df: Attempts from ``simulate_model_attempts``.
        monitor: Monitor configuration dict (expects ``fpr_target`` and optional
            ``det_b1`` sensitivity).

    Returns:
        Copy of input with success column and monitor_id attached.
    """
    df = models_df.copy()
    if df.empty:
        df["success"] = []
        df["monitor_id"] = monitor.get("id", "M0")
        return df
    # Intercept from a single FPR target
    b0 = _b0_from_fpr(monitor)

    # Effective covert load ratio (ECL); with runtime_s and delta_true, baseline T≈runtime/exp(delta)
    # ECL = (T+C)/T ≈ exp(delta_true). We compute directly for stability.
    delta = df["delta_true"].astype(float).to_numpy()
    ecl_ratio = np.exp(delta)

    # Sensitivity to covert load (det_b1); default modest slope
    det_b1 = float(monitor.get("det_b1", 2.0))
    p_det = sigmoid(b0 + det_b1 * ecl_ratio)

    u_det = df["u_det"].astype(float).to_numpy()
    u_cov = df["u_cov"].astype(float).to_numpy()
    p_cov = df["p_cov"].astype(float).to_numpy()
    covert_ok = u_cov < p_cov
    detected = u_det < p_det
    success = (covert_ok & (~detected)).astype(int)
    out = df.copy()
    out["success"] = success
    out["monitor_id"] = monitor.get("id", "M0")
    return out
