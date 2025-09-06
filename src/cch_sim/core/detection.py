from __future__ import annotations

from typing import Dict
import math
import numpy as np
import pandas as pd

from .utils import logit, sigmoid, band_for_seconds


def _b0_targets_by_band(monitor: Dict) -> Dict[str, float]:
    base = float(monitor.get("fpr_target", 1e-4))
    short = monitor.get("fpr_short", None)
    med = monitor.get("fpr_med", None)
    long = monitor.get("fpr_long", None)
    return {
        "short": logit(float(short) if short is not None else base),
        "med": logit(float(med) if med is not None else base),
        "long": logit(float(long) if long is not None else base),
    }


def apply_detection_models(
    models_df: pd.DataFrame,
    monitor: Dict,
) -> pd.DataFrame:
    df = models_df.copy()
    if df.empty:
        df["success"] = []
        df["monitor_id"] = monitor.get("id", "M0")
        return df
    secs = df["runtime_s"].astype(float).to_numpy()
    bands = np.array([band_for_seconds(s) for s in secs])
    b0_map = _b0_targets_by_band(monitor)
    b0 = np.array([b0_map[b] for b in bands], dtype=float)
    # effective ECL ratio: covert seconds relative to baseline T-time; here T-time ~ runtime / exp(delta)
    # but we only stored runtime and delta_true, so approximate baseline as runtime / exp(delta)
    delta = df["delta_true"].astype(float).to_numpy()
    t_T = np.exp(np.log(np.clip(secs, 1e-9, None)) - delta)
    ecl_ratio = np.clip(secs / np.clip(t_T, 1e-9, None), 0.0, 1e9)
    # time-dilation factor
    tref_map = {
        "short": math.sqrt(30.0 * 120.0),
        "med": math.sqrt(120.0 * 600.0),
        "long": math.sqrt(600.0 * 3600.0),
    }
    trefs = np.array([tref_map[b] for b in bands], dtype=float)
    eta = float(monitor.get("eta", 0.3))
    det_b1 = float(monitor.get("det_b1", 2.0))
    ecl_eff = ecl_ratio * np.power(np.clip(trefs / np.clip(secs, 1e-9, None), 1e-9, 1e9), eta)
    p_det = sigmoid(b0 + det_b1 * ecl_eff)

    u_det = df["u_det"].astype(float).to_numpy()
    u_cov = df["u_cov"].astype(float).to_numpy()
    p_cov = df["p_cov"].astype(float).to_numpy()
    covert_ok = (u_cov < p_cov)
    detected = (u_det < p_det)
    success = (covert_ok & (~detected)).astype(int)
    out = df.copy()
    out["success"] = success
    out["monitor_id"] = monitor.get("id", "M0")
    return out

