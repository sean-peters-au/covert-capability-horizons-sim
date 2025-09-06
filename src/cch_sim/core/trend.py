"""Trend utilities: simple OLS trend on log(H50 seconds)."""
from __future__ import annotations

from typing import Dict
import math
import numpy as np


def fit_trend_ols(release_month: np.ndarray, h50_seconds: np.ndarray) -> Dict[str, float]:
    if release_month.size == 0 or h50_seconds.size == 0:
        return dict(slope=float("nan"), r2=float("nan"), doubling_months=float("nan"))

    def _frac_year(yyyymm):
        y = int(yyyymm) // 100
        m = int(yyyymm) % 100
        return y + (m - 1) / 12.0

    xyear = np.array([_frac_year(v) for v in release_month], float)
    ylog = np.log(np.clip(h50_seconds, 1e-12, None))
    X = np.column_stack([np.ones_like(xyear), xyear])
    try:
        beta = np.linalg.lstsq(X, ylog, rcond=None)[0]
    except np.linalg.LinAlgError:
        return dict(slope=float("nan"), r2=float("nan"), doubling_months=float("nan"))
    yhat = X @ beta
    resid = ylog - yhat
    r2 = 1.0 - (resid @ resid) / max(1e-12, np.sum((ylog - ylog.mean()) ** 2))
    slope = float(beta[1])
    # Interpretation: capability increases as H50 increases (log-seconds).
    # Doubling months = time to double H50 ⇒ finite when slope > 0.
    doubling_months = (12.0 * math.log(2) / slope) if slope > 0 else float("inf")
    return dict(slope=slope, r2=float(r2), doubling_months=float(doubling_months))


# Legacy interval-censored frequentist fit removed in favor of Bayesian pipeline.
