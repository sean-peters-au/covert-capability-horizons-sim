from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def ci_from_draws(draws: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    v = np.asarray(draws, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return (float("nan"), float("nan"), float("nan"))
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    lo_v, med_v, hi_v = np.percentile(v, [lo, 50.0, hi])
    return float(lo_v), float(med_v), float(hi_v)


def delta50_precision_gate(delta50_draws_seconds: np.ndarray, max_width_seconds: float, alpha: float = 0.05) -> Dict[str, float | bool]:
    """Gate on Δ50 precision: 95% CI width must be ≤ max_width_seconds.

    Returns dict with pass flag and measured CI stats.
    """
    lo, med, hi = ci_from_draws(delta50_draws_seconds, alpha=alpha)
    width = hi - lo if np.isfinite(hi) and np.isfinite(lo) else float("nan")
    passed = bool(np.isfinite(width) and width <= float(max_width_seconds))
    return dict(pass_=passed, lo=lo, med=med, hi=hi, width=width)


def delta50_in_range_gate(
    d50_draws_seconds: np.ndarray,
    min_delta_seconds: float,
    max_delta_seconds: float,
    min_fraction_in_range: float = 0.8,
) -> Dict[str, float | bool]:
    """Guardrail for identifiability: Δ50 draws should mostly lie within the observed Δ range.

    - Pass if at least `min_fraction_in_range` of draws lie in [min_delta_seconds, max_delta_seconds]
      and the median lies within the same interval.
    Returns dict with pass flag, fraction, and median.
    """
    v = np.asarray(d50_draws_seconds, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return dict(pass_=False, frac_in_range=float("nan"), median=float("nan"))
    lo = float(min(min_delta_seconds, max_delta_seconds))
    hi = float(max(min_delta_seconds, max_delta_seconds))
    in_range = (v >= lo) & (v <= hi)
    frac = float(np.mean(in_range))
    med = float(np.median(v))
    passed = bool((frac >= float(min_fraction_in_range)) and (lo <= med <= hi))
    return dict(pass_=passed, frac_in_range=frac, median=med, lo=lo, hi=hi)



def trend_recovery_rope_gate(
    trend_info: Dict[str, object],
    true_dm_months: float,
    rel_factor: float = 1.33,
    min_prob_in_window: float = 0.6,
    rel_width_max: float | None = None,
) -> Dict[str, float | bool]:
    """Bayesian ROPE-style gate on Δ50 doubling-months.

    - Let window be [dm_true/γ, dm_true*γ] with γ=rel_factor.
    - Compute p_in = P(dm in window) from 'doubling_months_draws'.
    - Gate passes if p_in ≥ min_prob_in_window and (optional) relative width (dm_hi - dm_lo)/dm_med ≤ rel_width_max.
    Returns dict with pass flag and diagnostics.
    """
    if not isinstance(trend_info, dict):
        return dict(pass_=False, p_in=float('nan'), rel_width=float('nan'))
    draws = trend_info.get("doubling_months_draws")
    dm_ci = trend_info.get("dm_ci")
    dm_med = trend_info.get("dm_median")
    v = np.asarray(draws, dtype=float) if draws is not None else np.array([])
    if v.size == 0 or not np.isfinite(v).any() or not np.isfinite(true_dm_months):
        return dict(pass_=False, p_in=float('nan'), rel_width=float('nan'))
    # Window on linear scale
    gamma = float(rel_factor)
    lo = float(true_dm_months) / gamma
    hi = float(true_dm_months) * gamma
    p_in = float(np.mean((v >= lo) & (v <= hi)))
    # Relative width check from CI if available
    rel_width_ok = True
    rel_width = float('nan')
    if isinstance(dm_ci, (list, tuple)) and len(dm_ci) == 2 and np.isfinite(dm_ci[0]) and np.isfinite(dm_ci[1]) and dm_med is not None and np.isfinite(dm_med):
        width = float(dm_ci[1]) - float(dm_ci[0])
        rel_width = width / float(dm_med) if float(dm_med) > 0 else float('inf')
        if rel_width_max is not None and np.isfinite(rel_width_max):
            rel_width_ok = bool(np.isfinite(rel_width) and (rel_width <= float(rel_width_max)))
    passed = bool((p_in >= float(min_prob_in_window)) and rel_width_ok)
    return dict(pass_=passed, p_in=p_in, lo=lo, hi=hi, rel_factor=gamma, rel_width=rel_width, true_dm=true_dm_months)
