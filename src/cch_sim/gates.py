"""Decision gates and simple interval utilities.

Implements gates used to declare designs “good enough” under different criteria:
precision of Δ50, identifiability (Δ50 within observed Δ range), and trend recovery
via a ROPE‑style window on doubling months. Also provides a helper to compute CI
triples from posterior draws.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def ci_from_draws(draws: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute a central credible interval triple (lo, median, hi).

    Args:
        draws: Posterior draws for a scalar quantity.
        alpha: Tail probability (default 0.05 → 95% interval).

    Returns:
        Tuple of floats ``(lo, median, hi)``. Returns ``(nan, nan, nan)`` when draws
        are empty or entirely non‑finite.
    """
    v = np.asarray(draws, dtype=float)

    # Short‑circuit on empty or non‑finite input.
    if v.size == 0 or not np.isfinite(v).any():
        return (float("nan"), float("nan"), float("nan"))

    # Convert alpha to percentile endpoints and compute the summary.
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    lo_v, med_v, hi_v = np.percentile(v, [lo, 50.0, hi])
    return float(lo_v), float(med_v), float(hi_v)


def delta50_precision_gate(
    delta50_draws_seconds: np.ndarray, max_width_seconds: float, alpha: float = 0.05
) -> Dict[str, float | bool]:
    """Precision gate on Δ50 draws.

    Passes when the central credible interval (``1 - alpha``) has width less than or
    equal to ``max_width_seconds``.

    Args:
        delta50_draws_seconds: Draws for Δ50 (seconds) for one model/monitor.
        max_width_seconds: Maximum allowed CI width in seconds.
        alpha: Tail probability for the interval (default 0.05 → 95% CI).

    Returns:
        Dict with ``{"pass_", "lo", "med", "hi", "width"}``.
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
    """Identifiability gate: Δ50 draws should lie mostly within observed Δ range.

    Passes when at least ``min_fraction_in_range`` of draws lie within the interval
    ``[min_delta_seconds, max_delta_seconds]`` and the median is inside the same interval.

    Args:
        d50_draws_seconds: Draws for Δ50 (seconds) for one model/monitor.
        min_delta_seconds: Minimum observed Δ seconds over attempts.
        max_delta_seconds: Maximum observed Δ seconds over attempts.
        min_fraction_in_range: Minimum fraction of draws that must lie in range.

    Returns:
        Dict with ``{"pass_", "frac_in_range", "median", "lo", "hi"}``.
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
    trend_info: dict[str, object],
    true_dm_months: float,
    rel_factor: float = 1.33,
    min_prob_in_window: float = 0.6,
    rel_width_max: float | None = None,
) -> Dict[str, float | bool]:
    """ROPE‑style trend‑recovery gate on doubling months.

    The gate considers a relative window around ``true_dm_months``:
    ``[true/γ, true*γ]`` where ``γ = rel_factor``. It computes the probability mass
    inside this window from the posterior draws and optionally enforces a bound on the
    relative CI width.

    Args:
        trend_info: Dict with keys ``{"doubling_months_draws", "dm_ci", "dm_median"}``.
        true_dm_months: Target doubling‑months to recover.
        rel_factor: Window factor ``γ`` for the ROPE.
        min_prob_in_window: Minimum required probability mass inside the window.
        rel_width_max: Optional upper bound for relative CI width ``(hi − lo)/median``.

    Returns:
        Dict with pass flag and diagnostics: ``{"pass_", "p_in", "rel_width", "lo", "hi",
        "rel_factor", "true_dm"}``.
    """
    draws = trend_info.get("doubling_months_draws")
    dm_ci = trend_info.get("dm_ci")
    dm_med_raw = trend_info.get("dm_median")
    v = np.asarray(draws, dtype=float) if draws is not None else np.array([])

    # Guard against empty or completely non‑finite draws.
    if v.size == 0 or not np.isfinite(v).any() or not np.isfinite(true_dm_months):
        return dict(pass_=False, p_in=float("nan"), rel_width=float("nan"))

    # Define the ROPE window on linear scale around the target doubling months.
    gamma = float(rel_factor)
    lo = float(true_dm_months) / gamma
    hi = float(true_dm_months) * gamma
    p_in = float(np.mean((v >= lo) & (v <= hi)))

    # Relative width check from CI if available.
    rel_width_ok = True
    rel_width = float("nan")
    if (
        isinstance(dm_ci, (list, tuple))
        and len(dm_ci) == 2
        and np.isfinite(dm_ci[0])
        and np.isfinite(dm_ci[1])
        and dm_med_raw is not None
        and isinstance(dm_med_raw, (int, float))
        and np.isfinite(float(dm_med_raw))
    ):
        width = float(dm_ci[1]) - float(dm_ci[0])
        dm_med = float(dm_med_raw)
        rel_width = width / dm_med if dm_med > 0 else float("inf")
        if rel_width_max is not None and np.isfinite(rel_width_max):
            rel_width_ok = bool(np.isfinite(rel_width) and (rel_width <= float(rel_width_max)))

    passed = bool((p_in >= float(min_prob_in_window)) and rel_width_ok)
    return dict(
        pass_=passed,
        p_in=p_in,
        lo=lo,
        hi=hi,
        rel_factor=gamma,
        rel_width=rel_width,
        true_dm=true_dm_months,
    )
