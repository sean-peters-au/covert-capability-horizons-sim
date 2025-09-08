"""Task/time binning and covert‑overhead generation.

Builds synthetic task instances arranged into log‑time bins, and assigns each task a
baseline log‑time and a covert overhead (in seconds) drawn from user‑configured
mixtures. The covert overhead is converted to a human log‑time addend via
``delta_true = log(1 + C / T)`` where ``C`` is covert seconds and ``T`` is the
baseline T‑condition seconds for that task.

Functions provided here are pure and deterministic given the RNG seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TaskRow:
    """A single synthetic task instance.

    Attributes:
        task_id: Task identifier (e.g., "T003").
        bin: Index of the time bin the task belongs to.
        log_t_baseline_true: True baseline log‑seconds for T‑condition.
        delta_true: Human log‑time addend for T+C (``log(1 + C/T)``).
        c_overhead_s: Covert overhead seconds used to construct ``delta_true``.
    """

    task_id: str
    bin: int
    log_t_baseline_true: float
    delta_true: float  # covert overhead in log-time units
    c_overhead_s: float


def make_bins(
    t_seconds_min: float, t_seconds_max: float, n_t_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create log‑time bin edges and centers.

    Args:
        t_seconds_min: Minimum baseline seconds for tasks.
        t_seconds_max: Maximum baseline seconds for tasks.
        n_t_bins: Number of log‑time bins between the min and max.

    Returns:
        A pair of numpy arrays ``(edges_log, centers_log)`` in natural‑log seconds.
    """
    a, b = math.log(t_seconds_min), math.log(t_seconds_max)
    edges = np.linspace(a, b, n_t_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def generate_tasks(
    *,
    seed: int,
    n_t_bins: int,
    tasks_per_bin: int,
    t_seconds_min: float,
    t_seconds_max: float,
    sigma_task: float,
    c_over_bins: List[Dict[str, float]],
    c_over_mix_by_t_bin: List[List[float]],
    c_over_sample: str = "log_uniform",
) -> pd.DataFrame:
    """Generate tasks across log‑time bins and sample covert overhead seconds.

    Args:
        seed: RNG seed.
        n_t_bins: Number of log‑time bins.
        tasks_per_bin: Tasks to draw per bin.
        t_seconds_min: Minimum baseline seconds across tasks.
        t_seconds_max: Maximum baseline seconds across tasks.
        sigma_task: Standard deviation on log‑seconds for task baselines within a bin.
        c_over_bins: List of ``{"lo_s": float, "hi_s": float}`` covert‑seconds ranges.
        c_over_mix_by_t_bin: Mixture weights per T‑bin over the covert ranges (each row sums to > 0).
        c_over_sample: Sampling scheme for covert seconds: ``"uniform"`` or ``"log_uniform"``.

    Returns:
        DataFrame of tasks with baseline log‑time, covert seconds, and ``delta_true``.
    """
    rng = np.random.default_rng(int(seed))
    edges_log, centers_log = make_bins(t_seconds_min, t_seconds_max, n_t_bins)

    # Prepare covert‑overhead ranges and per‑T‑bin mixtures
    ranges = [(float(r["lo_s"]), float(r["hi_s"])) for r in c_over_bins]

    def _norm(v: List[float]) -> np.ndarray:
        arr = np.asarray(v, float)
        s = float(np.sum(arr))
        return (arr / s) if s > 0 else np.full_like(arr, 1.0 / len(arr))

    rows: List[TaskRow] = []
    tid = 0
    for b in range(n_t_bins):
        for _ in range(tasks_per_bin):
            log_t_baseline_true = float(rng.normal(centers_log[b], sigma_task))
            mix = _norm(c_over_mix_by_t_bin[b])
            k = int(rng.choice(np.arange(len(ranges)), p=mix))
            lo, hi = ranges[k]
            if c_over_sample == "uniform":
                c_s = float(rng.uniform(lo, hi))
            else:  # log‑uniform
                c_s = float(np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(max(hi, lo + 1e-6)))))
            # Human log‑time overhead: delta = log(T+C) − log(T) = log(1 + C/T)
            t_T = float(np.exp(log_t_baseline_true))
            delta_true = float(np.log(1.0 + (c_s / max(t_T, 1e-12))))
            rows.append(
                TaskRow(
                    task_id=f"T{tid:03d}",
                    bin=b,
                    log_t_baseline_true=log_t_baseline_true,
                    delta_true=delta_true,
                    c_overhead_s=c_s,
                )
            )
            tid += 1

    return pd.DataFrame([r.__dict__ for r in rows])
