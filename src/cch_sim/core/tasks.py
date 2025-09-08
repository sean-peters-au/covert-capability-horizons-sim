"""Task/time binning and covert-overhead generation.

This module will:
- Build log-time bins between t_min_s and t_max_s.
- Sample task baselines per bin with sigma_task.
- Sample covert-overhead seconds per band using configured mixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TaskRow:
    task_id: str
    bin: int
    log_t_baseline_true: float
    delta_true: float  # covert overhead in log-time units
    c_overhead_s: float


def make_bins(
    t_seconds_min: float, t_seconds_max: float, n_t_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(int(seed))
    edges_log, centers_log = make_bins(t_seconds_min, t_seconds_max, n_t_bins)

    # Prepare covert-overhead ranges and per-T-bin mixtures
    ranges = [(float(r["lo_s"]), float(r["hi_s"])) for r in c_over_bins]

    def _norm(v: List[float]):
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
            else:  # log-uniform
                c_s = float(np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(max(hi, lo + 1e-6)))))
            # Difficulty is human log-time overhead: delta = log(T+C) - log(T) = log(1 + C/T)
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

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df
