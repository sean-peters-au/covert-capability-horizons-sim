from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import sigmoid


def _ym_to_frac_year(yyyymm: int) -> float:
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return y + (m - 1) / 12.0


def generate_models(cfg: Dict, rng: np.random.Generator) -> List[Dict]:
    mode = str(cfg.get("models_mode", "custom")).lower()
    if mode == "custom":
        out = []
        for i, m in enumerate(cfg.get("models", [])):
            # Accept ability in seconds (preferred) or minutes (legacy); convert to seconds.
            if "ability_seconds" in m:
                ability_seconds = float(m.get("ability_seconds", 0.0))
            elif "ability_minutes" in m:
                ability_seconds = 60.0 * float(m.get("ability_minutes", 0.0))
            else:
                # legacy ambiguous 'ability' treated as seconds if positive, else minutes if small magnitude
                raw = float(m.get("ability", 0.0))
                ability_seconds = raw if raw > 10.0 else 60.0 * raw
            out.append(
                dict(
                    model_id=m.get("model_id", f"model_{i + 1}"),
                    release_month=int(m.get("release_month", 202401)),
                    ability_seconds=float(ability_seconds),
                )
            )
        return out
    if mode == "trend":
        n = int(cfg.get("n_models_auto", 5))
        start = int(cfg.get("trend_start_month", 202101))
        end = int(cfg.get("trend_end_month", 202508))
        months = np.linspace(_ym_to_frac_year(start), _ym_to_frac_year(end), n)
        D = float(cfg.get("trend_doubling_months", 6.0))
        d0 = float(cfg.get("trend_start_delta_s", cfg.get("trend_start_h50_s", 10.0)))
        noise_sd = float(cfg.get("trend_noise_sd_log", 0.12))
        out = []
        for i, x in enumerate(months):
            months_since = (x - months[0]) * 12.0
            d50 = d0 * (2.0 ** (months_since / D))  # seconds
            if noise_sd > 0:
                d50 = float(np.exp(np.log(max(d50, 1e-9)) + rng.normal(0.0, noise_sd)))
            # recover yyyymm from fractional year approx
            year = int(x)
            mon = 1 + int(round((x - year) * 12.0))
            rm = int(year * 100 + max(1, min(12, mon)))
            out.append(
                dict(model_id=f"model_{i + 1:02d}", release_month=rm, ability_seconds=float(d50))
            )
        return out
    # fallback
    return generate_models(dict(cfg, models_mode="custom"), rng)


def simulate_model_attempts(
    *,
    rng: np.random.Generator,
    tasks: pd.DataFrame,
    models: List[Dict],
    sc_alpha: float,
    sc_beta: float,
    attempts_per_pair: int,
) -> pd.DataFrame:
    rows = []
    for m in models:
        for _, trow in tasks.iterrows():
            # Absolute covert overhead seconds for this task
            d_seconds = float(
                trow.get(
                    "c_overhead_s",
                    np.exp(trow.log_t_baseline_true) * (np.exp(trow.delta_true) - 1.0),
                )
            )
            ability_sec = float(m.get("ability_seconds", m.get("ability", 30.0)))
            p_cov = float(sigmoid(sc_alpha + sc_beta * (ability_sec - d_seconds)))
            runtime_s = float(np.exp(trow.log_t_baseline_true + trow.delta_true))
            for a in range(attempts_per_pair):
                rows.append(
                    dict(
                        model_id=m["model_id"],
                        release_month=int(m["release_month"]),
                        task_id=trow.task_id,
                        attempt_id=a,
                        runtime_s=runtime_s,
                        d_seconds=d_seconds,
                        delta_true=float(trow.delta_true),
                        u_cov=float(rng.random()),
                        u_det=float(rng.random()),
                        p_cov=float(p_cov),
                    )
                )
    return pd.DataFrame(rows)
