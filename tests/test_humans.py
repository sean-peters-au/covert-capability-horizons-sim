import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def numpyro_available():
    try:
        import numpyro  # noqa: F401
        import arviz  # noqa: F401

        return True
    except Exception:
        return False


def test_pipeline_humans_models_trend_small_smoke():
    if not numpyro_available():
        import pytest

        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import (
        sample_humans_posterior,
        sample_models_posterior,
        sample_trend_posterior,
    )

    # Build small but well-identified synthetic humans dataset
    rows = []
    participants = ["P0", "P1", "P2", "P3"]
    rng = np.random.default_rng(0)
    u_part = {pid: float(rng.normal(0.0, 0.12)) for pid in participants}
    deltas = {"T0": 0.25, "T1": 0.35}
    for tid, base in [("T0", 1.0), ("T1", 1.2)]:
        for pid in participants:
            eps_T = float(rng.normal(0.0, 0.20))
            eps_TC = float(rng.normal(0.0, 0.20))
            rows.append(
                dict(
                    participant_id=pid,
                    task_id=tid,
                    condition="T",
                    log_t_obs=base + u_part[pid] + eps_T,
                    censored=0,
                )
            )
            rows.append(
                dict(
                    participant_id=pid,
                    task_id=tid,
                    condition="T+C",
                    log_t_obs=base + deltas[tid] + u_part[pid] + eps_TC,
                    censored=0,
                )
            )
    humans = pd.DataFrame(rows)
    draws_h = sample_humans_posterior(
        humans,
        priors=dict(
            seed=0,
            num_warmup=350,
            num_samples=350,
            num_chains=1,
            target_accept=0.95,
            max_tree_depth=14,
            # tighten hierarchical scales to stabilize tiny-data inference
            tau_task_sd=0.25,
            tau_d_sd=0.20,
            sigma_sd=0.4,
            mu_sd=1.5,
            betaC_loc=0.30,
            betaC_sd=0.15,
        ),
    )
    assert draws_h["tT_s"].shape[0] == 2 and draws_h["Delta_s"].shape[0] == 2

    # Build tiny attempts for one model/monitor
    # Use Δ minutes ~ 20 for mix of success outcomes
    attempts = pd.DataFrame(
        {
            "monitor_id": ["M0"] * 40,
            "model_id": ["m0"] * 40,
            "task_id": ["T0"] * 20 + ["T1"] * 20,
            "d_seconds": [600.0] * 10 + [1800.0] * 10 + [600.0] * 10 + [1800.0] * 10,
            "success": [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10,
        }
    )
    draws_m = sample_models_posterior(
        attempts,
        draws_h,
        priors=dict(
            seed=0,
            num_warmup=400,
            num_samples=800,
            num_chains=1,
            target_accept=0.95,
            max_tree_depth=14,
        ),
    )
    d50 = list(draws_m["delta50_s_draws"].values())[0]
    assert np.isfinite(np.median(d50))

    # Trend over two models
    # Duplicate key structure
    d50_draws = {"delta50_s_draws": {"M0:m0": np.array(d50), "M0:m1": np.array(d50) * 1.2}}
    release = {"m0": 202401, "m1": 202501}
    tr = sample_trend_posterior(
        d50_draws, release, priors=dict(seed=0, num_warmup_trend=200, num_samples_trend=200)
    )
    d = tr["trend"]["M0"]
    assert np.isfinite(d.get("slope_median", float("nan")))
