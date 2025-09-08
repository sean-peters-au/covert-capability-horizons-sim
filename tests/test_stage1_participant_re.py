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


def test_stage1_infers_participant_random_effect():
    if not numpyro_available():
        import pytest

        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import sample_humans_posterior

    rng = np.random.default_rng(123)

    # Construct synthetic data with clear participant intercept variation
    tasks = ["T0", "T1", "T2", "T3"]
    base_log = {"T0": np.log(300.0), "T1": np.log(600.0), "T2": np.log(900.0), "T3": np.log(1200.0)}
    delta_log = {"T0": 0.30, "T1": 0.40, "T2": 0.35, "T3": 0.45}

    parts = [f"P{i:02d}" for i in range(8)]
    # Participant random intercepts with substantial spread
    u_part = {p: rng.normal(0.0, 0.35) for p in parts}
    sigma_eps = 0.10

    rows = []
    for t in tasks:
        for p in parts:
            # T condition
            yT = base_log[t] + u_part[p] + rng.normal(0.0, sigma_eps)
            rows.append(
                dict(participant_id=p, task_id=t, condition="T", log_t_obs=float(yT), censored=0)
            )
            # T+C condition
            yTC = base_log[t] + delta_log[t] + u_part[p] + rng.normal(0.0, sigma_eps)
            rows.append(
                dict(participant_id=p, task_id=t, condition="T+C", log_t_obs=float(yTC), censored=0)
            )

    humans = pd.DataFrame(rows)

    draws = sample_humans_posterior(
        humans,
        priors=dict(
            seed=0,
            num_warmup=400,
            num_samples=400,
            num_chains=2,
            target_accept=0.95,
            max_tree_depth=14,
            mu_sd=2.0,
            sigma_sd=0.6,
            tau_task_sd=0.4,
            tau_d_sd=0.3,
            tau_part_sd=0.4,
            betaC_loc=0.35,
            betaC_sd=0.3,
        ),
    )

    # Basic sanity on shapes
    tT_s = np.asarray(draws["tT_s"])  # [n_tasks, S]
    Delta_s = np.asarray(draws["Delta_s"])  # [n_tasks, S]
    assert tT_s.shape[0] == len(tasks)
    assert Delta_s.shape[0] == len(tasks)

    # Posterior summaries should include participant RE signal via tau_part (exposed by function)
    post = draws.get("post_summ", {})
    tau_part_med = float(post.get("tau_part_median", float("nan")))
    sigma_med = float(post.get("sigma_median", float("nan")))

    # Expect a non-trivial participant sd and a modest residual noise level
    assert np.isfinite(tau_part_med) and tau_part_med >= 0.15
    assert np.isfinite(sigma_med) and sigma_med <= 0.35
