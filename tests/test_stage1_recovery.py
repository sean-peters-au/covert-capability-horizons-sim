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


def test_stage1_task_baseline_and_delta_recovery_small():
    if not numpyro_available():
        import pytest
        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import sample_humans_posterior

    rng = np.random.default_rng(42)
    # Ground truth (log-seconds)
    tasks = ["T0", "T1"]
    base_log = {"T0": np.log(300.0), "T1": np.log(900.0)}  # 5 min and 15 min baselines
    delta_log = {"T0": 0.30, "T1": 0.45}  # covert addends on log-time
    participants = [f"P{i:02d}" for i in range(6)]

    rows = []
    for tid in tasks:
        for pid in participants:
            # T
            yT = base_log[tid] + rng.normal(0.0, 0.10)
            rows.append(dict(participant_id=pid, task_id=tid, condition="T", log_t_obs=float(yT), censored=0))
            # T+C
            yTC = base_log[tid] + delta_log[tid] + rng.normal(0.0, 0.10)
            rows.append(dict(participant_id=pid, task_id=tid, condition="T+C", log_t_obs=float(yTC), censored=0))

    humans = pd.DataFrame(rows)

    draws_h = sample_humans_posterior(
        humans,
        priors=dict(
            seed=0,
            num_warmup=300,
            num_samples=300,
            num_chains=2,
            target_accept=0.95,
            max_tree_depth=14,
            mu_sd=2.0,
            sigma_sd=0.6,
            tau_task_sd=0.4,
            tau_d_sd=0.3,
            betaC_loc=0.3,
            betaC_sd=0.3,
        ),
    )

    # Posterior medians per task
    tT_s = np.asarray(draws_h["tT_s"])  # [n_tasks, S]
    Delta_s = np.asarray(draws_h["Delta_s"])  # [n_tasks, S]
    assert tT_s.shape[0] == 2 and Delta_s.shape[0] == 2

    med_t = np.median(tT_s, axis=1)
    med_D = np.median(Delta_s, axis=1)

    # Ground truth seconds
    truth_t = np.array([np.exp(base_log["T0"]), np.exp(base_log["T1"])], dtype=float)
    truth_D = truth_t * (np.exp(np.array([delta_log["T0"], delta_log["T1"]], dtype=float)) - 1.0)

    # Tolerances: allow 25% relative error on medians for small N MCMC
    rel_err_t = np.abs(med_t - truth_t) / truth_t
    rel_err_D = np.abs(med_D - truth_D) / np.maximum(truth_D, 1e-9)

    assert np.all(rel_err_t <= 0.25), f"tT medians off: {rel_err_t}"
    assert np.all(rel_err_D <= 0.30), f"Delta medians off: {rel_err_D}"

