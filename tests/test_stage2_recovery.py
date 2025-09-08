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


def test_stage2_delta50_recovery_from_attempts():
    if not numpyro_available():
        import pytest

        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import sample_models_posterior

    rng = np.random.default_rng(7)
    # True logistic on Δ seconds: logit p = theta0 + theta1 * Δ, with theta1 < 0
    theta0_true = 6.0
    theta1_true = -0.01  # monotone decreasing
    d50_true = -theta0_true / theta1_true  # expected ≈ 600s

    # Simulate attempts across a bracketed range around d50
    xs = rng.uniform(200.0, 1000.0, size=1200)
    logits = theta0_true + theta1_true * xs
    ps = 1.0 / (1.0 + np.exp(-logits))
    ys = (rng.random(xs.shape[0]) < ps).astype(int)

    attempts = pd.DataFrame(
        {
            "monitor_id": ["M0"] * xs.shape[0],
            "model_id": ["m0"] * xs.shape[0],
            "task_id": ["T0"] * xs.shape[0],
            "d_seconds": xs,
            "success": ys,
        }
    )

    # Provide minimal Stage-1 draws (only used to compute H50; we test Δ50 here)
    S = 800
    tT_const = 900.0  # arbitrary baseline seconds
    humans_draws = {
        "draws": S,
        "tT_s": np.tile(np.array([[tT_const]], dtype=float), (1, S)),  # [n_tasks=1, S]
    }

    draws_m = sample_models_posterior(
        attempts,
        humans_draws,
        priors=dict(
            seed=0,
            num_warmup=600,
            num_samples=800,
            num_chains=1,
            target_accept=0.95,
            max_tree_depth=14,
        ),
    )

    key = "M0:m0"
    assert key in draws_m["delta50_s_draws"], "missing Δ50 draws"
    d50_draws = np.asarray(draws_m["delta50_s_draws"][key], dtype=float)
    d50_med = float(np.median(d50_draws))

    # Checks
    # 1) Δ50 median within data range
    xmin, xmax = float(xs.min()), float(xs.max())
    assert xmin <= d50_med <= xmax
    # 2) Δ50 median close to ground truth (tolerance 100s for small posterior)
    assert abs(d50_med - d50_true) <= 100.0, (d50_med, d50_true)
