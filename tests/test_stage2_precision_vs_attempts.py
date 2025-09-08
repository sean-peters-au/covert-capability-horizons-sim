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


def test_delta50_precision_improves_with_more_attempts():
    if not numpyro_available():
        import pytest
        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import sample_models_posterior

    rng = np.random.default_rng(11)
    # Ground-truth logistic: logit p = theta0 + theta1 * Δ, theta1 < 0
    theta0_true = 6.0
    theta1_true = -0.01

    def make_attempts(n):
        xs = rng.uniform(200.0, 1000.0, size=n)
        logits = theta0_true + theta1_true * xs
        ps = 1.0 / (1.0 + np.exp(-logits))
        ys = (rng.random(xs.shape[0]) < ps).astype(int)
        return pd.DataFrame({
            "monitor_id": ["M0"] * xs.shape[0],
            "model_id": ["m0"] * xs.shape[0],
            "task_id": ["T0"] * xs.shape[0],
            "d_seconds": xs,
            "success": ys,
        })

    # Provide minimal Stage-1 draws (unused for Δ50 estimation)
    S = 400
    humans_draws = {"draws": S, "tT_s": np.tile(np.array([[900.0]], dtype=float), (1, S))}

    small = make_attempts(600)
    big = make_attempts(1800)

    draws_small = sample_models_posterior(
        small,
        humans_draws,
        priors=dict(seed=0, num_warmup=400, num_samples=600, num_chains=1, target_accept=0.95, max_tree_depth=14),
    )
    draws_big = sample_models_posterior(
        big,
        humans_draws,
        priors=dict(seed=1, num_warmup=400, num_samples=600, num_chains=1, target_accept=0.95, max_tree_depth=14),
    )

    import numpy as _np
    d50_small = _np.asarray(draws_small["delta50_s_draws"]["M0:m0"], float)
    d50_big = _np.asarray(draws_big["delta50_s_draws"]["M0:m0"], float)
    w_small = float(_np.percentile(d50_small, 97.5) - _np.percentile(d50_small, 2.5))
    w_big = float(_np.percentile(d50_big, 97.5) - _np.percentile(d50_big, 2.5))

    # Expect narrower CI with more attempts
    assert w_big < w_small
    # And typically a substantial reduction
    assert w_big <= 0.9 * w_small

