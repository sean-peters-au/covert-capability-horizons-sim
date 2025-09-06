import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def numpyro_available():
    try:
        import numpyro  # noqa: F401
        import arviz  # noqa: F401
        return True
    except Exception:
        return False


def test_bayes_trend_gate_probability_strong_signal():
    if not numpyro_available():
        import pytest
        pytest.skip("NumPyro/ArviZ not installed in test env")

    from cch_sim.pipeline import sample_trend_posterior
    # Construct synthetic Δ50 draws per model with strong improvement
    # Two models at 2024-01 and 2025-01, Δ50 doubles over the year (doubling months ~ 12)
    d50_m0 = np.exp(np.random.normal(np.log(300.0), 0.05, size=300))
    d50_m1 = np.exp(np.random.normal(np.log(600.0), 0.05, size=300))
    h50_draws = {"delta50_s_draws": {"M0:m0": d50_m0, "M0:m1": d50_m1}}
    release = {"m0": 202401, "m1": 202501}
    tr = sample_trend_posterior(h50_draws, release, priors=dict(seed=0, num_warmup_trend=300, num_samples_trend=300))
    d = tr["trend"]["M0"]
    assert np.isfinite(d.get("slope_median", np.nan))
    assert d.get("dm_median", np.inf) < 24.0
