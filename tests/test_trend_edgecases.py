import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_trend_single_model_edgecase(monkeypatch):
    from cch_sim import pipeline as pl

    # Avoid requiring numpyro for this test (trend doesn't actually use it)
    monkeypatch.setattr(pl, "_require_numpyro", lambda: None)

    draws = {"delta50_s_draws": {"M0:m0": np.exp(np.random.normal(np.log(300.0), 0.05, size=50))}}
    release = {"m0": 202401}
    out = pl.sample_trend_posterior(draws, release, priors={})
    d = out["trend"]["M0"]
    # With a single model, slopes and dm_draws arrays are empty → dm_median = inf, dm_ci = [nan, nan]
    assert np.isinf(d.get("dm_median", np.inf))
    lo, hi = d.get("dm_ci", [np.nan, np.nan])
    assert not np.isfinite(lo) and not np.isfinite(hi)


def test_trend_empty_series_no_crash(monkeypatch):
    from cch_sim import pipeline as pl

    monkeypatch.setattr(pl, "_require_numpyro", lambda: None)
    draws = {"delta50_s_draws": {}}
    out = pl.sample_trend_posterior(draws, {}, priors={})
    assert isinstance(out, dict) and "trend" in out and out["trend"] == {}
