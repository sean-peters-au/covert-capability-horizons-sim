import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_trend_recovery_noise_free(monkeypatch):
    from cch_sim import pipeline as pl

    # Avoid requiring numpyro for this test (trend itself doesn't use it)
    monkeypatch.setattr(pl, "_require_numpyro", lambda: None)

    # Known exponential trend: Δ50 doubles every D months
    D = 6.0
    S = 200
    models = ["m0", "m1", "m2", "m3", "m4"]
    releases = {"m0": 202401, "m1": 202407, "m2": 202501, "m3": 202507, "m4": 202601}

    def months_since(start, yyyymm):
        y0, m0 = divmod(start, 100)
        y1, m1 = divmod(yyyymm, 100)
        return (y1 - y0) * 12 + (m1 - m0)

    d0 = 10.0
    vals = {}
    for mid in models:
        ms = months_since(releases[models[0]], releases[mid])
        v = d0 * (2.0 ** (ms / D))
        vals[mid] = np.full(S, v, dtype=float)

    draws = {"delta50_s_draws": {f"M0:{mid}": vals[mid] for mid in models}}
    out = pl.sample_trend_posterior(draws, releases, priors={})
    d = out["trend"]["M0"]
    dm_med = float(d.get("dm_median", np.inf))
    lo, hi = d.get("dm_ci", [np.nan, np.nan])

    assert 0.9 * D <= dm_med <= 1.1 * D
    # CI should be very tight under noise-free per-draw values
    assert np.isfinite(lo) and np.isfinite(hi)
    assert (hi - lo) <= 0.1 * D
