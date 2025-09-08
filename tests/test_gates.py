import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.gates import delta50_precision_gate, delta50_in_range_gate, trend_recovery_rope_gate


def test_delta50_precision_gate_pass_when_width_small():
    rng = np.random.default_rng(0)
    # Tight draws around 600s; CI width should be well under 60s
    draws = np.exp(rng.normal(np.log(600.0), 0.02, size=2000))
    g = delta50_precision_gate(draws, max_width_seconds=60.0)
    assert g["pass_"] is True
    assert 0.0 < g["width"] <= 60.0


def test_delta50_precision_gate_fail_when_width_large():
    rng = np.random.default_rng(1)
    # Wide draws; CI width should exceed 300s
    draws = np.exp(rng.normal(np.log(600.0), 0.40, size=2000))
    g = delta50_precision_gate(draws, max_width_seconds=300.0)
    assert g["pass_"] is False
    assert g["width"] > 300.0


def test_delta50_in_range_gate_pass():
    rng = np.random.default_rng(2)
    # Δ50 draws centered at 600s, range 400..800
    draws = np.exp(rng.normal(np.log(600.0), 0.08, size=2000))
    g = delta50_in_range_gate(draws, min_delta_seconds=400.0, max_delta_seconds=800.0, min_fraction_in_range=0.8)
    assert g["pass_"] is True
    assert g["frac_in_range"] >= 0.8
    assert 400.0 <= g["median"] <= 800.0


def test_delta50_in_range_gate_fail():
    rng = np.random.default_rng(3)
    # Δ50 draws mostly outside the 200..300 band
    draws = np.exp(rng.normal(np.log(800.0), 0.10, size=2000))
    g = delta50_in_range_gate(draws, min_delta_seconds=200.0, max_delta_seconds=300.0, min_fraction_in_range=0.8)
    assert g["pass_"] is False
    assert g["frac_in_range"] < 0.8



def test_trend_recovery_rope_gate_prob_window():
    import numpy as np
    # synthetic dm draws centered near 6 with moderate spread
    draws = np.exp(np.random.normal(np.log(6.0), 0.1, size=1000))
    trend_info = {"doubling_months_draws": draws, "dm_ci": list(np.percentile(draws, [2.5, 97.5])), "dm_median": float(np.median(draws))}
    # Window factor γ=1.33 → [4.5, 8.0]; expect high probability mass inside
    g = trend_recovery_rope_gate(trend_info, true_dm_months=6.0, rel_factor=1.33, min_prob_in_window=0.6, rel_width_max=0.8)
    assert g["pass_"] is True and g["p_in"] >= 0.6
