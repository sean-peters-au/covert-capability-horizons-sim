import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.design_search import _minutes_per_run_from_humans_draws as minutes_from_draws
from cch_sim.design_search import _compute_cost, run as design_run
from cch_sim.config import SimConfig


def test_minutes_per_run_and_cost_helpers():
    # Stage-1 draws: tT=600s, Delta=100s for two tasks
    S = 10
    draws = {
        "tT_s": np.tile([[600.0], [600.0]], (1, S)),
        "Delta_s": np.tile([[100.0], [100.0]], (1, S)),
    }
    mins = minutes_from_draws(draws)
    # Typical seconds per task-run = median(T) + median(T+C) = 600 + 700 = 1300s → ~21.67 min
    assert abs(mins - (1300.0 / 60.0)) < 1e-6

    # Cost arithmetic
    cfg = SimConfig(n_t_bins=2)
    knobs = dict(attempts_per_pair=10, tasks_per_bin=3, n_participants=4, repeats_per_condition=2)
    costs = _compute_cost(
        cfg,
        knobs,
        minutes_per_task_run=mins,
        n_models=5,
        human_hourly_rate=120.0,
        task_prep_hours_per_task=0.5,
        model_cost_per_attempt=0.02,
    )
    # Simple consistency checks
    assert set(["cost_total", "cost_human", "cost_model", "cost_prep"]).issubset(costs.keys())
    assert costs["cost_total"] >= costs["cost_human"]


def test_design_search_run_with_fake_eval(tmp_path, monkeypatch):
    from cch_sim import design_search as ds

    # Fake candidate evaluation: two seeds → one passes both gates, one fails one gate
    fake_results = [
        (
            {"delta50_precision": True, "delta50_in_range": True, "delta50_trend_rope": True},
            15.0,
            3,
        ),
        (
            {"delta50_precision": False, "delta50_in_range": True, "delta50_trend_rope": True},
            16.0,
            3,
        ),
    ]

    def fake_eval(base_cfg, knobs, gates, seed):
        return fake_results[seed % 2]

    monkeypatch.setattr(ds, "_eval_candidate_seed", fake_eval)

    cfg = SimConfig(
        n_t_bins=2,
        tasks_per_bin=2,
        n_participants=2,
        repeats_per_condition=1,
        t_seconds_min=30,
        t_seconds_max=120,
        c_over_bins=[{"lo_s": 1, "hi_s": 2}],
        c_over_mix_by_t_bin=[[1.0], [1.0]],
    )
    gates = {
        "delta50_max_ci_width_s": 60.0,
        "d50_min_fraction_in_range": 0.8,
        "delta50_true_doubling_months": 6.0,
    }
    search = {
        "attempts_per_pair": [10],
        "tasks_per_bin": [2],
        "n_participants": [2],
        "repeats_per_condition": [1],
        "seeds": 2,
        "base_seed": 0,
    }

    best, rows = design_run(cfg, gates, search, str(tmp_path))

    # Files exist
    assert (tmp_path / "design_search.csv").exists()
    assert (tmp_path / "pareto.csv").exists()
    assert (tmp_path / "best_design.json").exists()

    # CSV columns
    import pandas as pd

    df = pd.read_csv(tmp_path / "design_search.csv")
    assert set(["assurance", "cost_total", "gate_delta50_in_range_rate"]).issubset(df.columns)
