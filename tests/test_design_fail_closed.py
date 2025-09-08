import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_eval_candidate_seed_fail_closed(monkeypatch):
    # Import inside to patch local symbols used by _eval_candidate_seed
    import cch_sim.design_search as ds
    from cch_sim.config import SimConfig

    # Base cfg with minimal valid covert-overhead schema
    cfg = SimConfig(
        n_t_bins=1,
        tasks_per_bin=1,
        t_seconds_min=30,
        t_seconds_max=120,
        c_over_bins=[{"lo_s": 5, "hi_s": 10}],
        c_over_mix_by_t_bin=[[1.0]],
    )

    # Light generators
    monkeypatch.setattr(ds, "generate_tasks", lambda **kw: pd.DataFrame({
        "task_id": ["T0"],
        "log_t_baseline_true": [np.log(60.0)],
        "delta_true": [np.log(1.1)],
        "c_overhead_s": [6.0],
    }))
    monkeypatch.setattr(ds, "simulate_humans", lambda **kw: pd.DataFrame({
        "participant_id": ["P0"], "task_id": ["T0"], "condition": ["T"], "log_t_obs": [np.log(60.0)], "censored": [0]
    }))
    # Stage 1 posterior
    S = 16
    h_draws = {"tasks": ["T0"], "draws": S, "tT_s": np.full((1, S), 900.0), "Delta_s": np.full((1, S), 100.0)}
    monkeypatch.setattr(ds, "sample_humans_posterior", lambda *a, **k: h_draws)
    # Models and attempts
    monkeypatch.setattr(ds, "generate_models", lambda cfg, rng: [dict(model_id="m0", release_month=202401)])
    monkeypatch.setattr(ds, "simulate_model_attempts", lambda **kw: pd.DataFrame({
        "model_id": ["m0"], "task_id": ["T0"], "runtime_s": [60.0], "d_seconds": [6.0], "success": [1]
    }))
    monkeypatch.setattr(ds, "apply_detection_models", lambda df, mon: df.assign(monitor_id=mon.get("id", "M0")))

    # Force Stage 2 failure
    monkeypatch.setattr(ds, "sample_models_posterior", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Stage2 failure")))

    gates = {"delta50_max_ci_width_s": 60.0, "d50_min_fraction_in_range": 0.8, "delta50_true_doubling_months": 6.0}
    knobs = {"attempts_per_pair": 1, "tasks_per_bin": 1, "n_participants": 1, "repeats_per_condition": 1}

    g_pass, minutes, n_models = ds._eval_candidate_seed(cfg, knobs, gates, seed=0)

    # All active gates should be False when failure occurs
    assert all(v is False for v in g_pass.values()) and len(g_pass) >= 2
    # minutes should be fallback-based and positive
    assert minutes > 0.0
    # model count still derived from generate_models
    assert n_models == 1

