import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_sweep_in_range_gate_fail(tmp_path, monkeypatch):
    from cch_sim import cli as cli_mod

    # Light generators
    monkeypatch.setattr(cli_mod, "generate_tasks", lambda **kw: pd.DataFrame({"task_id": ["T0"]}))
    monkeypatch.setattr(cli_mod, "simulate_humans", lambda **kw: pd.DataFrame({"participant_id": ["P0"], "task_id": ["T0"], "condition": ["T"], "log_t_obs": [1.0], "censored": [0]}))

    # Stage 1 posterior
    S = 16
    def fake_sample_humans_posterior(humans, priors=None):
        return {"tasks": ["T0"], "draws": S, "tT_s": np.full((1, S), 900.0), "Delta_s": np.full((1, S), 100.0)}
    monkeypatch.setattr(cli_mod, "sample_humans_posterior", fake_sample_humans_posterior)

    # One model
    monkeypatch.setattr(cli_mod, "generate_models", lambda cfg, rng: [dict(model_id="m0", release_month=202401)])

    # Attempts Δ in narrow range [100, 200]
    attempts = pd.DataFrame({
        "model_id": ["m0", "m0", "m0", "m0"],
        "task_id": ["T0", "T0", "T0", "T0"],
        "runtime_s": [10.0, 10.0, 10.0, 10.0],
        "d_seconds": [100.0, 120.0, 180.0, 200.0],
        "success": [1, 1, 0, 0],
    })
    monkeypatch.setattr(cli_mod, "simulate_model_attempts", lambda **kw: attempts)
    monkeypatch.setattr(cli_mod, "apply_detection_models", lambda df, mon: df)

    # Stage 2 posterior Δ50 centered far outside observed Δ range
    def fake_sample_models_posterior(models_mon, humans_draws, priors=None):
        key = "M0:m0"
        d50 = np.full(64, 350.0)  # outside [100,200]
        return {"delta50_s_draws": {key: d50}}
    monkeypatch.setattr(cli_mod, "sample_models_posterior", fake_sample_models_posterior)
    monkeypatch.setattr(cli_mod, "sample_trend_posterior", lambda *a, **k: {"trend": {}})

    scenarios = {
        "scenarios": [
            {
                "name": "demo_fail",
                "cfg": {
                    "seed": 1,
                    "n_t_bins": 1,
                    "tasks_per_bin": 1,
                    "t_seconds_min": 10,
                    "t_seconds_max": 20,
                    "c_over_bins": [{"lo_s": 5, "hi_s": 10}],
                    "c_over_mix_by_t_bin": [[1.0]],
                    "models_mode": "custom",
                    "models": [{"model_id": "m0", "release_month": 202401, "ability_seconds": 100.0}],
                    "monitors": [{"id": "M0"}],
                    "attempts_per_pair": 1,
                },
            }
        ],
        "gates": {
            "delta50_max_ci_width_s": 60.0,
            "d50_min_fraction_in_range": 0.8,
        },
    }
    p = tmp_path / "scenarios.json"
    p.write_text(json.dumps(scenarios))

    outdir = tmp_path / "out"
    args = type("Args", (), {"scenarios": str(p), "out": str(outdir), "replicates": 2})()
    cli_mod.cmd_sweep(args)

    df = pd.read_csv(outdir / "assurance.csv")
    row = df.iloc[0]
    # In-range gate should fail in both seeds
    assert abs(float(row["gate_delta50_in_range_rate"])) < 1e-12

