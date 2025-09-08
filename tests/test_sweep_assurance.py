import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_sweep_writes_assurance_files_with_expected_content(tmp_path, monkeypatch):
    # Import target module
    from cch_sim import cli as cli_mod

    # Monkeypatch heavy functions with lightweight fakes
    monkeypatch.setattr(cli_mod, "generate_tasks", lambda **kw: pd.DataFrame({"task_id": ["T0"]}))
    monkeypatch.setattr(
        cli_mod,
        "simulate_humans",
        lambda **kw: pd.DataFrame(
            {
                "participant_id": ["P0"],
                "task_id": ["T0"],
                "condition": ["T"],
                "log_t_obs": [1.0],
                "censored": [0],
            }
        ),
    )

    # Stage 1 posterior: 1 task, 16 draws
    def fake_sample_humans_posterior(humans, priors=None):
        S = 16
        return {
            "tasks": ["T0"],
            "draws": S,
            "tT_s": np.full((1, S), 900.0),
            "Delta_s": np.full((1, S), 100.0),
        }

    monkeypatch.setattr(cli_mod, "sample_humans_posterior", fake_sample_humans_posterior)

    # Models list
    monkeypatch.setattr(
        cli_mod, "generate_models", lambda cfg, rng: [dict(model_id="m0", release_month=202401)]
    )

    # Attempts: ensure d_seconds has a range to evaluate Δ50 in-range
    def fake_attempts(**kw):
        return pd.DataFrame(
            {
                "model_id": ["m0", "m0"],
                "task_id": ["T0", "T0"],
                "runtime_s": [10.0, 10.0],
                "d_seconds": [400.0, 800.0],
                "success": [1, 0],
            }
        )

    monkeypatch.setattr(cli_mod, "simulate_model_attempts", lambda **kw: fake_attempts())
    monkeypatch.setattr(cli_mod, "apply_detection_models", lambda df, mon: df)

    # Stage 2 posterior: Δ50 draws well inside [400, 800]; H50 draws narrow
    def fake_sample_models_posterior(models_mon, humans_draws, priors=None):
        key = "M0:m0"
        d50 = np.full(32, 600.0)
        return {"delta50_s_draws": {key: d50}}

    monkeypatch.setattr(cli_mod, "sample_models_posterior", fake_sample_models_posterior)

    # No trend needed
    monkeypatch.setattr(cli_mod, "sample_trend_posterior", lambda *a, **k: {"trend": {}})

    # Build scenarios file
    scenarios = {
        "scenarios": [
            {
                "name": "demo",
                "cfg": {
                    "seed": 1,
                    "n_t_bins": 1,
                    "tasks_per_bin": 1,
                    "t_seconds_min": 10,
                    "t_seconds_max": 20,
                    "c_over_bins": [{"lo_s": 5, "hi_s": 10}],
                    "c_over_mix_by_t_bin": [[1.0]],
                    "models_mode": "custom",
                    "models": [
                        {"model_id": "m0", "release_month": 202401, "ability_seconds": 100.0}
                    ],
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
    args = Namespace(scenarios=str(p), out=str(outdir), replicates=2)
    cli_mod.cmd_sweep(args)

    # Assert assurance files exist and content matches expectations
    p_csv = outdir / "assurance.csv"
    p_json = outdir / "assurance.json"
    assert p_csv.exists() and p_json.exists()
    df = pd.read_csv(p_csv)
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["design_id"] == "demo|M0"
    # Both seeds pass both gates under our fakes
    assert abs(float(row["assurance"]) - 1.0) < 1e-9
    assert abs(float(row["gate_delta50_precision_rate"]) - 1.0) < 1e-9
    assert abs(float(row["gate_delta50_in_range_rate"]) - 1.0) < 1e-9
    assert int(row["seeds"]) == 2
