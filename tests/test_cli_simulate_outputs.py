import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_cmd_simulate_writes_expected_outputs(tmp_path, monkeypatch):
    from cch_sim import cli as cli_mod

    # Minimal config
    cfg = {
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
    }

    # Fakes for heavy steps
    monkeypatch.setattr(
        cli_mod,
        "generate_tasks",
        lambda **kw: pd.DataFrame(
            {
                "task_id": ["T0"],
                "bin": [0],
                "log_t_baseline_true": [np.log(10.0)],
                "delta_true": [np.log(1.2)],
                "c_overhead_s": [2.0],
            }
        ),
    )
    # Humans: both conditions present so anchors_snr has data
    humans_df = pd.DataFrame(
        {
            "participant_id": ["P0", "P1"],
            "task_id": ["T0", "T0"],
            "condition": ["T", "T+C"],
            "log_t_obs": [np.log(10.0), np.log(12.0)],
            "censored": [0, 0],
        }
    )
    monkeypatch.setattr(cli_mod, "simulate_humans", lambda **kw: humans_df)

    # Stage 1 posterior
    S = 16
    humans_draws = {
        "tasks": ["T0"],
        "draws": S,
        "tT_s": np.full((1, S), 600.0),
        "Delta_s": np.full((1, S), 100.0),
    }
    monkeypatch.setattr(cli_mod, "sample_humans_posterior", lambda *a, **k: humans_draws)

    # Models
    monkeypatch.setattr(
        cli_mod, "generate_models", lambda cfg, rng: [dict(model_id="m0", release_month=202401)]
    )

    # Attempts through detection: ensure required cols present
    attempts = pd.DataFrame(
        {
            "model_id": ["m0", "m0"],
            "task_id": ["T0", "T0"],
            "runtime_s": [10.0, 10.0],
            "d_seconds": [400.0, 800.0],
            "delta_true": [np.log(1.2), np.log(1.2)],
            "u_cov": [0.0, 0.0],
            "u_det": [1.0, 1.0],
            "p_cov": [1.0, 1.0],
            "success": [1, 0],
        }
    )
    monkeypatch.setattr(cli_mod, "simulate_model_attempts", lambda **kw: attempts)
    monkeypatch.setattr(
        cli_mod, "apply_detection_models", lambda df, mon: df.assign(monitor_id=mon.get("id", "M0"))
    )

    # Stage 2 posterior for Δ50
    d50 = np.full(32, 600.0)
    monkeypatch.setattr(
        cli_mod, "sample_models_posterior", lambda *a, **k: {"delta50_s_draws": {"M0:m0": d50}}
    )

    # Trend posterior
    trend = {
        "trend": {
            "M0": {
                "slope_median": 0.5,
                "slope_ci": [0.1, 0.9],
                "dm_median": 16.0,
                "dm_ci": [12.0, 20.0],
                "doubling_months_draws": np.array([16.0, 18.0, 15.0]),
            }
        }
    }
    monkeypatch.setattr(cli_mod, "sample_trend_posterior", lambda *a, **k: trend)

    p_cfg = tmp_path / "cfg.json"
    p_cfg.write_text(json.dumps(cfg))
    outdir = tmp_path / "run"

    cli_mod.cmd_simulate(type("Args", (), {"config": str(p_cfg), "out": str(outdir)})())

    # Assert outputs exist
    files = [
        "config.normalized.json",
        "delta50_by_monitor.csv",
        "delta50_trend_by_monitor.csv",
        "preflight_coverage.csv",
        "anchors_snr.csv",
        "summary.json",
    ]
    for fn in files:
        assert (outdir / fn).exists(), f"missing {fn}"

    # Basic schema checks
    df_d = pd.read_csv(outdir / "delta50_by_monitor.csv")
    assert set(
        ["monitor_id", "model_id", "release_month", "delta50_s_lo", "delta50_s_med", "delta50_s_hi"]
    ).issubset(df_d.columns)
    df_t = pd.read_csv(outdir / "delta50_trend_by_monitor.csv")
    assert set(["monitor_id", "slope_median", "doubling_months_median"]).issubset(df_t.columns)
