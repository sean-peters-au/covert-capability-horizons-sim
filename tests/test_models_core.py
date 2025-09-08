import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.core.models import generate_models


def test_generate_models_custom_abilities():
    rng = np.random.default_rng(0)
    cfg = {
        "models_mode": "custom",
        "models": [
            {"model_id": "a", "release_month": 202401, "ability_seconds": 20},
            {"model_id": "b", "release_month": 202402, "ability_minutes": 2},  # 120s
            {"model_id": "c", "release_month": 202403, "ability": 5},  # minutes -> 300s
            {"model_id": "d", "release_month": 202404, "ability": 600},  # seconds (raw > 10)
        ],
    }
    out = generate_models(cfg, rng)
    m = {r["model_id"]: r for r in out}
    assert m["a"]["ability_seconds"] == 20.0
    assert m["b"]["ability_seconds"] == 120.0
    assert m["c"]["ability_seconds"] == 300.0
    assert m["d"]["ability_seconds"] == 600.0


def test_generate_models_trend_zero_noise_monotone():
    rng = np.random.default_rng(1)
    cfg = {
        "models_mode": "trend",
        "n_models_auto": 5,
        "trend_start_month": 202401,
        "trend_end_month": 202501,
        "trend_start_h50_s": 5.0,
        "trend_doubling_months": 6.0,
        "trend_noise_sd_log": 0.0,
    }
    out = generate_models(cfg, rng)
    # Release months within range and increasing ability_seconds
    rms = [r["release_month"] for r in out]
    d50s = [r["ability_seconds"] for r in out]
    assert min(rms) >= 202401 and max(rms) <= 202512
    assert all(d50s[i] <= d50s[i + 1] for i in range(len(d50s) - 1))
