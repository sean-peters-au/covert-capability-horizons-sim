import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.core.detection import apply_detection_models


def _df_attempts(rows):
    return pd.DataFrame(rows)


def test_detection_success_bit_controls():
    # det_b1=0 => p_det = b0 (sigmoid(logit(fpr))) per band
    mon = {"id": "M0", "fpr_target": 0.5, "det_b1": 0.0, "eta": 0.0}
    rows = [
        # u_cov < p_cov → covert ok; success toggles on u_det < p_det
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=10.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.2,
        ),
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=10.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.8,
        ),
    ]
    df = apply_detection_models(_df_attempts(rows), mon)
    # p_det = 0.5 for both rows; detected when u_det < 0.5
    assert df.loc[df.index[0], "success"] == 0  # detected
    assert df.loc[df.index[1], "success"] == 1  # not detected


def test_detection_band_fpr_targets_ordering():
    # Keep det_b1=0 to isolate band intercepts
    mon = {"id": "M0", "fpr_short": 0.1, "fpr_med": 0.2, "fpr_long": 0.3, "det_b1": 0.0, "eta": 0.0}
    rows = [
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=60.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.2,
        ),  # short, p_det=0.1 < u_det -> not detected
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=300.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.2,
        ),  # med,   p_det=0.2 == u_det -> not detected (strict <)
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=1800.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.2,
        ),  # long,  p_det=0.3 > u_det -> detected
    ]
    df = apply_detection_models(_df_attempts(rows), mon)
    # Success = covert_ok and not detected
    assert df.loc[df.index[0], "success"] == 1
    assert df.loc[df.index[1], "success"] == 1
    assert df.loc[df.index[2], "success"] == 0


def test_detection_time_dilation_effect_same_band():
    # Medium band (120..600s). Shorter runtime should be more detectable when eta>0.
    mon = {"id": "M0", "fpr_target": 0.5, "det_b1": 0.1, "eta": 1.0}
    rows = [
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=150.0,
            delta_true=0.1,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.53,
        ),
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=500.0,
            delta_true=0.1,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.53,
        ),
    ]
    df = apply_detection_models(_df_attempts(rows), mon)
    # First row (shorter runtime) should be detected more often -> success lower than second.
    assert df.loc[df.index[0], "success"] == 0  # detected -> not successful
    assert df.loc[df.index[1], "success"] == 1  # not detected -> successful


def test_detection_empty_df_returns_empty_with_columns():
    df = apply_detection_models(pd.DataFrame(), {"id": "M0"})
    assert df.shape[0] == 0
    assert "success" in df.columns and "monitor_id" in df.columns
