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
    # det_b1=0 => p_det = fpr_target
    mon = {"id": "M0", "fpr_target": 0.5, "det_b1": 0.0}
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


def test_detection_fpr_target_monotonicity():
    # With det_b1=0, p_det = fpr_target; higher FPR increases detection (less success)
    rows = [
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=300.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.2,
        )
    ]
    df_low = apply_detection_models(
        _df_attempts(rows), {"id": "M0", "fpr_target": 0.1, "det_b1": 0.0}
    )
    df_mid = apply_detection_models(
        _df_attempts(rows), {"id": "M0", "fpr_target": 0.2, "det_b1": 0.0}
    )
    df_hi = apply_detection_models(
        _df_attempts(rows), {"id": "M0", "fpr_target": 0.3, "det_b1": 0.0}
    )
    # Success is 1 if not detected; as fpr increases, detection increases, so success declines
    s_low = int(df_low.loc[df_low.index[0], "success"])  # likely 1 (0.1 < 0.2)
    s_mid = int(df_mid.loc[df_mid.index[0], "success"])  # edge (0.2 == 0.2) still 1 due to strict <
    s_hi = int(df_hi.loc[df_hi.index[0], "success"])  # 0 (0.3 > 0.2)
    assert s_low >= s_mid >= s_hi


def test_detection_monotone_in_delta():
    # Larger delta_true -> larger ECL ratio -> higher detection probability -> lower success
    mon = {"id": "M0", "fpr_target": 0.2, "det_b1": 0.5}
    rows = [
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=300.0,
            delta_true=0.0,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.3,
        ),
        dict(
            model_id="m0",
            task_id="T0",
            runtime_s=300.0,
            delta_true=0.7,
            u_cov=0.0,
            p_cov=1.0,
            u_det=0.3,
        ),
    ]
    df = apply_detection_models(_df_attempts(rows), mon)
    # Smaller delta → lower p_det → success 1; Larger delta → higher p_det → success 0
    assert df.loc[df.index[0], "success"] == 1
    assert df.loc[df.index[1], "success"] == 0


def test_detection_empty_df_returns_empty_with_columns():
    df = apply_detection_models(pd.DataFrame(), {"id": "M0"})
    assert df.shape[0] == 0
    assert "success" in df.columns and "monitor_id" in df.columns
