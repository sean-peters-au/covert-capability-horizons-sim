import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.core.tasks import generate_tasks


def test_generate_tasks_shapes_and_ranges():
    df = generate_tasks(
        seed=0,
        n_t_bins=2,
        tasks_per_bin=5,
        t_seconds_min=10,
        t_seconds_max=100,
        sigma_task=0.1,
        c_over_bins=[{"lo_s": 1, "hi_s": 3}, {"lo_s": 3, "hi_s": 9}],
        c_over_mix_by_t_bin=[[0.8, 0.2], [0.2, 0.8]],
        c_over_sample="uniform",
    )
    # Row count
    assert df.shape[0] == 2 * 5
    # c_overhead_s within configured ranges
    assert (df["c_overhead_s"] >= 1.0 - 1e-9).all() and (df["c_overhead_s"] <= 9.0 + 1e-9).all()
    # delta_true computed as log(1 + C/T)
    t_T = np.exp(df["log_t_baseline_true"].astype(float).to_numpy())
    c = df["c_overhead_s"].astype(float).to_numpy()
    delta_calc = np.log(1.0 + (c / np.maximum(t_T, 1e-12)))
    assert np.allclose(delta_calc, df["delta_true"].astype(float).to_numpy(), rtol=1e-6, atol=1e-9)


def test_c_overhead_sampling_uniform_vs_log_uniform():
    # Single bin, many tasks: medians should differ markedly
    n = 400
    base_kwargs = dict(
        seed=1,
        n_t_bins=1,
        tasks_per_bin=n,
        t_seconds_min=10,
        t_seconds_max=100,
        sigma_task=0.1,
        c_over_bins=[{"lo_s": 1, "hi_s": 100}],
        c_over_mix_by_t_bin=[[1.0]],
    )
    df_u = generate_tasks(**base_kwargs, c_over_sample="uniform")
    df_l = generate_tasks(**base_kwargs, c_over_sample="log_uniform")
    med_u = float(np.median(df_u["c_overhead_s"]))
    med_l = float(np.median(df_l["c_overhead_s"]))
    # Uniform median near ~50, log-uniform median near ~10
    assert med_u > med_l
    assert med_u > 30 and med_l < 20
