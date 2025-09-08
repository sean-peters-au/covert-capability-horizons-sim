from __future__ import annotations

import copy
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .assurance import assurance_all_gates
from .config import SimConfig
from .core.detection import apply_detection_models
from .core.humans import simulate_humans
from .core.models import generate_models, simulate_model_attempts
from .core.tasks import generate_tasks
from .gates import (
    delta50_in_range_gate,
    delta50_precision_gate,
    trend_recovery_rope_gate,
)
from .pipeline import sample_humans_posterior, sample_models_posterior, sample_trend_posterior


def _mix_seed(base_seed: int, a: int, b: int) -> int:
    """Deterministic 32-bit integer hash from three ints (no Python hash())."""
    x = (int(base_seed) & 0xFFFFFFFF) ^ 0x9E3779B9
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (int(a) & 0xFFFFFFFF) * 0xC2B2AE35 & 0xFFFFFFFF
    x &= 0xFFFFFFFF
    x ^= (int(b) & 0xFFFFFFFF) * 0x27D4EB2F & 0xFFFFFFFF
    return int(x & 0x7FFFFFFF)


def _minutes_per_run_from_humans_draws(draws: Dict[str, Any]) -> float:
    """Compute minutes per task-run = median(T) + median(T+C), then convert to minutes.

    Uses Stage-1 posterior draws: tT_s [n_tasks, S], Delta_s [n_tasks, S] (seconds).
    Returns a single scalar minutes per task-run (typical task), using median across tasks.
    """
    tT = np.asarray(draws.get("tT_s"), float)  # [n_tasks, S]
    D = np.asarray(draws.get("Delta_s"), float)
    if tT.size == 0 or D.size == 0:
        return float("nan")
    # Per-task medians over draws
    med_T = np.median(tT, axis=1)
    med_TC = np.median(tT + D, axis=1)
    per_task_seconds = med_T + med_TC
    typical_seconds = float(np.median(per_task_seconds))
    return typical_seconds / 60.0


def _compute_cost(
    cfg: SimConfig,
    knobs: Dict[str, int],
    minutes_per_task_run: float,
    n_models: int,
    human_hourly_rate: float,
    task_prep_hours_per_task: float,
    model_cost_per_attempt: float,
) -> Dict[str, float]:
    """Minimal explicit cost model (see DESIGN_SEARCH.md)."""
    tasks_per_bin = int(knobs["tasks_per_bin"])  # per T bin
    n_t_bins = int(cfg.n_t_bins)
    n_participants = int(knobs["n_participants"])
    repeats = int(knobs["repeats_per_condition"])
    attempts_per_pair = int(knobs["attempts_per_pair"])

    # Human cost
    human_minutes = minutes_per_task_run * tasks_per_bin * n_t_bins * n_participants * repeats
    cost_human = (human_minutes / 60.0) * float(human_hourly_rate)
    # Model compute cost
    attempts = attempts_per_pair * tasks_per_bin * n_t_bins * int(max(n_models, 1))
    cost_model = float(attempts) * float(model_cost_per_attempt)
    # Task prep cost
    prep_hours = float(task_prep_hours_per_task) * tasks_per_bin * n_t_bins
    cost_prep = prep_hours * float(human_hourly_rate)
    total = float(cost_human + cost_model + cost_prep)
    return dict(
        cost_total=total,
        cost_human=float(cost_human),
        cost_model=float(cost_model),
        cost_prep=float(cost_prep),
    )


def _active_gate_keys(gates: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    if str(gates.get("delta50_max_ci_width_s", "nan")) != "nan":
        keys.append("delta50_precision")
    # Δ50 in-range uses observed Δ from attempts; always tracked if attempts exist
    keys.append("delta50_in_range")
    if (
        str(gates.get("delta50_true_doubling_months", "nan")) != "nan"
        or str(gates.get("true_doubling_months", "nan")) != "nan"
    ):
        keys.append("delta50_trend_rope")
    return keys


def _fallback_minutes_per_run(cfg: SimConfig) -> float:
    # 2*T_median + Δ_median (seconds) → minutes
    t_med = math.sqrt(float(cfg.t_seconds_min) * float(cfg.t_seconds_max))
    if cfg.c_over_bins:
        mids = [
            math.sqrt(float(b.get("lo_s", 1.0)) * float(b.get("hi_s", 1.0)))
            for b in cfg.c_over_bins
        ]
        d_med = float(np.median(mids)) if mids else 0.0
    else:
        d_med = 0.0
    return float((2.0 * t_med + d_med) / 60.0)


def _eval_candidate_seed(
    base_cfg: SimConfig,
    knobs: Dict[str, int],
    gates: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, bool], float, int]:
    """Run one seed for a candidate. Returns (gate_pass_dict, minutes_per_task_run, n_models).

    Fail-closed: if any stage raises (e.g., diagnostics), mark all active gates False and return a fallback minutes estimate.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.attempts_per_pair = int(knobs["attempts_per_pair"])
    cfg.tasks_per_bin = int(knobs["tasks_per_bin"])
    cfg.n_participants = int(knobs["n_participants"])
    cfg.repeats_per_condition = int(knobs["repeats_per_condition"])
    cfg.seed = int(seed)
    cfg.validate()

    try:
        # Generate data
        tasks = generate_tasks(
            seed=cfg.seed,
            n_t_bins=cfg.n_t_bins,
            tasks_per_bin=cfg.tasks_per_bin,
            t_seconds_min=cfg.t_seconds_min,
            t_seconds_max=cfg.t_seconds_max,
            sigma_task=cfg.sigma_task,
            c_over_bins=cfg.c_over_bins or [],
            c_over_mix_by_t_bin=cfg.c_over_mix_by_t_bin or [],
            c_over_sample=cfg.c_over_sample,
        )
        humans = simulate_humans(
            seed=cfg.seed,
            tasks=tasks,
            n_participants=cfg.n_participants,
            repeats_per_condition=cfg.repeats_per_condition,
            sigma_participant=cfg.sigma_participant,
            sigma_noise=cfg.sigma_noise,
            human_skill_sd=cfg.human_skill_sd,
            human_cov_alpha=cfg.human_cov_alpha,
            human_cov_beta=cfg.human_cov_beta,
        )
        h_draws = sample_humans_posterior(
            humans,
            priors={
                "seed": cfg.seed,
                "num_warmup": 400,
                "num_samples": 600,
                "num_chains": 2,
                "target_accept": 0.95,
                "max_tree_depth": 14,
            },
        )
        minutes_per_task_run = _minutes_per_run_from_humans_draws(h_draws)

        # Models and attempts
        models_list = generate_models(cfg.__dict__, np.random.default_rng(cfg.seed))
        n_models = int(len(models_list))
        attempts = simulate_model_attempts(
            rng=np.random.default_rng(cfg.seed),
            tasks=tasks,
            models=models_list,
            sc_alpha=cfg.sc_alpha,
            sc_beta=cfg.sc_beta,
            attempts_per_pair=cfg.attempts_per_pair,
        )
        mon = cfg.monitors[0]
        models_mon = apply_detection_models(
            attempts,
            dict(
                id=mon.id,
                fpr_target=mon.fpr_target,
                fpr_short=mon.fpr_short,
                fpr_med=mon.fpr_med,
                fpr_long=mon.fpr_long,
                det_b1=mon.det_b1,
                eta=mon.eta,
            ),
        )
        models_mon["monitor_id"] = mon.id

        # Stage 2 and trend
        m_draws = sample_models_posterior(
            models_mon,
            h_draws,
            priors={
                "seed": cfg.seed,
                "num_warmup": 600,
                "num_samples": 1200,
                "num_chains": 2,
                "target_accept": 0.95,
                "max_tree_depth": 14,
            },
        )
        rel_map = {m["model_id"]: int(m.get("release_month", 0)) for m in models_list}
        trend = sample_trend_posterior(
            m_draws,
            rel_map,
            priors={"seed": cfg.seed, "num_warmup_trend": 200, "num_samples_trend": 200},
        )

        # Gates
        g_pass: Dict[str, bool] = {}
        # Δ50 precision (pass if any model meets width threshold)
        max_w = float(gates.get("delta50_max_ci_width_s", math.nan))
        if math.isfinite(max_w):
            pass_any = False
            for arr in (m_draws.get("delta50_s_draws", {}) or {}).values():
                g = delta50_precision_gate(np.asarray(arr, float), max_width_seconds=max_w)
                if bool(g.get("pass_", False)):
                    pass_any = True
                    break
            g_pass["delta50_precision"] = pass_any

        # Δ50 in-range (pass if any model has Δ50 mostly within observed Δ)
        if "d_seconds" in models_mon.columns:
            series = models_mon["d_seconds"].astype(float)
        elif "c_overhead_s" in models_mon.columns:
            series = models_mon["c_overhead_s"].astype(float)
        else:
            series = None
        if series is not None:
            lo_d = float(np.nanmin(series))
            hi_d = float(np.nanmax(series))
            min_frac = float(
                gates.get(
                    "d50_min_fraction_in_range", gates.get("delta50_min_fraction_in_range", 0.8)
                )
            )
            pass_any = False
            for arr in (m_draws.get("delta50_s_draws", {}) or {}).values():
                g = delta50_in_range_gate(
                    np.asarray(arr, float),
                    min_delta_seconds=lo_d,
                    max_delta_seconds=hi_d,
                    min_fraction_in_range=min_frac,
                )
                if bool(g.get("pass_", False)):
                    pass_any = True
                    break
            g_pass["delta50_in_range"] = pass_any

        # Trend recovery (ROPE), if true dm provided
        true_dm = float(
            gates.get("delta50_true_doubling_months", gates.get("true_doubling_months", math.nan))
        )
        if math.isfinite(true_dm):
            mon_id = mon.id
            info = (trend.get("trend", {}) or {}).get(mon_id, {})
            rope = trend_recovery_rope_gate(
                info,
                true_dm,
                rel_factor=float(gates.get("delta50_dm_rel_factor", 1.33)),
                min_prob_in_window=float(gates.get("delta50_dm_min_prob_in_window", 0.6)),
                rel_width_max=(
                    float(gates["delta50_dm_rel_width_max"])
                    if str(gates.get("delta50_dm_rel_width_max", "nan")) != "nan"
                    else None
                ),
            )
            g_pass["delta50_trend_rope"] = bool(rope.get("pass_", False))

        return g_pass, float(minutes_per_task_run), n_models

    except Exception:
        # Fail-closed: mark all active gates as False and return fallback minutes and model count
        try:
            n_models = int(len(generate_models(cfg.__dict__, np.random.default_rng(cfg.seed))))
        except Exception:
            n_models = 1
        gkeys = _active_gate_keys(gates)
        g_pass = {k: False for k in gkeys} if gkeys else {"delta50_precision": False}
        return g_pass, float(_fallback_minutes_per_run(cfg)), n_models


def _compute_pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute non-dominated set: minimize cost_total, maximize assurance."""
    pareto: List[Dict[str, Any]] = []
    for r in rows:
        dominated = False
        for q in rows:
            if q is r:
                continue
            # q dominates r if cost_total <= and assurance >= with at least one strict
            if (
                float(q["cost_total"]) <= float(r["cost_total"])
                and float(q["assurance"]) >= float(r["assurance"])
                and (
                    float(q["cost_total"]) < float(r["cost_total"])
                    or float(q["assurance"]) > float(r["assurance"])
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto


def _select_min_cost_feasible(
    rows: List[Dict[str, Any]], assurance_target: float
) -> Tuple[Dict[str, Any] | None, bool]:
    """Return (best_row, feasible). If none meet target, return highest assurance and feasible=False."""
    feas = [r for r in rows if float(r["assurance"]) >= float(assurance_target)]
    if feas:
        best = min(feas, key=lambda r: float(r["cost_total"]))
        return best, True
    # fallback: highest assurance; break ties by lower cost
    if not rows:
        return None, False
    best = max(rows, key=lambda r: (float(r["assurance"]), -float(r["cost_total"])))
    return best, False


def run(
    scenario_cfg: SimConfig, gates: Dict[str, Any], search_cfg: Dict[str, Any], out_dir: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run automated design search.

    - scenario_cfg: base SimConfig for the scenario (will be copied per candidate/seed)
    - gates: dict of gate thresholds (Δ50 precision, identifiability, trend ROPE)
    - search_cfg: dict under 'study_search' with grid, costs, seeds, base_seed
    - out_dir: directory to write outputs

    Returns (best_design_row, all_rows)
    """
    os.makedirs(out_dir, exist_ok=True)

    knobs_grid = dict(
        attempts_per_pair=list(
            map(int, search_cfg.get("attempts_per_pair", [scenario_cfg.attempts_per_pair]))
        ),
        tasks_per_bin=list(map(int, search_cfg.get("tasks_per_bin", [scenario_cfg.tasks_per_bin]))),
        n_participants=list(
            map(int, search_cfg.get("n_participants", [scenario_cfg.n_participants]))
        ),
        repeats_per_condition=list(
            map(int, search_cfg.get("repeats_per_condition", [scenario_cfg.repeats_per_condition]))
        ),
    )
    assurance_target = float(search_cfg.get("assurance_target", 0.8))
    n_seeds = int(search_cfg.get("seeds", 5))
    base_seed = int(search_cfg.get("base_seed", 0))

    human_hourly_rate = float(search_cfg.get("human_hourly_rate", 150.0))
    task_prep_hours_per_task = float(search_cfg.get("task_prep_hours_per_task", 0.5))
    model_cost_per_attempt = float(search_cfg.get("model_cost_per_attempt", 0.0))

    # Enumerate candidates
    keys = ["attempts_per_pair", "tasks_per_bin", "n_participants", "repeats_per_condition"]
    grid_vals = [knobs_grid[k] for k in keys]
    candidates = [dict(zip(keys, vals)) for vals in product(*grid_vals)]

    import pandas as pd

    rows: List[Dict[str, Any]] = []

    # Evaluate each candidate; autoparallel over seeds within candidate, fallback to serial if needed
    from concurrent.futures import ProcessPoolExecutor, as_completed

    for i, cand in enumerate(candidates):
        seed_list = [_mix_seed(base_seed, i, j) for j in range(n_seeds)]
        # Submit tasks
        results: List[Tuple[Dict[str, bool], float, int]] = []
        try:
            with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
                futs = [
                    ex.submit(_eval_candidate_seed, scenario_cfg, cand, gates, s) for s in seed_list
                ]
                for f in as_completed(futs):
                    results.append(f.result())
        except Exception:
            # Fallback to serial
            results = [_eval_candidate_seed(scenario_cfg, cand, gates, s) for s in seed_list]

        # Aggregate assurance
        pass_records = [r[0] for r in results]
        mins_list = [r[1] for r in results]
        n_models_seen = [r[2] for r in results]
        ag = assurance_all_gates(pass_records)
        minutes_per_task_run = float(np.nanmean(mins_list)) if mins_list else float("nan")
        n_models = int(n_models_seen[0]) if n_models_seen else 1

        costs = _compute_cost(
            scenario_cfg,
            cand,
            minutes_per_task_run,
            n_models,
            human_hourly_rate,
            task_prep_hours_per_task,
            model_cost_per_attempt,
        )

        # Extract typed values from assurance aggregate
        a_val = ag.get("assurance")
        per_gate = ag.get("per_gate_rate")
        a_float = float(a_val) if isinstance(a_val, (int, float)) else float("nan")
        per_gate_dict: dict[str, float] = per_gate if isinstance(per_gate, dict) else {}

        row: Dict[str, Any] = {
            **{k: int(cand[k]) for k in keys},
            "assurance": a_float,
            "seeds": int(len(pass_records)),
            **{f"gate_{k}_rate": float(v) for k, v in per_gate_dict.items()},
            **costs,
        }
        rows.append(row)

    # Write design_search.csv
    df = pd.DataFrame(rows)
    p_csv = Path(out_dir) / "design_search.csv"
    df.to_csv(p_csv, index=False)

    # Pareto frontier and best selection
    pareto = _compute_pareto(rows)
    pd.DataFrame(pareto).to_csv(Path(out_dir) / "pareto.csv", index=False)
    best, feasible = _select_min_cost_feasible(rows, assurance_target)
    best_json = {
        "design": {k: int(best[k]) for k in keys} if best else {},
        "assurance": float(best["assurance"]) if best else float("nan"),
        "feasible": bool(feasible and best is not None),
        "cost": {k: float(best[k]) for k in ("cost_total", "cost_human", "cost_model", "cost_prep")}
        if best
        else {},
    }
    (Path(out_dir) / "best_design.json").write_text(json.dumps(best_json, indent=2))

    return (best or {}), rows


# Expose helpers for tests
compute_pareto = _compute_pareto
select_min_cost_feasible = _select_min_cost_feasible
