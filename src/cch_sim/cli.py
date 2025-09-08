"""Command‑line interface for running simulations, sweeps, and design searches."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional runtime dependency
    yaml = None

from .config import MonitorConfig, SimConfig
from .core.detection import apply_detection_models
from .core.humans import simulate_humans
from .core.models import generate_models, simulate_model_attempts
from .core.tasks import generate_tasks
from .design_search import run as run_design_search
from .gates import delta50_in_range_gate, delta50_precision_gate, trend_recovery_rope_gate
from .integrations import aggregate_assurance_table, write_assurance_table
from .pipeline import sample_humans_posterior, sample_models_posterior, sample_trend_posterior


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON file into a plain dict.

    Args:
        path: Path to a ``.yaml/.yml`` or ``.json`` file.

    Returns:
        Dict parsed from the file.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot read YAML config.")
        return dict(yaml.safe_load(txt))
    return dict(json.loads(txt))


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a single end‑to‑end simulation and write artifacts to disk."""
    # 1) Load and validate configuration
    cfg_raw = _load_yaml_or_json(args.config)
    cfg = _cfg_from_dict(cfg_raw)
    cfg.validate()
    os.makedirs(args.out, exist_ok=True)
    (Path(args.out) / "config.normalized.json").write_text(json.dumps(_cfg_to_dict(cfg), indent=2))
    # 2) Generate synthetic data: tasks → humans
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
    # 3) Stage‑1 posterior (humans)
    humans_draws = sample_humans_posterior(
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

    # 4) Build models and simulate attempts (with detection)
    models_list = generate_models(cfg.__dict__, np.random.default_rng(cfg.seed))
    attempts = simulate_model_attempts(
        rng=np.random.default_rng(cfg.seed),
        tasks=tasks,
        models=models_list,
        sc_alpha=cfg.sc_alpha,
        sc_beta=cfg.sc_beta,
        attempts_per_pair=cfg.attempts_per_pair,
    )
    mon0 = cfg.monitors[0]
    models_mon = apply_detection_models(
        attempts,
        dict(
            id=mon0.id,
            fpr_target=mon0.fpr_target,
            fpr_short=mon0.fpr_short,
            fpr_med=mon0.fpr_med,
            fpr_long=mon0.fpr_long,
            det_b1=mon0.det_b1,
            eta=mon0.eta,
        ),
    )
    models_mon["monitor_id"] = mon0.id

    # 5) Stage‑2 posterior (Δ50 per model/monitor)
    mod_draws = sample_models_posterior(
        models_mon,
        humans_draws,
        priors={
            "seed": cfg.seed,
            "num_warmup": 600,
            "num_samples": 1200,
            "num_chains": 2,
            "target_accept": 0.95,
            "max_tree_depth": 14,
        },
    )
    # 6) Trend posterior (per monitor)
    rel_map = {m["model_id"]: int(m.get("release_month", 0)) for m in models_list}
    trend = sample_trend_posterior(
        mod_draws,
        rel_map,
        priors={"seed": cfg.seed, "num_warmup_trend": 200, "num_samples_trend": 200},
    )

    # 7) Write outputs (Δ50 summaries, trend summaries, coverage checks)
    def ci_from_draws(arr):
        v = np.array(arr, float)
        if v.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        lo, med, hi = np.percentile(v, [2.5, 50.0, 97.5])
        return float(lo), float(med), float(hi)

    rows_delta = []
    for key, draws in mod_draws.get("delta50_s_draws", {}).items():
        mon_id, mid = key.split(":")
        lo, med, hi = ci_from_draws(draws)
        rows_delta.append(
            dict(
                monitor_id=mon_id,
                model_id=mid,
                release_month=rel_map.get(mid),
                delta50_s_lo=lo,
                delta50_s_med=med,
                delta50_s_hi=hi,
            )
        )
    pd.DataFrame(rows_delta).to_csv(Path(args.out) / "delta50_by_monitor.csv", index=False)

    # Trend summary (per monitor)
    rows_tr = []
    for mon, d in trend.get("trend", {}).items():
        slo_lo, slo_hi = d.get("slope_ci") or [float("nan"), float("nan")]
        dm_lo, dm_hi = d.get("dm_ci") or [float("nan"), float("nan")]
        rows_tr.append(
            dict(
                monitor_id=mon,
                slope_median=d.get("slope_median"),
                slope_lo=slo_lo,
                slope_hi=slo_hi,
                doubling_months_median=d.get("dm_median"),
                doubling_months_lo=dm_lo,
                doubling_months_hi=dm_hi,
            )
        )
    pd.DataFrame(rows_tr).to_csv(Path(args.out) / "delta50_trend_by_monitor.csv", index=False)
    # Coverage (near-Δ50) using Stage-1 Δ medians per task
    Delta_task_med = np.median(humans_draws["Delta_s"], axis=1)  # [n_tasks] seconds
    cov_rows = []
    for key, arr in mod_draws.get("delta50_s_draws", {}).items():
        mon_id, mid = key.split(":")
        d50_med = np.percentile(np.array(arr, float), 50.0)
        rel = abs(Delta_task_med - d50_med) <= 0.3 * max(d50_med, 1e-6)
        absw = abs(Delta_task_med - d50_med) <= 30.0
        cov_rows.append(
            dict(
                monitor_id=mon_id,
                model_id=mid,
                release_month=rel_map.get(mid),
                near_rel_count=int(np.sum(rel)),
                near_abs30s_count=int(np.sum(absw)),
                d50_s_median=float(d50_med),
            )
        )
    if cov_rows:
        pd.DataFrame(cov_rows).to_csv(Path(args.out) / "preflight_coverage.csv", index=False)
    # Human anchors SNR proxy (as before)
    snr_rows = []
    for tid, g in humans.groupby("task_id"):
        T = g[g["condition"] == "T"]["log_t_obs"].astype(float)
        TC = g[g["condition"] == "T+C"]["log_t_obs"].astype(float)
        nT = int(T.size)
        nTC = int(TC.size)
        vT = float(T.var(ddof=1)) if nT > 1 else float("nan")
        vTC = float(TC.var(ddof=1)) if nTC > 1 else float("nan")
        se = (
            float(np.sqrt(max(vT, 0.0) / max(nT, 1) + max(vTC, 0.0) / max(nTC, 1)))
            if np.isfinite(vT) and np.isfinite(vTC)
            else float("nan")
        )
        snr_rows.append(dict(task_id=tid, n_T=nT, n_TC=nTC, se_delta_proxy=se))
    if snr_rows:
        pd.DataFrame(snr_rows).to_csv(Path(args.out) / "anchors_snr.csv", index=False)
    # Minimal summary
    (Path(args.out) / "summary.json").write_text(
        json.dumps({"n_tasks": len(tasks), "n_models": len(models_list)}, indent=2)
    )
    print(f"Wrote outputs to {args.out}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a sweep over scenarios and compute gate‑based assurance per design."""
    data = _load_yaml_or_json(args.scenarios)
    scenarios = data.get("scenarios", [])
    gates = data.get("gates", {})
    trend_max_dm = float(gates.get("trend_max_doubling_months", float("nan")))
    # Optional Δ50 precision gate (seconds). Alias removed; use only 'delta50_max_ci_width_s'.
    delta50_max_ci_width_s = float(gates.get("delta50_max_ci_width_s", float("nan")))
    # Optional Δ50 in-range fraction
    d50_min_fraction = float(gates.get("d50_min_fraction_in_range", 0.8))
    # Δ50 trend recovery parameters (true doubling months and optional CI width bound)
    true_dm = float(
        gates.get("delta50_true_doubling_months", gates.get("true_doubling_months", float("nan")))
    )
    dm_rel_factor = float(gates.get("delta50_dm_rel_factor", 1.33))
    dm_min_prob = float(gates.get("delta50_dm_min_prob_in_window", 0.6))
    dm_rel_width_max = float(gates.get("delta50_dm_rel_width_max", float("nan")))
    replicates = int(getattr(args, "replicates", 1))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    trend_rows = []
    # Collect per-design per-seed gate pass records
    design_records: dict[str, list[dict[str, bool]]] = {}
    for sc_idx, sc in enumerate(scenarios, start=1):
        name = sc.get("name") or f"scenario_{sc_idx}"
        base_cfg = _cfg_from_dict(sc.get("cfg", {}))
        base_cfg.validate()
        for r in range(1, replicates + 1):
            cfg = _cfg_from_dict(sc.get("cfg", {}))
            cfg.seed = int(1000 * sc_idx + r)
            cfg.validate()
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
            draws_h = sample_humans_posterior(
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
            models_list = generate_models(cfg.__dict__, np.random.default_rng(cfg.seed))
            attempts = simulate_model_attempts(
                rng=np.random.default_rng(cfg.seed),
                tasks=tasks,
                models=models_list,
                sc_alpha=cfg.sc_alpha,
                sc_beta=cfg.sc_beta,
                attempts_per_pair=cfg.attempts_per_pair,
            )
            rel_map = {m["model_id"]: int(m.get("release_month", 0)) for m in models_list}
            for mon in cfg.monitors:
                mon_d = dict(
                    id=mon.id,
                    fpr_target=mon.fpr_target,
                    fpr_short=mon.fpr_short,
                    fpr_med=mon.fpr_med,
                    fpr_long=mon.fpr_long,
                    det_b1=mon.det_b1,
                    eta=mon.eta,
                )
                models_mon = apply_detection_models(attempts, mon_d)
                models_mon["monitor_id"] = mon.id
                draws_m = sample_models_posterior(
                    models_mon,
                    draws_h,
                    priors={
                        "seed": cfg.seed,
                        "num_warmup": 600,
                        "num_samples": 1200,
                        "num_chains": 2,
                        "target_accept": 0.95,
                        "max_tree_depth": 14,
                    },
                )
                # Initialize per-seed gate record for this design
                design_id = f"{name}|{mon.id}"
                gate_pass: dict[str, bool] = {}
                for key, arr in draws_m.get("delta50_s_draws", {}).items():
                    mon_id, mid = key.split(":")
                    lo, med, hi = np.percentile(np.array(arr, float), [2.5, 50.0, 97.5])
                    rows.append(
                        dict(
                            scenario=name,
                            seed=cfg.seed,
                            monitor_id=mon_id,
                            model_id=mid,
                            release_month=rel_map.get(mid),
                            delta50_s=med,
                            delta50_s_lo=lo,
                            delta50_s_med=med,
                            delta50_s_hi=hi,
                        )
                    )
                tr = sample_trend_posterior(
                    draws_m,
                    rel_map,
                    priors={"seed": cfg.seed, "num_warmup_trend": 200, "num_samples_trend": 200},
                )
                d = tr.get("trend", {}).get(mon.id, None)
                if d:
                    slo_lo, slo_hi = d.get("slope_ci") or [float("nan"), float("nan")]
                    dm_lo, dm_hi = d.get("dm_ci") or [float("nan"), float("nan")]
                    # Posterior gate probability if threshold is provided
                    p_gate = (
                        float(
                            np.mean(
                                np.array(d.get("doubling_months_draws", []), float) <= trend_max_dm
                            )
                        )
                        if math.isfinite(trend_max_dm)
                        and d.get("doubling_months_draws") is not None
                        else float("nan")
                    )
                    trend_rows.append(
                        dict(
                            scenario=name,
                            seed=cfg.seed,
                            monitor_id=mon.id,
                            slope=d.get("slope_median"),
                            slope_lo=slo_lo,
                            slope_hi=slo_hi,
                            doubling_months=d.get("dm_median"),
                            doubling_months_lo=dm_lo,
                            doubling_months_hi=dm_hi,
                            assurance_p_dm_le_gate=p_gate,
                        )
                    )

                    if np.isfinite(true_dm):
                        gtrend = trend_recovery_rope_gate(
                            d,
                            true_dm,
                            rel_factor=dm_rel_factor,
                            min_prob_in_window=dm_min_prob,
                            rel_width_max=(
                                dm_rel_width_max if np.isfinite(dm_rel_width_max) else None
                            ),
                        )
                        gate_pass["delta50_trend_rope"] = bool(gtrend.get("pass_", False))
                # --- Gate evaluation per seed for assurance (optional gates) ---
                # Δ50 precision gate: pass if any model meets width threshold
                if np.isfinite(delta50_max_ci_width_s):
                    pass_any = False
                    for key, arr in draws_m.get("delta50_s_draws", {}).items():
                        g = delta50_precision_gate(
                            np.asarray(arr, float), max_width_seconds=delta50_max_ci_width_s
                        )
                        if bool(g.get("pass_", False)):
                            pass_any = True
                            break
                    gate_pass["delta50_precision"] = pass_any
                # Δ50 in-range gate: pass if any model has Δ50 mostly inside observed Δ range
                if "d_seconds" in models_mon.columns:
                    series = models_mon["d_seconds"].astype(float)
                elif "c_overhead_s" in models_mon.columns:
                    series = models_mon["c_overhead_s"].astype(float)
                else:
                    series = None
                dmins = float(series.min()) if series is not None else float("nan")
                dmaxs = float(series.max()) if series is not None else float("nan")
                if np.isfinite(dmins) and np.isfinite(dmaxs):
                    pass_any = False
                    for key, arr in draws_m.get("delta50_s_draws", {}).items():
                        g = delta50_in_range_gate(
                            np.asarray(arr, float),
                            dmins,
                            dmaxs,
                            min_fraction_in_range=d50_min_fraction,
                        )
                        if bool(g.get("pass_", False)):
                            pass_any = True
                            break
                    gate_pass["delta50_in_range"] = pass_any
                # Record
                design_records.setdefault(design_id, []).append(gate_pass)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "sweep_results.csv", index=False)
    # Trend summary with Trend Threshold gate
    if trend_rows:
        dft = pd.DataFrame(trend_rows)
        agg_rows = []
        for (scenario, monitor_id), g in dft.groupby(["scenario", "monitor_id"]):
            pass_share = float("nan")
            if math.isfinite(trend_max_dm):
                vals = g.get("assurance_p_dm_le_gate")
                if vals is not None:
                    pass_share = float(np.nanmean(vals.astype(float)))
            agg_rows.append(
                dict(
                    scenario=scenario,
                    monitor_id=monitor_id,
                    seeds=len(g["seed"].unique()),
                    assurance_trend_threshold=pass_share,
                    median_doubling_months=float(g["doubling_months"].median()),
                    median_doubling_months_hi=float(g["doubling_months_hi"].median()),
                    median_slope=float(g["slope"].median()),
                )
            )
        pd.DataFrame(agg_rows).to_csv(Path(args.out) / "trend_summary_by_scenario.csv", index=False)
        (Path(args.out) / "gate_params.json").write_text(
            json.dumps(dict(trend_max_doubling_months=trend_max_dm), indent=2)
        )
    # Write assurance table if we gathered any gate records
    if design_records:
        rows_assure = aggregate_assurance_table(design_records)
        write_assurance_table(rows_assure, args.out, basename="assurance")
    print(f"Wrote sweep results -> {Path(args.out) / 'sweep_results.csv'}")


def cmd_design_search(args: argparse.Namespace) -> None:
    """Run automated design search for one scenario and write plots/CSV."""
    data = _load_yaml_or_json(args.scenario)
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise RuntimeError("Scenario file must contain a 'scenarios' list")
    sc0 = scenarios[0]
    cfg = _cfg_from_dict(sc0.get("cfg", {}))
    cfg.validate()
    # Gates can live at root
    gates = data.get("gates", {})
    # Study search config may be provided separately or embedded
    if getattr(args, "search", None):
        search_doc = _load_yaml_or_json(args.search)
        search_cfg = search_doc.get("study_search", search_doc)
    else:
        search_cfg = data.get("study_search", {})
    if not search_cfg:
        raise RuntimeError(
            "No study_search configuration found (provide --search or add study_search to scenario YAML)"
        )
    os.makedirs(args.out, exist_ok=True)
    best, rows = run_design_search(cfg, gates, search_cfg, args.out)
    print(json.dumps({"best": best, "out": args.out}, indent=2))


def _cfg_from_dict(d: Dict[str, Any]) -> SimConfig:
    """Construct a ``SimConfig`` from a raw dict (YAML/JSON)."""
    mons = d.get("monitors")
    if mons is None:
        monitors = [MonitorConfig()]
    else:
        monitors = []
        for m in mons:
            monitors.append(
                MonitorConfig(
                    id=str(m.get("id", "M0")),
                    label=m.get("label"),
                    fpr_target=float(m.get("fpr_target", 1e-4)),
                    fpr_short=m.get("fpr_short"),
                    fpr_med=m.get("fpr_med"),
                    fpr_long=m.get("fpr_long"),
                    det_b1=float(m.get("det_b1", 2.0)),
                    eta=float(m.get("eta", 0.3)),
                )
            )
    cfg = SimConfig(
        n_t_bins=int(d.get("n_t_bins", 6)),
        tasks_per_bin=int(d.get("tasks_per_bin", 2)),
        t_seconds_min=float(d.get("t_seconds_min", 30.0)),
        t_seconds_max=float(d.get("t_seconds_max", 3600.0)),
        n_participants=int(d.get("n_participants", 8)),
        repeats_per_condition=int(d.get("repeats_per_condition", 2)),
        sigma_task=float(d.get("sigma_task", 0.25)),
        sigma_participant=float(d.get("sigma_participant", 0.20)),
        sigma_noise=float(d.get("sigma_noise", 0.20)),
        human_skill_sd=float(d.get("human_skill_sd", 0.6)),
        human_cov_alpha=float(d.get("human_cov_alpha", -1.0)),
        human_cov_beta=float(d.get("human_cov_beta", 1.0)),
        human_l2_task=float(d.get("human_l2_task", 1.0)),
        human_l2_part=float(d.get("human_l2_part", 1.0)),
        human_l2_betaC=float(d.get("human_l2_betaC", 0.01)),
        human_l2_mu=float(d.get("human_l2_mu", 0.0)),
        human_l2_delta_task=float(d.get("human_l2_delta_task", 0.25)),
        human_opt_iters=int(d.get("human_opt_iters", 1200)),
        human_opt_lr=float(d.get("human_opt_lr", 0.05)),
        c_over_bins=d.get("c_over_bins"),
        c_over_mix_by_t_bin=d.get("c_over_mix_by_t_bin"),
        c_over_sample=str(d.get("c_over_sample", "log_uniform")),
        sc_alpha=float(d.get("sc_alpha", 0.0)),
        sc_beta=float(d.get("sc_beta", 1.5)),
        models_mode=str(d.get("models_mode", "custom")),
        models=d.get("models", []),
        n_models_auto=int(d.get("n_models_auto", 5)),
        trend_start_month=int(d.get("trend_start_month", 202101)),
        trend_end_month=int(d.get("trend_end_month", 202508)),
        trend_start_h50_s=float(d.get("trend_start_h50_s", 4.0)),
        trend_doubling_months=float(d.get("trend_doubling_months", 6.0)),
        trend_noise_sd_log=float(d.get("trend_noise_sd_log", 0.12)),
        trend_baseline_ref_sec=d.get("trend_baseline_ref_sec"),
        n_boot=int(d.get("n_boot", 200)),
        attempts_per_pair=int(d.get("attempts_per_pair", 15)),
        monitors=monitors,
        seed=int(d.get("seed", 42)),
    )
    return cfg


def _cfg_to_dict(cfg: SimConfig) -> Dict[str, Any]:
    """Render a ``SimConfig`` to a JSON‑serializable dict."""
    return {
        "n_t_bins": cfg.n_t_bins,
        "tasks_per_bin": cfg.tasks_per_bin,
        "t_seconds_min": cfg.t_seconds_min,
        "t_seconds_max": cfg.t_seconds_max,
        "n_participants": cfg.n_participants,
        "repeats_per_condition": cfg.repeats_per_condition,
        "sigma_task": cfg.sigma_task,
        "sigma_participant": cfg.sigma_participant,
        "sigma_noise": cfg.sigma_noise,
        "human_skill_sd": cfg.human_skill_sd,
        "human_cov_alpha": cfg.human_cov_alpha,
        "human_cov_beta": cfg.human_cov_beta,
        "human_l2_task": cfg.human_l2_task,
        "human_l2_part": cfg.human_l2_part,
        "human_l2_betaC": cfg.human_l2_betaC,
        "human_l2_mu": cfg.human_l2_mu,
        "human_l2_delta_task": cfg.human_l2_delta_task,
        "human_opt_iters": cfg.human_opt_iters,
        "human_opt_lr": cfg.human_opt_lr,
        "c_over_bins": cfg.c_over_bins,
        "c_over_mix_by_t_bin": cfg.c_over_mix_by_t_bin,
        "c_over_sample": cfg.c_over_sample,
        "sc_alpha": cfg.sc_alpha,
        "sc_beta": cfg.sc_beta,
        "models_mode": cfg.models_mode,
        "models": cfg.models,
        "n_models_auto": cfg.n_models_auto,
        "trend_start_month": cfg.trend_start_month,
        "trend_end_month": cfg.trend_end_month,
        "trend_start_h50_s": cfg.trend_start_h50_s,
        "trend_doubling_months": cfg.trend_doubling_months,
        "trend_noise_sd_log": cfg.trend_noise_sd_log,
        "trend_baseline_ref_sec": cfg.trend_baseline_ref_sec,
        "n_boot": cfg.n_boot,
        "attempts_per_pair": cfg.attempts_per_pair,
        "monitors": [
            {
                "id": m.id,
                "label": m.label,
                "fpr_target": m.fpr_target,
                "fpr_short": m.fpr_short,
                "fpr_med": m.fpr_med,
                "fpr_long": m.fpr_long,
                "det_b1": m.det_b1,
                "eta": m.eta,
            }
            for m in cfg.monitors
        ],
        "seed": cfg.seed,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the top‑level argument parser and subcommands."""
    p = argparse.ArgumentParser(prog="cch-sim", description="Covert Capability Horizons simulation")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simulate", help="Run a single simulation")
    ps.add_argument("--config", required=True, help="Path to YAML/JSON config")
    ps.add_argument("--out", required=True, help="Output directory")
    ps.set_defaults(func=cmd_simulate)

    pw = sub.add_parser("sweep", help="Run a sweep over scenarios")
    pw.add_argument("--scenarios", required=True, help="YAML/JSON with scenarios list")
    pw.add_argument("--out", required=True, help="Output directory")
    pw.add_argument("--replicates", type=int, default=5, help="Replicates per scenario (seeds)")
    pw.set_defaults(func=cmd_sweep)

    pdz = sub.add_parser("design-search", help="Automated design search for one scenario")
    pdz.add_argument("--scenario", required=True, help="Path to YAML/JSON scenario file")
    pdz.add_argument("--out", required=True, help="Output directory")
    pdz.add_argument("--search", required=False, help="Optional YAML/JSON with study_search block")
    pdz.set_defaults(func=cmd_design_search)

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``cch-sim`` command."""
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
