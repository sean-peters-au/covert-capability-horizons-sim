from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import json
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def _ym_to_date(yyyymm: int) -> pd.Timestamp:
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


def _ensure_out(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def _set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "font.size": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _fmt_thousands(x, pos):
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)


def _non_dominated(rows: pd.DataFrame) -> pd.DataFrame:
    # Return non-dominated set for (min cost_total, max assurance)
    nd = []
    vals = rows[["cost_total", "assurance"]].to_numpy(dtype=float)
    for i, (c, a) in enumerate(vals):
        dominated = False
        for j, (cj, aj) in enumerate(vals):
            if j == i:
                continue
            if (cj <= c and aj >= a) and (cj < c or aj > a):
                dominated = True
                break
        if not dominated:
            nd.append(True)
        else:
            nd.append(False)
    return rows.loc[nd]


def plot_pareto(run_dir: Path, out_dir: Path) -> Path:
    dfp = pd.read_csv(run_dir / "pareto.csv")
    dfa = pd.read_csv(run_dir / "design_search.csv")
    # Join a flag for pareto membership
    key_cols = ["attempts_per_pair", "tasks_per_bin", "n_participants", "repeats_per_condition"]
    dfp["_pareto_key"] = list(zip(*[dfp[k] for k in key_cols]))
    dfa["_pareto_key"] = list(zip(*[dfa[k] for k in key_cols]))
    dfa["on_pareto"] = dfa["_pareto_key"].isin(set(dfp["_pareto_key"]))

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    # Visual encodings: color=attempts, size=tasks, marker=repeats
    sizes = 36 + 8 * (dfa["tasks_per_bin"].astype(float) - dfa["tasks_per_bin"].min())
    markers = {1: "o", 2: "s", 3: "D"}
    colors = dfa["attempts_per_pair"].astype(float)
    for rep, dfrep in dfa.groupby("repeats_per_condition"):
        ax.scatter(dfrep["cost_total"], dfrep["assurance"], c=colors.loc[dfrep.index], cmap="viridis",
                   s=sizes.loc[dfrep.index], alpha=0.85, label=f"repeats={int(rep)}", marker=markers.get(int(rep), "o"), edgecolors="k", linewidths=0.2)
    # Highlight pareto
    pare = dfa[dfa["on_pareto"]]
    ax.scatter(pare["cost_total"], pare["assurance"], facecolors="none", edgecolors="crimson", s=64, linewidths=1.5, label="Pareto")
    ax.set_xlabel("Cost (total)")
    ax.set_ylabel("Assurance")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(dfa["attempts_per_pair"])  # type: ignore[arg-type]
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("Attempts per pair")
    ax.set_title("Pareto Frontier: Cost vs Assurance")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower right")
    _ensure_out(out_dir)
    p = out_dir / "fig_pareto.png"
    fig.savefig(p)
    fig.savefig(out_dir / "fig_pareto.svg")
    plt.close(fig)
    return p


def _load_best_design(run_dir: Path) -> Tuple[Dict, Dict]:
    with open(run_dir / "best_design.json", "r") as f:
        best = json.load(f)
    dfa = pd.read_csv(run_dir / "design_search.csv")
    design = best.get("design", {})
    mask = (
        (dfa["attempts_per_pair"] == design.get("attempts_per_pair"))
        & (dfa["tasks_per_bin"] == design.get("tasks_per_bin"))
        & (dfa["n_participants"] == design.get("n_participants"))
        & (dfa["repeats_per_condition"] == design.get("repeats_per_condition"))
    )
    row = dfa[mask].iloc[0].to_dict() if mask.any() else {}
    return best, row


def plot_best_gates(run_dir: Path, out_dir: Path) -> Path:
    best, row = _load_best_design(run_dir)
    labels = []
    vals = []
    for k in ["gate_delta50_precision_rate", "gate_delta50_in_range_rate", "gate_delta50_trend_rope_rate"]:
        if k in row:
            labels.append(k.replace("gate_", "").replace("_rate", "").replace("_", " "))
            vals.append(float(row[k]))
    if not labels:
        return out_dir / "fig_best_gates.png"
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#4c78a8")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pass rate across seeds")
    ax.set_title("Best Design: Gate Pass Rates")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center")
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    _ensure_out(out_dir)
    p = out_dir / "fig_best_gates.png"
    fig.savefig(p)
    fig.savefig(out_dir / "fig_best_gates.svg")
    plt.close(fig)
    return p


def plot_trend(run_dir: Path, out_dir: Path) -> Path | None:
    p_d = run_dir / "delta50_by_monitor.csv"
    p_t = run_dir / "delta50_trend_by_monitor.csv"
    if not p_d.exists() or not p_t.exists():
        return None
    df_d = pd.read_csv(p_d)
    df_t = pd.read_csv(p_t)
    # Assume one monitor for simplicity
    mon = df_d["monitor_id"].iloc[0]
    dmon = df_d[df_d["monitor_id"] == mon].copy()
    dmon.sort_values("release_month", inplace=True)
    # X axis dates
    dmon["date"] = dmon["release_month"].apply(_ym_to_date)
    # Use medians and CI for errorbars
    y = dmon["delta50_s_med"].astype(float).to_numpy()
    ylo = dmon["delta50_s_lo"].astype(float).to_numpy()
    yhi = dmon["delta50_s_hi"].astype(float).to_numpy()
    yerr = np.vstack([y - ylo, yhi - y])

    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    ax.errorbar(dmon["date"], y, yerr=yerr, fmt="o", ms=4, lw=1.0, color="#4c78a8", ecolor="#9ecae1", alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Δ50 (seconds, log scale)")
    ax.set_xlabel("Release month")
    ax.set_title("Δ50 Trend by Release")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    # Overlay median trend line using slope and an intercept calibrated to first point
    dt = df_t[df_t["monitor_id"] == mon].iloc[0].to_dict()
    slope = float(dt.get("slope_median", np.nan))
    if np.isfinite(slope) and len(dmon) >= 2:
        x = pd.to_datetime(dmon["date"]).view("int64") / 1e9  # seconds since epoch
        x = (x - x.min()) / (3600 * 24 * 30.4375)  # months from start (approx)
        # Fit intercept in log-space using first observation
        y0 = float(y[0])
        x0 = float(x.iloc[0])
        # slope is per year on log-seconds in pipeline; convert to per month
        slope_pm = slope / 12.0
        # log(y) = a + slope_pm * months; a = log(y0) - slope_pm * x0
        a = math.log(max(y0, 1e-12)) - slope_pm * x0
        xx = np.linspace(float(x.min()), float(x.max()), 100)
        yy = np.exp(a + slope_pm * xx)
        # Map back to dates for plotting
        t0 = pd.to_datetime(dmon["date"].min())
        xdates = [t0 + pd.Timedelta(days=float(m) * 30.4375) for m in xx]
        ax.plot(xdates, yy, color="#e45756", lw=1.5, label="Trend (median)")
        ax.legend()

    _ensure_out(out_dir)
    p = out_dir / "fig_trend.png"
    fig.savefig(p)
    fig.savefig(out_dir / "fig_trend.svg")
    plt.close(fig)
    return p


def plot_heatmaps(run_dir: Path, out_dir: Path) -> Tuple[int, int]:
    dfa = pd.read_csv(run_dir / "design_search.csv")
    parts = sorted(dfa["n_participants"].unique().tolist())
    reps = sorted(dfa["repeats_per_condition"].unique().tolist())
    n_saved = 0
    for npart in parts:
        for rep in reps:
            sub = dfa[(dfa["n_participants"] == npart) & (dfa["repeats_per_condition"] == rep)]
            if sub.empty:
                continue
            pivot = sub.pivot(index="attempts_per_pair", columns="tasks_per_bin", values="assurance")
            fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns.tolist())
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index.tolist())
            ax.set_xlabel("Tasks per bin")
            ax.set_ylabel("Attempts per pair")
            ax.set_title(f"Assurance heatmap (n_participants={npart}, repeats={rep})")
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Assurance")
            _ensure_out(out_dir)
            p = out_dir / f"fig_heatmap_p{int(npart)}_r{int(rep)}.png"
            fig.savefig(p)
            fig.savefig(out_dir / f"fig_heatmap_p{int(npart)}_r{int(rep)}.svg")
            plt.close(fig)
            n_saved += 1
    return n_saved, len(parts) * len(reps)


def _agg_main_effect(dfa: pd.DataFrame, param: str) -> pd.DataFrame:
    g = dfa.groupby(param)
    out = g.agg(
        assurance_mean=("assurance", "mean"),
        assurance_q25=("assurance", lambda s: float(np.quantile(s, 0.25))),
        assurance_q75=("assurance", lambda s: float(np.quantile(s, 0.75))),
        cost_median=("cost_total", "median"),
    ).reset_index().rename(columns={param: "level"})
    return out.sort_values("level")


def plot_pareto_simple(run_dir: Path, out_dir: Path) -> Path:
    dfa = pd.read_csv(run_dir / "design_search.csv")
    best, _ = _load_best_design(run_dir)
    design = best.get("design", {})
    # Figure
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    # All candidates in light gray
    ax.scatter(dfa["cost_total"], dfa["assurance"], color="#bbbbbb", s=26, alpha=0.7, label="Candidates")
    # Pareto line
    nd = _non_dominated(dfa)
    nd_sorted = nd.sort_values("cost_total")
    ax.plot(nd_sorted["cost_total"], nd_sorted["assurance"], color="#4c78a8", lw=2.0, label="Pareto frontier")
    # Selected design
    if design:
        mask = (
            (dfa["attempts_per_pair"] == design.get("attempts_per_pair")) &
            (dfa["tasks_per_bin"] == design.get("tasks_per_bin")) &
            (dfa["n_participants"] == design.get("n_participants")) &
            (dfa["repeats_per_condition"] == design.get("repeats_per_condition"))
        )
        dd = dfa[mask].iloc[0]
        ax.scatter([dd["cost_total"]], [dd["assurance"]], s=120, marker="*", color="#e45756", edgecolors="k", zorder=5, label="Selected")
    ax.set_xlabel("Cost (total)")
    ax.set_ylabel("Assurance")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.set_title("Cost vs Assurance with Pareto Frontier")
    ax.legend(loc="lower right")
    p = out_dir / "fig_pareto_simple.png"
    fig.savefig(p)
    fig.savefig(out_dir / "fig_pareto_simple.svg")
    plt.close(fig)
    return p


def plot_main_effects(run_dir: Path, out_dir: Path) -> Path:
    dfa = pd.read_csv(run_dir / "design_search.csv")
    params = ["attempts_per_pair", "tasks_per_bin", "n_participants", "repeats_per_condition"]
    aggs = {p: _agg_main_effect(dfa, p) for p in params}
    # Global cost range for right axes
    cost_min = min(float(df.cost_median.min()) for df in aggs.values())
    cost_max = max(float(df.cost_median.max()) for df in aggs.values())

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for ax, p in zip(axes.flatten(), params):
        dfp = aggs[p]
        x = dfp["level"].astype(float).to_numpy()
        a_mu = dfp["assurance_mean"].to_numpy()
        a_lo = dfp["assurance_q25"].to_numpy()
        a_hi = dfp["assurance_q75"].to_numpy()
        ax.plot(x, a_mu, color="#4c78a8", marker="o")
        ax.fill_between(x, a_lo, a_hi, color="#4c78a8", alpha=0.2, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_xlabel(p.replace("_", " "))
        ax.set_ylabel("Assurance")
        # Right axis for cost
        ax2 = ax.twinx()
        ax2.plot(x, dfp["cost_median"], color="#7f7f7f", linestyle="--", marker="s", alpha=0.8)
        ax2.set_ylim(cost_min * 0.95, cost_max * 1.05)
        ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
        ax2.set_ylabel("Cost (total)")
        ax.set_title(p.replace("_", " ").title())

    # Figure-level legend (assurance mean, interquartile band, median cost)
    handles = [
        Line2D([0], [0], color="#4c78a8", marker="o", label="Assurance (mean)"),
        mpatches.Patch(color="#4c78a8", alpha=0.2, label="Assurance IQR (25–75%)"),
        Line2D([0], [0], color="#7f7f7f", linestyle="--", marker="s", label="Cost (median, right axis)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    p = out_dir / "fig_main_effects.png"
    fig.savefig(p)
    fig.savefig(out_dir / "fig_main_effects.svg")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from design-search and trend outputs")
    ap.add_argument("--run-dir", required=True, help="Path to run output directory (contains design_search.csv, pareto.csv, etc.)")
    ap.add_argument("--out", required=False, help="Output directory for plots (default: <run-dir>/plots)")
    args = ap.parse_args()
    _set_style()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run from {run_dir}")
    p1 = plot_pareto_simple(run_dir, out_dir)
    p2 = plot_main_effects(run_dir, out_dir)
    print("Saved:")
    print(f" - {p1}")
    print(f" - {p2}")


if __name__ == "__main__":  # pragma: no cover
    main()
