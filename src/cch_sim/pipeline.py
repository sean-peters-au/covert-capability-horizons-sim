"""
Bayesian simulation pipeline (single, simple path):

log inside (humans, log-seconds) → absolute outside (Δ seconds, models) → trend on log(Δ50).

This module defines three pure functions. They import NumPyro/JAX/ArviZ lazily
so the rest of the codebase can import this file without requiring those
dependencies until you actually call the sampling functions.

Functions
- sample_humans_posterior(humans_df, priors) -> dict with draws for tT_s[j,k], Δ_s[j,k]
- sample_models_posterior(attempts_df, draws_humans, priors) -> dict with draws for Δ50_s
- sample_trend_posterior(metric_draws, release_months, priors) -> dict with slope/doubling-months draws
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _require_numpyro():
    try:
        import arviz as az  # noqa: F401
        import jax  # noqa: F401
        import numpyro  # noqa: F401
        import numpyro.distributions as dist  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "NumPyro/JAX/ArviZ are required for the Bayesian pipeline."
            " Please install numpyro[jax,cpu] and arviz in your environment."
        ) from e


def sample_humans_posterior(
    humans_df: pd.DataFrame, priors: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Sample the Bayesian censored hierarchical log‑time model for humans.

    Returns a dict with:
      - tasks: list of task_ids in model order
      - draws: K posterior draws
      - tT_s: array [n_tasks, K] baseline seconds per task per draw
      - Delta_s: array [n_tasks, K] covert seconds per task per draw

    Raises RuntimeError if NumPyro/JAX/ArviZ not installed.
    """
    _require_numpyro()
    # Lazy imports inside function
    import arviz as az
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    df = humans_df.copy()
    # Required columns: participant_id, task_id, condition ('T' or 'T+C'), log_t_obs, censored (0/1)
    if not set(["participant_id", "task_id", "condition", "log_t_obs"]).issubset(df.columns):
        raise ValueError("humans_df missing required columns")

    tasks = sorted(df["task_id"].unique().tolist())
    # parts = sorted(df["participant_id"].unique().tolist())
    tmap = {t: i for i, t in enumerate(tasks)}
    tj = df["task_id"].map(tmap).to_numpy()
    zc = (df["condition"].astype(str) == "T+C").astype(int).to_numpy()
    y = df["log_t_obs"].astype(float).to_numpy()
    cens = df.get("censored", 0)
    cens = cens.astype(int).to_numpy()

    n_tasks = len(tasks)

    p = priors or {}

    def model(y, tj, zc, cens):
        # Priors
        mu = numpyro.sample("mu", dist.Normal(0.0, p.get("mu_sd", 2.0)))
        sigma = numpyro.sample("sigma", dist.HalfNormal(p.get("sigma_sd", 0.5)))

        tau_task = numpyro.sample("tau_task", dist.HalfNormal(p.get("tau_task_sd", 0.3)))
        tau_d = numpyro.sample("tau_d", dist.HalfNormal(p.get("tau_d_sd", 0.2)))

        a_task_raw = numpyro.sample("a_task_raw", dist.Normal(0, 1).expand([n_tasks]).to_event(1))
        d_task_raw = numpyro.sample("d_task_raw", dist.Normal(0, 1).expand([n_tasks]).to_event(1))

        # Sum-to-zero constraints to reduce location non-identifiability
        a_task = (a_task_raw - jnp.mean(a_task_raw)) * tau_task
        d_task = (d_task_raw - jnp.mean(d_task_raw)) * tau_d

        betaC = numpyro.sample(
            "betaC", dist.Normal(p.get("betaC_loc", 0.3), p.get("betaC_sd", 0.25))
        )

        mean = mu + a_task[tj] + (betaC + d_task[tj]) * zc

        # Base likelihood via obs; then adjust censored rows to use survival contribution
        z = (y - mean) / sigma
        ll_unc = dist.Normal(mean, sigma).log_prob(y)
        ll_cen = jsp.stats.norm.logsf(z)
        m_cen = jnp.asarray(cens, dtype=jnp.float32)
        numpyro.sample("y_obs", dist.Normal(mean, sigma), obs=y)
        # For censored rows, replace ll_unc with ll_cen by adding (ll_cen - ll_unc)
        numpyro.factor("y_cens_adjust", jnp.sum(m_cen * (ll_cen - ll_unc)))

        # Return deterministics for derived quantities
        numpyro.deterministic("baseline_log_t_T_hat", mu + a_task)
        numpyro.deterministic("delta_log", betaC + d_task)

    nuts = NUTS(
        model,
        target_accept_prob=p.get("target_accept", 0.9),
        max_tree_depth=p.get("max_tree_depth", 12),
    )
    mcmc = MCMC(
        nuts,
        num_warmup=int(p.get("num_warmup", 800)),
        num_samples=int(p.get("num_samples", 800)),
        num_chains=int(p.get("num_chains", 4)),
    )
    mcmc.run(jax.random.PRNGKey(int(p.get("seed", 0))), y=y, tj=tj, zc=zc, cens=cens)
    samples = mcmc.get_samples(group_by_chain=False)
    # Diagnostics
    idata = az.from_numpyro(mcmc)
    # Convergence checks on the estimands we consume downstream
    chk_vars = ["baseline_log_t_T_hat", "delta_log"]
    rhat = az.rhat(idata, var_names=chk_vars)
    ess = az.ess(idata, var_names=chk_vars, method="bulk")
    r_arr = rhat.to_array()
    e_arr = ess.to_array()
    if np.isfinite(r_arr).any():
        max_rhat = float(np.nanmax(r_arr))
    else:
        max_rhat = float("nan")
    if np.isfinite(e_arr).any():
        min_ess = float(np.nanmin(e_arr))
    else:
        min_ess = float("nan")
    bad_rhat = np.isfinite(max_rhat) and (max_rhat > float(p.get("rhat_max", 1.05)))
    bad_ess = np.isfinite(min_ess) and (min_ess < float(p.get("ess_min", 200)))
    if bad_rhat or bad_ess:
        raise RuntimeError(
            f"Stage1 convergence failure: rhat_max={max_rhat:.3f}, ess_min={min_ess:.1f}"
        )

    # Derive seconds per task per draw
    base_log = np.asarray(samples["baseline_log_t_T_hat"])  # shape [n_tasks]
    # Above are per-task summaries; we used deterministic per task but sampled across draws => need to access latent per task per draw
    # Since deterministic captured baseline per task per draw, NumPyro records with shape [num_samples, n_tasks]
    base_log = samples["baseline_log_t_T_hat"]  # [S, n_tasks]
    delta_log = samples["delta_log"]  # [S, n_tasks]
    base_t = np.exp(np.array(base_log))  # [S, n_tasks] seconds
    Delta = base_t * (np.exp(np.array(delta_log)) - 1.0)  # [S, n_tasks] seconds

    # Transpose to [n_tasks, S]
    tT_s = base_t.T
    Delta_s = Delta.T
    return dict(tasks=tasks, draws=tT_s.shape[1], tT_s=tT_s, Delta_s=Delta_s)


def sample_models_posterior(
    attempts_df: pd.DataFrame, humans_draws: Dict[str, Any], priors: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Sample Bayesian logistic on absolute Δ seconds for each model/monitor.

    Returns a dict with per model/monitor:
      - draws: K posterior draws
      - delta50_s_draws: [K] seconds per model/monitor (Δ50)

    Raises RuntimeError if NumPyro/ArviZ not installed.
    """
    _require_numpyro()
    import arviz as az
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    _draws = humans_draws.get("draws")
    _tT_s = np.array(humans_draws["tT_s"])  # [n_tasks, S]
    # Δ seconds per task per draw (but attempts carry Δ per attempt; we align by task)
    # We'll use attempt-level Δ (d_seconds) for x; humans draws used to compute H50 from Δ50

    # Group attempts per (monitor_id?, model_id?)
    df = attempts_df.copy()
    if "monitor_id" not in df.columns:
        df["monitor_id"] = "M0"

    results: Dict[str, Any] = {"delta50_s_draws": {}}

    for (mon, mid), g in df.groupby(["monitor_id", "model_id"]):
        # Prepare x = Δ seconds per attempt; y = success
        if "d_seconds" in g.columns:
            x_raw = g["d_seconds"].astype(float).to_numpy()
        elif "c_overhead_s" in g.columns:
            x_raw = g["c_overhead_s"].astype(float).to_numpy()
        else:
            raise ValueError("attempts_df requires d_seconds or c_overhead_s for Δ (seconds)")
        y = g["success"].astype(float).to_numpy()
        # Standardize Δ seconds for stable geometry
        mu_x = float(np.mean(x_raw))
        sigma_x = float(np.std(x_raw))
        if not np.isfinite(sigma_x) or sigma_x < 1e-12:
            sigma_x = 1.0
        x = (x_raw - mu_x) / sigma_x

        # Bayesian logistic with θ1 < 0 via softplus
        def logit_model(x, y):
            theta0 = numpyro.sample(
                "theta0", dist.Normal(0.0, priors.get("theta0_sd", 5.0) if priors else 5.0)
            )
            beta1 = numpyro.sample(
                "beta1", dist.Normal(0.0, priors.get("beta1_sd", 2.0) if priors else 2.0)
            )
            theta1 = -jnp.log1p(jnp.exp(-beta1))  # -softplus(-beta1)
            logits = theta0 + theta1 * x
            numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

        nuts = NUTS(logit_model, target_accept_prob=(priors or {}).get("target_accept", 0.9))
        mcmc = MCMC(
            nuts,
            num_warmup=int((priors or {}).get("num_warmup", 600)),
            num_samples=int((priors or {}).get("num_samples", 600)),
            num_chains=int((priors or {}).get("num_chains", 4)),
        )
        mcmc.run(jax.random.PRNGKey(int((priors or {}).get("seed", 0))), x=x, y=y)
        s = mcmc.get_samples(group_by_chain=False)
        # Diagnostics
        idata = az.from_numpyro(mcmc)
        rhat = az.rhat(idata, var_names=["theta0", "beta1"])
        ess = az.ess(idata, var_names=["theta0", "beta1"], method="bulk")
        r_arr = rhat.to_array()
        e_arr = ess.to_array()
        if np.isfinite(r_arr).any():
            max_rhat = float(np.nanmax(r_arr))
        else:
            max_rhat = float("nan")
        if np.isfinite(e_arr).any():
            min_ess = float(np.nanmin(e_arr))
        else:
            min_ess = float("nan")
        bad_rhat = np.isfinite(max_rhat) and (
            max_rhat > float((priors or {}).get("rhat_max", 1.05))
        )
        bad_ess = np.isfinite(min_ess) and (min_ess < float((priors or {}).get("ess_min", 200)))
        if bad_rhat or bad_ess:
            raise RuntimeError(
                f"Stage2 convergence failure for {mon}:{mid} rhat_max={max_rhat:.3f}, ess_min={min_ess:.1f}"
            )
        theta0 = np.array(s["theta0"])  # [S]
        beta1 = np.array(s["beta1"])  # [S]
        theta1 = -np.log1p(np.exp(-beta1))
        # Δ50 on standardized scale mapped back to seconds
        d50_z = -theta0 / theta1
        d50_s = mu_x + sigma_x * d50_z
        # Physical constraint: Δ50 cannot be negative. Also avoid exploding far beyond observed range.
        try:
            xmin_f = float(np.min(x_raw))
            xmax_f = float(np.max(x_raw))
        except Exception:
            xmin_f = 0.0
            xmax_f = float("inf")
        if not np.isfinite(xmin_f):
            xmin_f = 0.0
        if not np.isfinite(xmax_f):
            xmax_f = float("inf")
        lo = max(0.0, xmin_f)
        hi = xmax_f
        d50_s = np.clip(d50_s, lo, hi)

        key = f"{mon}:{mid}"
        results["delta50_s_draws"][key] = d50_s

    return results


def sample_trend_posterior(
    metric_draws: dict[str, Any],
    release_months: dict[str, int],
    priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-monitor trend of log(series seconds) vs release month using per-draw OLS.

    Returns a dict with posterior draws for slope and doubling months, plus summaries.

    Series units are seconds (Δ50 by default); slopes are per year on log-seconds; doubling months is 12*ln(2)/slope for positive slopes.
    """
    _require_numpyro()

    # Build arrays for regression per draw
    # metric_draws: {"delta50_s_draws" or "h50_s_draws": {key: [S]}} where key = "mon:model"
    # release_months: mapping model_id -> yyyymm; need per key; assume key encodes monitor:model; we'll map model part
    series_key = "delta50_s_draws" if "delta50_s_draws" in metric_draws else "h50_s_draws"
    keys = sorted(metric_draws[series_key].keys())
    # Stack H50 across models for a given monitor; we’ll do separate regression per monitor
    out: dict[str, dict[str, Any]] = {"trend": {}}
    for mon in set(k.split(":")[0] for k in keys):
        mon_keys = [k for k in keys if k.startswith(mon + ":")]

        # Determine x (year) per model
        def frac_year(yyyymm: int) -> float:
            y = int(yyyymm) // 100
            m = int(yyyymm) % 100
            return y + (m - 1) / 12.0

        x_year = np.array(
            [frac_year(release_months[k.split(":")[1]]) for k in mon_keys], float
        )  # [M]
        # Series draws matrix [M, S]
        H = np.vstack([metric_draws[series_key][k] for k in mon_keys])  # [M, S]
        S = H.shape[1]

        # Compute slope per draw via OLS
        x = x_year
        x_bar = float(np.mean(x))
        Sxx = float(np.sum((x - x_bar) ** 2)) if len(x) > 1 else 0.0
        slopes_list: list[float] = []
        if Sxx > 0:
            for s in range(S):
                ylog = np.log(np.clip(H[:, s], 1e-12, None))
                y_bar = float(np.mean(ylog))
                Sxy = float(np.sum((x - x_bar) * (ylog - y_bar)))
                slopes_list.append(Sxy / Sxx)
        slopes = np.array(slopes_list)
        # Positive slope ⇒ finite doubling months; non-positive ⇒ inf
        dm = np.where(slopes > 0, 12.0 * np.log(2.0) / (slopes + 1e-12), np.inf)
        # Robust summaries for small M (e.g., one model ⇒ no slope draws)
        if slopes.size > 0:
            slope_median = float(np.nanmedian(slopes))
            slope_ci = list(np.nanpercentile(slopes, [2.5, 97.5]))
        else:
            slope_median = float("nan")
            slope_ci = [float("nan"), float("nan")]
        if np.isfinite(dm).any():
            dm_median = float(np.nanmedian(dm[np.isfinite(dm)]))
            dm_ci = list(np.nanpercentile(dm[np.isfinite(dm)], [2.5, 97.5]))
        else:
            dm_median = float("inf")
            dm_ci = [float("nan"), float("nan")]
        out["trend"][mon] = dict(
            slope_draws=slopes,
            doubling_months_draws=dm,
            slope_median=slope_median,
            slope_ci=slope_ci,
            dm_median=dm_median,
            dm_ci=dm_ci,
        )
    return out
