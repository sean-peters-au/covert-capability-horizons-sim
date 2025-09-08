"""Human data generation for T vs T+C conditions.

Simulates human observations per task and condition (T and T+C). Each task observation
in T+C includes an attempt‑success bit used to mark right‑censoring (timeout) in the
Stage‑1 model. Participant‑level random effects capture baseline variation and skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_humans(
    *,
    seed: int,
    tasks: pd.DataFrame,
    n_participants: int,
    repeats_per_condition: int,
    sigma_participant: float,
    sigma_noise: float,
    human_skill_sd: float,
    human_cov_alpha: float,
    human_cov_beta: float,
) -> pd.DataFrame:
    """Simulate human observations per task for T and T+C conditions.

    Args:
        seed: RNG seed.
        tasks: DataFrame from ``generate_tasks``.
        n_participants: Pool size; participants are sampled without replacement per task.
        repeats_per_condition: Observations per condition per task.
        sigma_participant: Stddev for participant random intercepts on log‑time.
        sigma_noise: Residual log‑time noise per observation.
        human_skill_sd: Stddev for participant skill in covert success probability.
        human_cov_alpha: Intercept for covert success (logit scale).
        human_cov_beta: Slope on delta_true for covert success (logit scale).

    Returns:
        DataFrame with columns including ``participant_id``, ``task_id``, ``condition``,
        observed log time, censoring flag, and attempt success indicator.
    """
    rng = np.random.default_rng(int(seed))
    participants = [f"P{i:02d}" for i in range(n_participants)]
    u_part = rng.normal(0.0, sigma_participant, size=n_participants)
    u_skill = rng.normal(0.0, human_skill_sd, size=n_participants)

    rows = []
    for _, trow in tasks.iterrows():
        # Choose participants for each condition (without enforcing overlap for simplicity)
        choices = rng.choice(
            participants, size=min(2 * repeats_per_condition, n_participants), replace=False
        )
        T_part = choices[:repeats_per_condition]
        TC_part = choices[repeats_per_condition : repeats_per_condition * 2]

        for pid in T_part:
            idx = int(pid[1:])
            log_t = float(trow.log_t_baseline_true + u_part[idx] + rng.normal(0.0, sigma_noise))
            t_s = float(np.exp(log_t))
            rows.append(
                dict(
                    participant_id=pid,
                    task_id=trow.task_id,
                    condition="T",
                    t_seconds=t_s,
                    log_t_obs=log_t,
                    censored=0,
                    censor_reason="",
                    attempt_success=np.nan,
                )
            )

        for pid in TC_part:
            idx = int(pid[1:])
            log_t = float(
                trow.log_t_baseline_true
                + trow.delta_true
                + u_part[idx]
                + rng.normal(0.0, sigma_noise)
            )
            t_s = float(np.exp(log_t))
            p_covert = 1.0 / (
                1.0
                + np.exp(
                    -(human_cov_alpha + human_cov_beta * float(trow.delta_true) + u_skill[idx])
                )
            )
            att_succ = int(rng.random() < p_covert)
            cens = int(not att_succ)
            rows.append(
                dict(
                    participant_id=pid,
                    task_id=trow.task_id,
                    condition="T+C",
                    t_seconds=t_s,
                    log_t_obs=log_t,
                    censored=cens,
                    censor_reason=("timeout" if cens == 1 else ""),
                    attempt_success=att_succ,
                )
            )

    return pd.DataFrame(rows)
