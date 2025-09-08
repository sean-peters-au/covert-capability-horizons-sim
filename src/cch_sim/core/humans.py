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
    rng = np.random.default_rng(int(seed))
    participants = [f"P{i:02d}" for i in range(n_participants)]
    u_part = rng.normal(0.0, sigma_participant, size=n_participants)
    u_skill = rng.normal(0.0, human_skill_sd, size=n_participants)

    rows = []
    for _, trow in tasks.iterrows():
        # choose participants for each condition (without ensuring overlap for simplicity)
        choices = rng.choice(participants, size=min(2 * repeats_per_condition, n_participants), replace=False)
        T_part = choices[:repeats_per_condition]
        TC_part = choices[repeats_per_condition: repeats_per_condition * 2]

        for pid in T_part:
            idx = int(pid[1:])
            log_t = float(trow.log_t_baseline_true + u_part[idx] + rng.normal(0.0, sigma_noise))
            t_s = float(np.exp(log_t))
            rows.append(dict(participant_id=pid, task_id=trow.task_id, condition="T", t_seconds=t_s, log_t_obs=log_t,
                             censored=0, censor_reason="", attempt_success=np.nan))

        for pid in TC_part:
            idx = int(pid[1:])
            log_t = float(trow.log_t_baseline_true + trow.delta_true + u_part[idx] + rng.normal(0.0, sigma_noise))
            t_s = float(np.exp(log_t))
            p_covert = 1.0 / (1.0 + np.exp(-(human_cov_alpha + human_cov_beta * float(trow.delta_true) + u_skill[idx])))
            att_succ = int(rng.random() < p_covert)
            cens = int(not att_succ)
            rows.append(dict(participant_id=pid, task_id=trow.task_id, condition="T+C", t_seconds=t_s, log_t_obs=log_t,
                             censored=cens, censor_reason=("timeout" if cens==1 else ""), attempt_success=att_succ))

    return pd.DataFrame(rows)
