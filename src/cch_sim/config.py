from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MonitorConfig:
    id: str = "M0"
    label: Optional[str] = None
    fpr_target: float = 1e-4  # ultra-low false-positive rate target
    fpr_short: Optional[float] = None
    fpr_med: Optional[float] = None
    fpr_long: Optional[float] = None
    det_b1: float = 2.0       # sensitivity to covert overhead
    eta: float = 0.3          # time-dilation exponent on detection difficulty


@dataclass
class SimConfig:
    # Task/time bins (T)
    n_t_bins: int = 6
    tasks_per_bin: int = 2
    t_seconds_min: float = 30.0
    t_seconds_max: float = 3600.0

    # Human sampling
    n_participants: int = 8
    repeats_per_condition: int = 2

    # Noise & skill
    sigma_task: float = 0.25
    sigma_participant: float = 0.20
    sigma_noise: float = 0.20
    human_skill_sd: float = 0.6
    human_cov_alpha: float = -1.0
    human_cov_beta: float = 1.0

    # Covert overhead (seconds) ranges per T-bin (consistent construction)
    # c_over_bins: list of {lo_s, hi_s}; c_over_mix_by_t_bin: list of length n_t_bins, each a list of len(c_over_bins)
    c_over_bins: Optional[List[Dict[str, float]]] = None
    c_over_mix_by_t_bin: Optional[List[List[float]]] = None
    c_over_sample: str = "log_uniform"  # or "uniform"

    # Success curve versus absolute covert overhead (seconds)
    sc_alpha: float = 0.0
    sc_beta: float = 1.5

    # Models
    models_mode: str = "custom"  # "custom" or "trend"
    models: List[Dict[str, Any]] = field(default_factory=list)
    n_models_auto: int = 5
    trend_start_month: int = 202101
    trend_end_month: int = 202508
    trend_start_h50_s: float = 4.0
    trend_doubling_months: float = 6.0
    trend_noise_sd_log: float = 0.12
    trend_baseline_ref_sec: Optional[float] = None

    # Bootstrap
    n_boot: int = 200

    # Attempts per model-task pair
    attempts_per_pair: int = 15

    # Monitors
    monitors: List[MonitorConfig] = field(default_factory=lambda: [MonitorConfig()])

    # RNG
    seed: int = 42

    # Human FE fit hyperparameters
    human_l2_task: float = 1.0
    human_l2_part: float = 1.0
    human_l2_betaC: float = 0.01
    human_l2_mu: float = 0.0
    human_l2_delta_task: float = 0.25
    human_opt_iters: int = 1200
    human_opt_lr: float = 0.05

    def validate(self) -> None:
        assert self.n_t_bins >= 1
        assert self.tasks_per_bin >= 1
        assert 0 < self.t_seconds_min < self.t_seconds_max
        assert self.repeats_per_condition >= 1
        # covert-overhead schema must be present and consistent
        assert self.c_over_bins is not None and len(self.c_over_bins) >= 1, "c_over_bins must be provided"
        assert self.c_over_mix_by_t_bin is not None and len(self.c_over_mix_by_t_bin) == self.n_t_bins, "c_over_mix_by_t_bin must have n_t_bins rows"
        M = len(self.c_over_bins)
        for i, rng in enumerate(self.c_over_bins):
            assert 'lo_s' in rng and 'hi_s' in rng, "c_over_bins entries must have lo_s and hi_s"
            assert 0.0 <= float(rng['lo_s']) < float(rng['hi_s']), "each covert-overhead range must satisfy 0 <= lo_s < hi_s"
        for b, row in enumerate(self.c_over_mix_by_t_bin):
            assert len(row) == M, f"c_over_mix_by_t_bin[{b}] must have length {M}"
            assert sum([float(x) for x in row]) > 0.0, "mixture weights per T-bin must sum to > 0"
        for m in self.monitors:
            assert m.fpr_target >= 0.0
            if m.fpr_short is not None: assert m.fpr_short >= 0.0
            if m.fpr_med is not None: assert m.fpr_med >= 0.0
            if m.fpr_long is not None: assert m.fpr_long >= 0.0
