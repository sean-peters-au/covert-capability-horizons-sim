"""Assurance aggregation utilities."""

from __future__ import annotations

import numpy as np


def assurance_all_gates(pass_records: list[dict[str, bool]]) -> dict[str, float | dict[str, float]]:
    """Compute assurance as the fraction of seeds where all gates pass.

    Args:
        pass_records: One dict per seed, e.g., ``[{"h50": True, "coverage": False}, ...]``.

    Returns:
        Dict with overall assurance and per‑gate pass rates:
        ``{"assurance": float, "per_gate_rate": {gate: rate}}``.
    """
    if not pass_records:
        return {"assurance": float("nan"), "per_gate_rate": {}}
    gates = sorted({k for d in pass_records for k in d})
    per_gate = {}
    for g in gates:
        rate = float(np.mean([bool(d.get(g, False)) for d in pass_records]))
        per_gate[g] = rate
    all_pass = [all(bool(d.get(g, False)) for g in gates) for d in pass_records]
    assurance = float(np.mean(all_pass))
    return {"assurance": assurance, "per_gate_rate": per_gate}
