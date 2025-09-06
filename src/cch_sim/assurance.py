from __future__ import annotations

from typing import Dict, List
import numpy as np


def assurance_all_gates(pass_records: List[Dict[str, bool]]) -> Dict[str, float | Dict[str, float]]:
    """Compute assurance as the fraction of seeds where all gates pass.

    pass_records: list of dicts per seed, e.g., [{"h50": True, "coverage": False}, ...]
    Returns: {"assurance": float, "per_gate_rate": {gate: rate}}
    """
    if not pass_records:
        return {"assurance": float("nan"), "per_gate_rate": {}}
    gates = sorted({k for d in pass_records for k in d.keys()})
    per_gate = {}
    for g in gates:
        rate = float(np.mean([bool(d.get(g, False)) for d in pass_records]))
        per_gate[g] = rate
    all_pass = [all(bool(d.get(g, False)) for g in gates) for d in pass_records]
    assurance = float(np.mean(all_pass))
    return {"assurance": assurance, "per_gate_rate": per_gate}

