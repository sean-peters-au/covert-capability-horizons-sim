from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .assurance import assurance_all_gates


def aggregate_assurance_for_design(
    design_id: str, pass_records: List[Dict[str, bool]]
) -> Dict[str, object]:
    """Aggregate assurance for a single design from per-seed gate pass records.

    Returns a flat dict suitable for CSV/JSON rows.
    """
    out = assurance_all_gates(pass_records)
    row: Dict[str, object] = {
        "design_id": design_id,
        "assurance": out.get("assurance"),
    }
    per_gate = out.get("per_gate_rate", {}) or {}
    if isinstance(per_gate, dict):
        for k, v in per_gate.items():
            row[f"gate_{k}_rate"] = v
    row["seeds"] = len(pass_records)
    return row


def aggregate_assurance_table(
    design_records: Dict[str, List[Dict[str, bool]]],
) -> List[Dict[str, object]]:
    """Aggregate assurance rows for multiple designs.

    Returns a list of flat rows suitable for CSV/JSON.
    """
    rows: List[Dict[str, object]] = []
    for design_id in sorted(design_records.keys()):
        rows.append(aggregate_assurance_for_design(design_id, design_records[design_id]))
    return rows


def write_assurance_table(
    rows: List[Dict[str, object]], out_dir: str | Path, basename: str = "assurance"
) -> Dict[str, str]:
    """Write assurance rows to CSV and JSON; returns paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_csv = out / f"{basename}.csv"
    p_json = out / f"{basename}.json"
    df = pd.DataFrame(rows)
    df.to_csv(p_csv, index=False)
    p_json.write_text(json.dumps(rows, indent=2))
    return {"csv": str(p_csv), "json": str(p_json)}
