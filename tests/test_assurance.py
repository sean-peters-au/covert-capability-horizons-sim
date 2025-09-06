import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.assurance import assurance_all_gates


def test_assurance_all_gates_simple_mix():
    # 10 seeds; 6 pass both gates; per-gate rates: h50=0.8, coverage=0.7
    records = [
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": True},
        {"h50": True,  "coverage": False},
        {"h50": True,  "coverage": False},
        {"h50": False, "coverage": True},
        {"h50": False, "coverage": False},
    ]
    out = assurance_all_gates(records)
    assert abs(out["assurance"] - 0.6) < 1e-9
    assert abs(out["per_gate_rate"].get("h50", 0.0) - 0.8) < 1e-9
    assert abs(out["per_gate_rate"].get("coverage", 0.0) - 0.7) < 1e-9

