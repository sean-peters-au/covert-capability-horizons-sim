import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.integrations import aggregate_assurance_for_design


def test_aggregate_assurance_for_design_two_examples():
    # Design A: 5 seeds, all gates pass in 4 seeds
    records_A = [
        {"h50": True, "coverage": True},
        {"h50": True, "coverage": True},
        {"h50": True, "coverage": True},
        {"h50": True, "coverage": True},
        {"h50": False, "coverage": True},
    ]
    row_A = aggregate_assurance_for_design("A", records_A)
    assert row_A["design_id"] == "A"
    assert abs(row_A["assurance"] - 0.8) < 1e-9  # 4/5 seeds pass all
    assert abs(row_A["gate_h50_rate"] - 0.8) < 1e-9
    assert abs(row_A["gate_coverage_rate"] - 1.0) < 1e-9
    assert row_A["seeds"] == 5

    # Design B: 5 seeds, mixed; only 1 seed passes both
    records_B = [
        {"h50": True, "coverage": False},
        {"h50": False, "coverage": True},
        {"h50": True, "coverage": True},
        {"h50": False, "coverage": False},
        {"h50": True, "coverage": False},
    ]
    row_B = aggregate_assurance_for_design("B", records_B)
    assert row_B["design_id"] == "B"
    assert abs(row_B["assurance"] - 0.2) < 1e-9  # 1/5
    assert abs(row_B["gate_h50_rate"] - 0.6) < 1e-9
    assert abs(row_B["gate_coverage_rate"] - 0.4) < 1e-9
    assert row_B["seeds"] == 5
