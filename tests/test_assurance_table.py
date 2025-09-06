import sys
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cch_sim.integrations import aggregate_assurance_table, write_assurance_table


def test_aggregate_assurance_table_two_designs():
    design_records = {
        "A": [
            {"h50": True, "coverage": True},
            {"h50": True, "coverage": True},
            {"h50": True, "coverage": True},
            {"h50": True, "coverage": True},
            {"h50": False, "coverage": True},
        ],
        "B": [
            {"h50": True, "coverage": False},
            {"h50": False, "coverage": True},
            {"h50": True, "coverage": True},
            {"h50": False, "coverage": False},
            {"h50": True, "coverage": False},
        ],
    }
    rows = aggregate_assurance_table(design_records)
    # Sorted by design id
    assert [r["design_id"] for r in rows] == ["A", "B"]
    ra, rb = rows
    assert abs(ra["assurance"] - 0.8) < 1e-9
    assert abs(rb["assurance"] - 0.2) < 1e-9
    assert abs(ra["gate_h50_rate"] - 0.8) < 1e-9
    assert abs(ra["gate_coverage_rate"] - 1.0) < 1e-9


def test_write_assurance_table_roundtrip(tmp_path: Path):
    rows = [
        {"design_id": "A", "assurance": 0.8, "gate_h50_rate": 0.8, "gate_coverage_rate": 1.0, "seeds": 5},
        {"design_id": "B", "assurance": 0.2, "gate_h50_rate": 0.6, "gate_coverage_rate": 0.4, "seeds": 5},
    ]
    out = write_assurance_table(rows, tmp_path, basename="assure")
    # Files exist
    p_csv = Path(out["csv"]); p_json = Path(out["json"])
    assert p_csv.exists() and p_json.exists()
    # CSV content matches number of rows and keys
    df = pd.read_csv(p_csv)
    assert df.shape[0] == 2 and set(df.columns) >= {"design_id", "assurance", "seeds"}
    # JSON content is list of rows
    js = json.loads(p_json.read_text())
    assert isinstance(js, list) and len(js) == 2

