import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def matplotlib_available():
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def test_plot_pareto_simple_minimal(tmp_path):
    if not matplotlib_available():
        import pytest
        pytest.skip("matplotlib not available")

    # Minimal inputs
    dfa = pd.DataFrame({
        "attempts_per_pair": [10, 20],
        "tasks_per_bin": [2, 2],
        "n_participants": [2, 2],
        "repeats_per_condition": [1, 1],
        "assurance": [0.6, 0.8],
        "cost_total": [1000, 1500],
    })
    dfa.to_csv(tmp_path / "design_search.csv", index=False)
    dfa.iloc[[1]].to_csv(tmp_path / "pareto.csv", index=False)
    # Provide a minimal best_design.json selecting the second row
    (tmp_path / "best_design.json").write_text(
        '{"design": {"attempts_per_pair": 20, "tasks_per_bin": 2, "n_participants": 2, "repeats_per_condition": 1}, "assurance": 0.8, "feasible": true, "cost": {"cost_total": 1500}}'
    )

    from scripts.make_plots import plot_pareto_simple
    out = plot_pareto_simple(tmp_path, tmp_path)
    assert out.exists()
