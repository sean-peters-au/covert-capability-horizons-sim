from __future__ import annotations

from cch_sim.design_search import compute_pareto, select_min_cost_feasible


def test_pareto_frontier_simple():
    rows = [
        {"cost_total": 100, "assurance": 0.7, "id": 0},
        {"cost_total": 120, "assurance": 0.9, "id": 1},
        {"cost_total": 80,  "assurance": 0.6, "id": 2},
        {"cost_total": 120, "assurance": 0.7, "id": 3},
    ]
    pareto = compute_pareto(rows)
    ids = sorted(r["id"] for r in pareto)
    # (80,0.6), (100,0.7), (120,0.9) are non-dominated
    assert ids == [0, 1, 2]


def test_select_min_cost_feasible_and_fallback():
    rows = [
        {"cost_total": 100, "assurance": 0.75},
        {"cost_total": 150, "assurance": 0.8},  # feasible
        {"cost_total": 140, "assurance": 0.85}, # feasible and cheaper than any higher assurance? actually cheaper than 150? no, 140 < 150 -> should win
    ]
    best, feasible = select_min_cost_feasible(rows, assurance_target=0.8)
    assert feasible is True
    assert best["cost_total"] == 140
    # If target too high, fallback to highest assurance then lower cost tie-breaker
    best2, feasible2 = select_min_cost_feasible(rows, assurance_target=0.9)
    assert feasible2 is False
    assert best2["assurance"] == 0.85
