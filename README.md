# Covert Capability Horizons — Simulation (cch‑sim)

This repository accompanies the essay “Deceptive Capability Horizons and Control Protocols.” If you’re landing here from the write‑up, you already know the question we care about: can we run a small, carefully designed study that credibly detects early covert capability signal, and track how that capability trends over releases? The simulation here exists to answer exactly that. It is deliberately narrow in scope.

https://sean-peters-au.github.io/2025/08/08/deceptive-capability-horizons-and-control-protocols.html

This codebase is preliminary. I built it quickly to pressure‑test the study design and to run a sensitivity analysis that felt honest. The pipeline is Bayesian end‑to‑end (NumPyro/JAX), with strict diagnostics, and a small design search that trades off assurance against explicit costs. Early runs suggest that a well‑chosen pilot in the $10–20k range is enough to capture early signal under realistic noise, but treat that as directional while we continue to iterate and ground the scenarios.

## What the pipeline does
Stage 1 fits human baselines and covert overhead on log‑time, handling censoring cleanly. Stage 2 fits model success against absolute Δ seconds and extracts Δ50 per model/monitor. We then trend Δ50 over releases using per‑draw regression to get a posterior over doubling months. Gates formalize what “good enough” means (precision, identifiability, and trend recovery), and assurance is simply the fraction of seeds that pass all active gates. A small, explicit cost model lets us pick the lowest‑cost design that meets the target assurance.

## Quick Start
You need uv to manage Python and dependencies. The Makefile provides simple, reproducible commands.

1) Install and set up a local environment
- `make sync`

2) Pilot design search (scenario + plots)
- `make pilot-design`

This runs the pilot scenario in `scenarios/pilot_signal.yaml` against the study grid in `scenarios/study_search_pilot.yaml`, writes results to `out/design_search_pilot_signal`, and generates plots under `out/design_search_pilot_signal/plots`.

## Run Your Own Scenarios
- Single simulate run (writes Δ50 per monitor and trend summaries):
  - `uv run -m cch_sim.cli simulate --config <your_scenario.yaml> --out out/your_sim`
- Sweep across scenarios with assurance tables:
  - `uv run -m cch_sim.cli sweep --scenarios <your_scenarios.yaml> --out out/your_sweep`
- Automated design search for one scenario:
  - `uv run -m cch_sim.cli design-search --scenario <your_scenario.yaml> --search <your_study_search.yaml> --out out/your_design_search`

## Notes
This is a work in progress. The methodology is becoming more stable, and the ergonomics will improve as we go. If you spot something off, please flag it.
