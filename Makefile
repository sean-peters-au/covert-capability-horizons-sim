.PHONY: help py lock sync test run sweep clean

PY ?= 3.11

help:
	@echo "Targets:"
	@echo "  make py     # ensure Python $(PY) via uv"
	@echo "  make lock   # uv lock deps"
	@echo "  make sync   # create .venv and install"
	@echo "  make test   # run pytest"
	@echo "  make run    # example simulate run"
	@echo "  make sweep  # example sweep run"
	@echo "  make clean  # remove .venv and cache"

py:
	uv python install $(PY)

lock: py
	uv lock

sync: py
	uv sync

test:
	uv run -m pytest -q

run:
	uv run -m cch_sim.cli simulate --config scenarios/example.yaml --out out/sim

sweep:
	uv run -m cch_sim.cli sweep --scenarios scenarios/example.yaml --out out/sweep

clean:
	rm -rf .venv
	uv cache clean
