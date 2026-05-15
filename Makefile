.PHONY: help install dev test lint typecheck eval smoke serve docker clean

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install    pip install -r requirements.txt"
	@echo "  dev        run uvicorn with --reload"
	@echo "  test       pytest tests/ -v  (mocked Groq, safe in CI)"
	@echo "  lint       ruff check ."
	@echo "  typecheck  mypy backend/"
	@echo "  eval       python -m eval.run_eval  (needs real GROQ_API_KEY)"
	@echo "  smoke      python scripts/smoke_demo.py  (needs real GROQ_API_KEY)"
	@echo "  serve      production-shaped uvicorn run"
	@echo "  docker     docker compose up --build"
	@echo "  clean      remove caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy backend/ --ignore-missing-imports

eval:
	$(PYTHON) -m eval.run_eval

smoke:
	$(PYTHON) scripts/smoke_demo.py

serve:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
