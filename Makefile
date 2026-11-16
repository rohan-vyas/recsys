PY := .venv/bin/python
PIP_INDEX := https://pypi.org/simple

.PHONY: venv data tune final ranker report test lint all

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url $(PIP_INDEX) -e ".[dev]"

data:            ## download ml-25m.zip (md5 checked) and build data/processed
	$(PY) -m recsys.cli data

tune:            ## all tuning on the val window only
	$(PY) -m recsys.cli tune-als
	$(PY) -m recsys.cli tune-tt

final:           ## refit on everything before test_start, score the test window once
	$(PY) -m recsys.cli final

ranker:          ## optional second stage (trained on val, applied to test)
	$(PY) -m recsys.ranker

report:          ## bootstrap CIs, paired tests, analysis -> results/metrics.json, results/report.md
	$(PY) -m recsys.cli report

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

all: data tune final ranker report
