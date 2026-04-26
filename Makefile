PYTHON ?= python3

.PHONY: test smoke-math smoke-coding

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

smoke-math:
	$(PYTHON) -m trs.cli datasets prepare --dataset math_hendrycks --source data/raw/smoke/math_hendrycks.jsonl --output data/processed/smoke_math.jsonl

smoke-coding:
	$(PYTHON) -m trs.cli datasets prepare --dataset mbpp --source data/raw/smoke/mbpp.jsonl --output data/processed/smoke_mbpp.jsonl
