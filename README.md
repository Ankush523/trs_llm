# Reasoning LLM

`reasoning_llm` is a local-first implementation of a `Thinking with Reasoning Skills (TRS)` pipeline.

The project separates:

- offline trace generation and skill distillation
- library compilation and retrieval index building
- online direct or TRS-guided inference
- experiment reporting and comparison

The repository is intentionally stdlib-first so the core pipeline and tests run without extra dependencies. Optional integrations such as `FastAPI` and `PyYAML` are loaded lazily if installed.

## Quick start

1. Create a virtualenv and install optional dependencies if you want YAML parsing or API serving.
2. Prepare datasets into `data/processed/`.
3. Generate direct traces into `artifacts/traces/`.
4. Distill skill cards into `artifacts/skills/`.
5. Build a library snapshot into `artifacts/indexes/`.
6. Run `direct` and `trs` evaluations and compare the reports.

Example smoke flow:

```bash
python3 -m trs.cli datasets prepare \
  --dataset math_hendrycks \
  --source data/raw/smoke/math_hendrycks.jsonl \
  --output data/processed/smoke_math.jsonl

python3 -m trs.cli traces generate \
  --dataset-file data/processed/smoke_math.jsonl \
  --task-type math \
  --dataset-name math_hendrycks \
  --model-config configs/models/generator.yaml \
  --prompt-config configs/prompts/direct.yaml \
  --output artifacts/traces/smoke_math_traces.jsonl

python3 -m trs.cli skills distill \
  --dataset-file data/processed/smoke_math.jsonl \
  --traces-file artifacts/traces/smoke_math_traces.jsonl \
  --model-config configs/models/summarizer.yaml \
  --prompt-config configs/prompts/skill_extract.yaml \
  --output artifacts/skills/smoke_math_skills.jsonl

python3 -m trs.cli library build \
  --cards-file artifacts/skills/smoke_math_skills.jsonl \
  --profile-config configs/retrieval/math_bm25.yaml \
  --output-dir artifacts/indexes/smoke_math_library

python3 -m trs.cli eval run \
  --mode trs \
  --dataset-file data/processed/smoke_math.jsonl \
  --task-type math \
  --dataset-name math_hendrycks \
  --model-config configs/models/inference.yaml \
  --prompt-config configs/prompts/trs_infer.yaml \
  --library-dir artifacts/indexes/smoke_math_library \
  --output-dir artifacts/runs/smoke_math_trs
```

## Dataset expectations

- `math_hendrycks`: JSONL or JSON array with `id`, `problem`, `answer`, optional `solution`.
- `mbpp`: JSONL or JSON array with `task_id`, `text`, `tests`, optional `code`.
- `humaneval_plus`: JSONL or JSON array with `task_id`, `prompt`, `entry_point`, `tests`.

Smoke datasets are included under `data/raw/smoke/` for deterministic local validation.

## Optional dependencies

- `PyYAML`: load ordinary YAML configs instead of JSON-shaped YAML.
- `FastAPI` and `uvicorn`: enable `trs serve api`.

## Running tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
