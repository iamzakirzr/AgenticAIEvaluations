# Run `make` to list targets.
.DEFAULT_GOAL := help
PY := .venv/bin/python

.PHONY: help
help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup:  ## Create .venv and install (needs uv)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

.PHONY: models
models:  ## Pull the default Ollama models
	ollama pull llama3.1:8b

# --- Tiers -------------------------------------------------------------------

.PHONY: test
test:  ## Unit tier: no model, no network, ~1s (the PR gate)
	$(PY) -m pytest

.PHONY: eval
eval:  ## llm tier: every judged suite (needs a judge)
	$(PY) -m pytest -m llm -v -rs

.PHONY: eval-rag
eval-rag:  ## llm tier: RAG quality only
	$(PY) -m pytest -m llm tests/evals/rag -v -rs

.PHONY: eval-agent
eval-agent:  ## llm tier: agent evals only
	$(PY) -m pytest -m llm tests/evals/agent -v -rs

.PHONY: eval-chat
eval-chat:  ## llm tier: multi-turn conversations only
	$(PY) -m pytest -m llm tests/evals/chat -v -rs

.PHONY: eval-safety
eval-safety:  ## llm tier: curated safety probes only
	$(PY) -m pytest -m llm tests/evals/safety -v -rs

.PHONY: redteam
redteam:  ## redteam tier: DeepTeam attacks (slowest)
	$(PY) -m pytest -m redteam -v -rs -s

# --- Reporting ---------------------------------------------------------------

.PHONY: gate
gate:  ## Aggregate the last run and compare against baselines/baseline.json
	$(PY) -m evalkit.reporting.gate

.PHONY: baseline
baseline:  ## Record the last run as the baseline. Review the diff before committing!
	$(PY) -m evalkit.reporting.gate --record

# --- Hygiene -----------------------------------------------------------------

.PHONY: lint
lint:  ## Ruff lint + format check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

.PHONY: format
format:  ## Apply ruff formatting and safe fixes
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

.PHONY: clean
clean:  ## Remove caches and run outputs
	rm -rf reports .deepeval .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
