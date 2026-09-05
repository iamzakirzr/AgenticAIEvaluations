# =============================================================================
# Every command you need, in one place. Run `make` to see them.
# =============================================================================

.DEFAULT_GOAL := help
PY := .venv/bin/python

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------

.PHONY: setup
setup:  ## Create .venv and install everything (needs `uv`)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"
	@echo ""
	@echo "Next: start Ollama and pull the models ->  make models"

.PHONY: models
models:  ## Pull the Ollama models the judged tier needs
	ollama pull llama3.1:8b
	ollama pull nomic-embed-text
	@echo "Optional second judge for the calibration lesson:"
	@echo "  ollama pull qwen2.5:7b"

# --- Tests -------------------------------------------------------------------

.PHONY: test
test:  ## FAST tier: no model, no network, ~1s. This is the CI gate.
	$(PY) -m pytest -q

.PHONY: test-ollama
test-ollama:  ## Real embeddings and real generation (needs Ollama)
	$(PY) -m pytest -m ollama -v

.PHONY: test-judge
test-judge:  ## The expensive judged metrics (needs Ollama, slow, non-deterministic)
	$(PY) -m pytest -m judge -v

.PHONY: test-all
test-all:  ## Everything except the SaaS tier
	$(PY) -m pytest -m "not saas" -v

# --- Lessons -----------------------------------------------------------------

.PHONY: lesson-embeddings
lesson-embeddings:  ## Run the embeddings walkthrough (prints, does not assert)
	$(PY) 01_embeddings/walkthrough.py

.PHONY: chat
chat:  ## Serve the RAG chatbot at http://localhost:8000
	.venv/bin/uvicorn 02_langchain.server:app --reload --port 8000

# --- Regression gating -------------------------------------------------------

.PHONY: gate
gate:  ## Compare current scores against the recorded baseline (CI's gate)
	$(PY) scripts/run_regression_gate.py

.PHONY: gate-fast
gate-fast:  ## Same gate, retrieval metrics only -- no model needed
	$(PY) scripts/run_regression_gate.py --deterministic

.PHONY: baseline
baseline:  ## Record the CURRENT scores as the new reference. Review the diff!
	@echo "Moving a baseline is how a regression disappears. Check the diff."
	$(PY) scripts/run_regression_gate.py --record --repetitions 5

.PHONY: calibrate
calibrate:  ## Measure whether your judge agrees with human labels (Cohen's kappa)
	$(PY) 04_deepeval/calibrate.py

# --- Experiments -------------------------------------------------------------

.PHONY: experiment-chunking
experiment-chunking:  ## Show how chunk size moves retrieval metrics
	$(PY) 01_embeddings/experiment_chunk_size.py

.PHONY: lint
lint:  ## Ruff check + format check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

.PHONY: clean
clean:  ## Remove caches and generated artifacts
	rm -rf .artifacts .chroma .pytest_cache .ruff_cache .deepeval
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
