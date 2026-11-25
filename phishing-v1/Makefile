.PHONY: install pipeline test clean lint

PYTHON := python3
PIP := pip3

# ── Install dependencies ──────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

# ── Run the full pipeline ─────────────────────────────────────────
pipeline:
	$(PYTHON) run_all.py

# ── Run tests ─────────────────────────────────────────────────────
# Run tests/test_xai_metrics.py first (no data needed).
# Run tests/test_no_leakage.py after the data pipeline has been run.
test-xai:
	$(PYTHON) -m pytest tests/test_xai_metrics.py -v

test-leakage:
	$(PYTHON) -m pytest tests/test_no_leakage.py -v

test:
	$(PYTHON) -m pytest tests/ -v

# ── Individual pipeline stages (for debugging / partial reruns) ───
fetch:
	$(PYTHON) src/data/fetch_phishtank.py
	$(PYTHON) src/data/fetch_legitimate.py

split:
	$(PYTHON) src/data/dedupe_and_split.py

features:
	$(PYTHON) src/features/extract_features.py

train:
	$(PYTHON) src/models/train.py

evaluate:
	$(PYTHON) src/models/evaluate.py

xai:
	$(PYTHON) src/xai/run_shap_lime.py
	$(PYTHON) src/xai/xai_metrics.py

graph:
	$(PYTHON) src/graph/build_graph.py
	$(PYTHON) src/graph/cluster_campaigns.py
	$(PYTHON) src/eval/cluster_eval.py

benchmark:
	$(PYTHON) src/eval/latency_benchmark.py

# ── Clean generated outputs ───────────────────────────────────────
clean:
	rm -rf results/models results/figures results/graph
	rm -f results/*.json results/*.md
	rm -rf data/processed data/splits

# ── Clean everything including caches ────────────────────────────
clean-all: clean
	rm -rf data/cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# ── Lint ──────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503 || true
