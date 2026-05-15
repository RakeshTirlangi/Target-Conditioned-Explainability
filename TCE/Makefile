# Convenience targets. On Windows, run them with `make` from Git Bash, WSL,
# or just copy the underlying commands.

PYTHON ?= python
PIP    ?= pip

.PHONY: help install install-dev test test-fast lint clean smoke \
        run-deberta run-bert run-distilbert run-roberta \
        sensitivity-deberta significance correlations plots

help:
	@echo "Targets:"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install runtime + dev dependencies"
	@echo "  test            Run all tests (including smoke)"
	@echo "  test-fast       Run unit tests only (skip smoke)"
	@echo "  lint            Run ruff"
	@echo "  smoke           Run TCE on a 5-row subset of the dataset"
	@echo "  run-deberta     Run full TCE for DeBERTa"
	@echo "  run-bert        Run full TCE for BERT-base"
	@echo "  run-distilbert  Run full TCE for DistilBERT"
	@echo "  run-roberta     Run full TCE for RoBERTa"
	@echo "  sensitivity-deberta  Run hyperparameter sweeps for DeBERTa"
	@echo "  significance    Run pairwise Wilcoxon across all 4 model outputs"
	@echo "  correlations    Run TCE-vs-prediction-quality correlation for DeBERTa"
	@echo "  plots           Regenerate plots for DeBERTa"
	@echo "  clean           Remove build artifacts and caches"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/

test-fast:
	TCE_SKIP_SMOKE=1 $(PYTHON) -m pytest tests/ -k "not smoke"

lint:
	ruff check src/ scripts/ tests/

smoke:
	$(PYTHON) scripts/run_tce.py --config configs/deberta.yaml --sample-size 5

run-deberta:
	$(PYTHON) scripts/run_tce.py --config configs/deberta.yaml

run-bert:
	$(PYTHON) scripts/run_tce.py --config configs/bert.yaml

run-distilbert:
	$(PYTHON) scripts/run_tce.py --config configs/distilbert.yaml

run-roberta:
	$(PYTHON) scripts/run_tce.py --config configs/roberta.yaml

sensitivity-deberta:
	$(PYTHON) scripts/run_sensitivity.py --config configs/deberta.yaml

significance:
	$(PYTHON) scripts/run_significance.py \
		--metric CED \
		--csvs outputs/deberta/tce_deberta_per_instance.csv \
		       outputs/bert/tce_bert_per_instance.csv \
		       outputs/distilbert/tce_distilbert_per_instance.csv \
		       outputs/roberta/tce_roberta_per_instance.csv \
		--names DeBERTa BERT DistilBERT RoBERTa \
		--output outputs/significance/CED_pairwise.csv \
		--summary-output outputs/significance/CED_summary.csv

correlations:
	$(PYTHON) scripts/run_correlation.py \
		--instances outputs/deberta/tce_deberta_per_instance.csv \
		--pairs     outputs/deberta/tce_deberta_pairs.csv \
		--output    outputs/deberta/correlations.csv

plots:
	$(PYTHON) scripts/make_plots.py --results-dir outputs/deberta --prefix tce_deberta

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
