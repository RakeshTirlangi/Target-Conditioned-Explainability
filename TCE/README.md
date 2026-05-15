# TCE: Target-Conditioned Explainability

Official implementation of **"TCE: Quantifying Explanation Stability in Stance Detection via Target-Conditioned Explainability"**.

TCE is an evaluation framework that quantifies how consistently a stance-detection model explains its predictions when the target entity is rephrased to a semantic equivalent. It compares SHAP attribution vectors across a tweet's ground-truth target and three semantically similar targets, producing three complementary metrics:

| Metric | Captures | Range |
|---|---|---|
| **ESS** — Explanation Shift Score | Mean absolute attribution shift (magnitude) | `[0, ∞)`, lower is better |
| **CED** — Cosine Explanation Divergence | `1 − cos(φ_GT, φ_sim)` (direction) | `[0, 2]`, lower is better |
| **TFR** — Token Flip Rate | Fraction of top-k tokens that change (identity) | `[0, 1]`, lower is better |

> Paper code: <https://github.com/RakeshTirlangi/Target-Conditioned-Explainability>

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/RakeshTirlangi/Target-Conditioned-Explainability.git
cd Target-Conditioned-Explainability/TCE
pip install -r requirements.txt

# 2. Place the dataset (or use the sample)
#    data/cleaned_tweets.csv  -- full OpenTarget sample
#    data/sample.csv          -- 5-row smoke-test CSV (included)

# 3. Smoke test on 5 rows (uses base HuggingFace weights, runs on CPU)
python scripts/run_tce.py --config configs/deberta.yaml --sample-size 5 \
    --data-path data/sample.csv

# 4. Full run for one model
python scripts/run_tce.py --config configs/deberta.yaml
```

Outputs land in `outputs/<model>/` and contain four CSVs:

- `*_pairs.csv` — one row per (instance, similar-target) pair with ESS / CED / TFR
- `*_per_instance.csv` — averaged across the 3 similar targets
- `*_per_stance.csv` — aggregated by GT stance class
- `*_per_target.csv` — aggregated by GT target

---

## Reproducing the paper

The paper reports four model comparisons + sensitivity sweeps + correlation analyses. Run them sequentially:

```bash
# Per-model TCE (each ~30 min on GPU, several hours on CPU)
for cfg in deberta bert distilbert roberta; do
  python scripts/run_tce.py --config configs/${cfg}.yaml
done

# Hyperparameter sensitivity (Section VI.C)
python scripts/run_sensitivity.py --config configs/deberta.yaml

# Pairwise Wilcoxon significance (Section V.E)
python scripts/run_significance.py \
    --metric CED \
    --csvs outputs/deberta/tce_deberta_per_instance.csv \
           outputs/bert/tce_bert_per_instance.csv \
           outputs/distilbert/tce_distilbert_per_instance.csv \
           outputs/roberta/tce_roberta_per_instance.csv \
    --names DeBERTa BERT DistilBERT RoBERTa \
    --output outputs/significance/CED_pairwise.csv

# Correlation with prediction quality (Section V.F)
python scripts/run_correlation.py \
    --instances outputs/deberta/tce_deberta_per_instance.csv \
    --pairs     outputs/deberta/tce_deberta_pairs.csv \
    --output    outputs/deberta/correlations.csv

# Aggregate everything into a cross-model summary
python scripts/aggregate_results.py \
    --pairs outputs/deberta/tce_deberta_pairs.csv \
            outputs/bert/tce_bert_pairs.csv \
            outputs/distilbert/tce_distilbert_pairs.csv \
            outputs/roberta/tce_roberta_pairs.csv \
    --names DeBERTa BERT DistilBERT RoBERTa \
    --output outputs/cross_model_summary.csv

# Plots (paper figures 1-3)
python scripts/make_plots.py --results-dir outputs/deberta --prefix tce_deberta
```

A `Makefile` provides shorthand for the same:

```bash
make run-deberta
make sensitivity-deberta
make significance
make correlations
make plots
```

---

## Using your own fine-tuned weights

The shipped configs point at base HuggingFace checkpoints (`bert-base-uncased`, `roberta-base`, etc.) so the code runs out of the box. To reproduce the paper's results you need stance-fine-tuned weights. Either:

1. **Edit the YAML config:** point `model.name_or_path` to your local checkpoint directory or HuggingFace hub ID.

   ```yaml
   model:
     name_or_path: ./checkpoints/deberta-stance
   ```

2. **Override on the command line:**

   ```bash
   python scripts/run_tce.py --config configs/deberta.yaml \
       --data-path /path/to/your/dataset.csv
   ```

Stance label order must be `["AGAINST", "NONE", "FAVOR"]` (indices 0, 1, 2) to match the paper.

---

## Repository structure

```
TCE/
├── README.md                       # You are here
├── LICENSE                         # MIT
├── CITATION.cff                    # How to cite this work
├── pyproject.toml                  # Build + tooling config
├── requirements.txt                # Runtime dependencies (pinned ranges)
├── requirements-dev.txt            # + pytest / ruff / mypy
├── Makefile                        # Common workflow shortcuts
│
├── configs/                        # YAML configs (one per model + a default)
│   ├── default.yaml                # Base config; all others inherit via `_base_`
│   ├── deberta.yaml
│   ├── bert.yaml
│   ├── distilbert.yaml
│   └── roberta.yaml
│
├── data/
│   ├── README.md                   # Data schema + download instructions
│   ├── sample.csv                  # 5-row smoke-test dataset
│   └── cleaned_tweets.csv          # (Place full dataset here)
│
├── src/tce/                        # Main package
│   ├── __init__.py
│   ├── config.py                   # YAML loader with `_base_` inheritance
│   ├── data.py                     # Dataset loading + validation
│   ├── preprocessing.py            # Input formatting + similar-target parsing
│   ├── model.py                    # HuggingFace model loader (ModelBundle)
│   ├── inference.py                # Batched softmax inference + confidence margin
│   ├── shap_explainer.py           # PartitionExplainer wrapper + background selection
│   ├── metrics.py                  # ESS, CED, TFR
│   ├── pipeline.py                 # End-to-end TCE pipeline + aggregations
│   ├── statistics.py               # Bootstrap CI, Wilcoxon, correlation
│   ├── sensitivity.py              # Hyperparameter sweeps (max_evals, bg, k)
│   ├── visualization.py            # Plotting (overview, correlations)
│   └── utils.py                    # Seeding, logging, device, paths
│
├── scripts/                        # CLI entry points
│   ├── run_tce.py
│   ├── run_sensitivity.py
│   ├── run_correlation.py
│   ├── run_significance.py
│   ├── aggregate_results.py
│   └── make_plots.py
│
└── tests/                          # pytest
    ├── test_metrics.py
    ├── test_preprocessing.py
    ├── test_config.py
    ├── test_statistics.py
    └── test_smoke.py               # End-to-end with a tiny HF model
```

---

## Dataset schema

`data/cleaned_tweets.csv` is a derivative of the OpenTarget dataset (Akash et al., ACL Findings 2025) augmented with LLM-generated similar targets. Required columns:

| Column | Meaning |
|---|---|
| `cleaned_tweet` | Pre-processed tweet text |
| `GT Target` | Ground-truth target entity |
| `GT Stance` | One of `AGAINST`, `NONE`, `FAVOR` |
| `similar targets` | Python-list-as-string of 3 LLM-generated similar targets |

See [`data/README.md`](data/README.md) for the full set of OpenTarget columns and provenance.

---

## Configuration system

All experiment parameters live in YAML configs under `configs/`. The base file is `configs/default.yaml`; per-model files inherit from it via `_base_: default.yaml` and override only what differs. Every CLI script accepts `--config <path>` and prints the fully-resolved config alongside its outputs (`resolved_config.yaml`), so every run is independently reproducible.

Key blocks:

```yaml
shap:
  max_evals: 50                 # Forward passes per SHAP call
  masker_regex: '\W+'           # Word-level masker
  n_background_candidates: 15   # Pool size for background centroid
  background_size: 1            # Single representative `none` tweet

metrics:
  top_k: 5                      # TFR's top-k threshold

statistics:
  bootstrap_samples: 1000
  bootstrap_ci: 0.95
  bonferroni_correction: true

sensitivity:
  k_values: [3, 5, 10, 20]
  max_evals_values: [25, 50, 100, 250, 500]
  background_sizes: [1, 5, 15, 50]
  sensitivity_sample_size: 500
```

---

## Testing

```bash
# All tests (including smoke test that downloads a small HF model)
pytest tests/

# Unit tests only — no network, no model downloads
TCE_SKIP_SMOKE=1 pytest tests/ -k "not smoke"

# Or via make
make test-fast
```

---

## Reproducibility notes

Every run writes a `resolved_config.yaml` next to its outputs that captures every effective hyperparameter (including command-line overrides and inherited values). The Python and PyTorch random seeds are set from `experiment.seed` (default `42`). The same seed across re-runs produces identical SHAP attribution sets and identical bootstrap CIs.

For hard reproducibility across hardware, set `CUDA_VISIBLE_DEVICES` and avoid mixed precision; PyTorch is not bit-deterministic across GPU architectures, so CIs may shift by < 1% between platforms.

---

## Citation

If you use TCE in your research, please cite the paper:

```bibtex
@inproceedings{tce2026,
  title  = {TCE: Quantifying Explanation Stability in Stance Detection via Target-Conditioned Explainability},
  author = {Tirlangi, Rakesh and Samineni, Bhavani},
  year   = {2026}
}
```

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [`LICENSE`](LICENSE).
