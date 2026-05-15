# Dataset

## Files in this directory

| File | Size | Purpose |
|---|---|---|
| `cleaned_tweets.csv` | ~3 MB | Full OpenTarget-derived dataset (place here) |
| `sample.csv` | < 1 KB | 5-row smoke-test dataset (included) |

## Source

The dataset is derived from the **OpenTarget** stance detection corpus introduced by Akash et al. (Findings of ACL 2025). It draws on the TSE and VAST sub-corpora and is augmented with three semantically similar targets per row, generated via chain-of-thought prompting over Gemini, Qwen, and LLaMA (Samineni & Bindu, "PrompTEL").

## Required columns

The TCE pipeline requires the following columns (defaults configurable in `configs/default.yaml > data`):

| Column | Type | Meaning |
|---|---|---|
| `cleaned_tweet` | str | Pre-processed tweet text |
| `GT Target` | str | Ground-truth target entity |
| `GT Stance` | str | One of `AGAINST`, `NONE`, `FAVOR` (case-insensitive) |
| `similar targets` | str | Python list-as-string with 3 similar targets, e.g. `"['climate crisis', 'global warming', 'env policy']"`. Comma-separated strings also accepted. |

## Optional columns (informational; not consumed by TCE)

The full OpenTarget-augmented CSV in our experiments additionally carries the original `Tweet` text plus the per-LLM generations (`Gemini Target`, `Gemini Stance`, `LLaMA Target`, `LLaMA Stance`, `Qwen Target`, `Qwen Stance`, and a `Dataset` source-tag column). These are kept for provenance but ignored by the pipeline.

## Sample row

```
cleaned_tweet, GT Target, GT Stance, similar targets
"Climate is changing rapidly.", "climate change", FAVOR, "['global warming', 'environmental policy', 'climate crisis']"
```

## Stance label convention

Indices for fine-tuned classification heads (paper Section IV.B):

| Index | Label   |
|------:|:--------|
| 0     | AGAINST |
| 1     | NONE    |
| 2     | FAVOR   |

The TCE code uppercases and strips whitespace on read.
