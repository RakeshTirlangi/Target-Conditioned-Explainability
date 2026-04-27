# config.py

import torch

# Select device (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed sequence length for tokenization
MAX_LEN = 128

# SHAP parameters
MAX_EVALS = 50          # controls speed vs quality
N_BACKGROUND = 15       # initial pool size
RANDOM_SEED = 42

# XAI metrics
TOP_K = 5
N_SIMILAR = 3

# Paths
DATA_PATH = "data/cleaned_tweets.csv"
OUTPUT_PATH = "outputs/metrics/xai_metrics_results.csv"