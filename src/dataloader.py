# data_loader.py

import pandas as pd

# Required dataset columns
REQUIRED_COLS = ["cleaned_tweet", "GT Target", "GT Stance", "similar targets"]

def load_data(path: str) -> pd.DataFrame:
    """Load dataset and validate required columns."""
    
    df = pd.read_csv(path)
    
    # Ensure all required columns exist
    assert all(c in df.columns for c in REQUIRED_COLS), \
        f"Missing columns: {[c for c in REQUIRED_COLS if c not in df.columns]}"
    
    # Drop rows with missing values
    df = df[REQUIRED_COLS].dropna().reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows")
    print(df["GT Stance"].value_counts())
    
    return df