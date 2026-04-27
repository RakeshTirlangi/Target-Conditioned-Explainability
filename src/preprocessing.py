# preprocessing.py

import ast
import pandas as pd

def format_input(tweet: str, target: str) -> str:
    """Combine tweet and target into model input."""
    return f"{tweet} [SEP] Target: {target}"


def parse_similar_targets(cell_value, n: int = 3) -> list:
    """Parse list of similar targets safely."""
    
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return []
    
    raw = str(cell_value).strip()
    
    # Try parsing Python list format
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()][:n]
    except:
        pass
    
    # Fallback: comma-separated string
    return [x.strip() for x in raw.split(",") if x.strip()][:n]