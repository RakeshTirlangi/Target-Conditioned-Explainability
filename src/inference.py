# inference.py

import torch
import numpy as np
from src.config import DEVICE, MAX_LEN

def predict_proba(texts, tokenizer, model):
    """Return softmax probabilities."""
    
    # Clean invalid inputs
    texts = [t if isinstance(t, str) and t.strip() else "neutral" for t in texts]
    
    # Tokenize
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    # Forward pass
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    
    return probs.cpu().numpy()