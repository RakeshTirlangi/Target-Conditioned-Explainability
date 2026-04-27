# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import DEVICE

def load_model(model_name: str):
    """Load tokenizer and model."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to device and set eval mode
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model