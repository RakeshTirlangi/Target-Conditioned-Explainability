# main.py

from src.config import DATA_PATH, OUTPUT_PATH
from src.data_loader import load_data
from src.model import load_model
from src.shap_explainer import build_explainer
from src.pipeline import run_pipeline

def main():
    """Entry point of the project."""
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Load model
    tokenizer, model = load_model("deberta-v3")
    
    # Dummy background (you can improve this)
    background_text = ["This is a neutral tweet"]
    
    # Build SHAP explainer
    explainer = build_explainer(tokenizer, model, background_text)
    
    # Run pipeline
    results_df = run_pipeline(df, tokenizer, model, explainer)
    
    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved results → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()