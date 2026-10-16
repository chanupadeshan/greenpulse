# Choose the best model from the saved evaluation results.
# Save the choice as the default model for standalone predictions.

import pandas as pd

from config import MODEL_DIR, EVALUATION_RESULTS_DIR
from ml_utils import save_json


# Run evaluate_models.py first to create the model comparison file.
def main():
    comparison_path = EVALUATION_RESULTS_DIR / "model_comparison.csv"

    # Load the scores for all evaluated models.
    comparison = pd.read_csv(comparison_path)

    # Put the highest synthetic ROC-AUC score first.
    # If scores are equal, use the highest synthetic F1 score.
    ranked = comparison.sort_values(["synthetic_auc", "synthetic_f1"], ascending=[False, False])

    # Choose the first model in the sorted list.
    best = ranked.iloc[0]

    # Keep the chosen model name, its scores, and the reason for this comparison.
    result = {
        "selected_model":
            best["model"],
        "synthetic_auc":
            float(best["synthetic_auc"]),
        "synthetic_f1":
            float(best["synthetic_f1"]),
        "note": (
            "Selection uses controlled synthetic "
            "anomaly evaluation because the "
            "original dataset has no ground-truth "
            "anomaly labels."
        ),
    }
    # Save the choice for standalone predictions.
    # The AI agent still uses its own ML_MODEL setting.
    save_json(result, MODEL_DIR / "selected_model.json")

    # Show the chosen model and its scores.
    print("Selected model:")
    print(result)


if __name__ == "__main__":
    main()
