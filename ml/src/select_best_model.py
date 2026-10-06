import pandas as pd

from config import (
    MODEL_DIR,
    EVALUATION_RESULTS_DIR,
)
from ml_utils import save_json


def main():
    comparison_path = (
        EVALUATION_RESULTS_DIR
        / "model_comparison.csv"
    )

    comparison = pd.read_csv(
        comparison_path
    )

    # Primary criterion:
    # synthetic anomaly ROC-AUC.
    # Secondary criterion:
    # synthetic F1 score.
    ranked = comparison.sort_values(
        [
            "synthetic_auc",
            "synthetic_f1",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = ranked.iloc[0]

    result = {
        "selected_model":
            best["model"],
        "synthetic_auc":
            float(
                best["synthetic_auc"]
            ),
        "synthetic_f1":
            float(
                best["synthetic_f1"]
            ),
        "note": (
            "Selection uses controlled synthetic "
            "anomaly evaluation because the "
            "original dataset has no ground-truth "
            "anomaly labels."
        ),
    }

    save_json(
        result,
        MODEL_DIR
        / "selected_model.json",
    )

    print("Selected model:")
    print(result)


if __name__ == "__main__":
    main()
