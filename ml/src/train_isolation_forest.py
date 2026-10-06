import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    MODEL_DIR,
    EVALUATION_RESULTS_DIR,
    RANDOM_STATE,
)
from ml_utils import (
    load_feature_splits,
    fit_scaler,
    transform_features,
    load_json,
    save_json,
    save_scaler,
)


def main():
    model_dir = (
        MODEL_DIR / "isolation_forest"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_feature_splits()

    scaler, X_train = fit_scaler(
        data["train"]
    )

    X_validation = transform_features(
        data["validation"],
        scaler,
    )

    X_test = transform_features(
        data["test"],
        scaler,
    )

    params = load_json(
        MODEL_DIR
        / "tuning"
        / "isolation_forest_best_params.json",
        default={},
    )

    n_estimators = int(
        params.get(
            "n_estimators",
            300,
        )
    )

    max_samples = float(
        params.get(
            "max_samples",
            1.0,
        )
    )

    max_features = float(
        params.get(
            "max_features",
            1.0,
        )
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    validation_scores = (
        -model.score_samples(
            X_validation
        )
    )

    # No true anomaly labels exist.
    # Use the upper 1% of validation scores
    # as the initial operational threshold.
    threshold = float(
        np.quantile(
            validation_scores,
            0.99,
        )
    )

    test_scores = (
        -model.score_samples(
            X_test
        )
    )

    test_predictions = (
        test_scores > threshold
    ).astype(int)

    joblib.dump(
        model,
        model_dir / "model.joblib",
    )

    save_scaler(
        scaler,
        model_dir / "scaler.joblib",
    )

    save_json({
        "model": "isolation_forest", "threshold": threshold,
        "n_estimators": n_estimators, "max_samples": max_samples,
        "max_features": max_features, "features": list(scaler.feature_names_in_),
        "threshold_quantile": 0.99,
    }, model_dir / "metadata.json")
    pd.DataFrame({
        "timestamp": data["test"]["timestamp"],
        "score": test_scores, "is_anomaly": test_predictions,
    }).to_csv(EVALUATION_RESULTS_DIR / "isolation_forest_test_results.csv", index=False)
    print(f"Isolation forest trained; threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
