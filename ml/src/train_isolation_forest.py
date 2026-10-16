# Train an Isolation Forest to find unusual sensor readings.
# Save the trained model, scaler, settings, and test results.

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import MODEL_DIR, EVALUATION_RESULTS_DIR, RANDOM_STATE
from ml_utils import load_feature_splits, fit_scaler, transform_features, load_json, save_json, save_scaler


# Train the model and save the files used for live predictions.
def main():
    model_dir = MODEL_DIR / "isolation_forest"
    # Create folders for the saved model and test results.
    model_dir.mkdir(parents=True, exist_ok=True)
    EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Load the training, validation, and test data in time order.
    data = load_feature_splits()
    # Learn how to scale the training data, then use the same scaler for the other data.
    scaler, X_train = fit_scaler(data["train"])
    X_validation = transform_features(data["validation"], scaler)
    X_test = transform_features(data["test"], scaler)
    # Load the best settings if saved, or use the default values below.
    params = load_json(MODEL_DIR / "tuning" / "isolation_forest_best_params.json", default={})
    n_estimators = int(params.get("n_estimators", 300))
    max_samples = float(params.get("max_samples", 1.0))
    max_features = float(params.get("max_features", 1.0))
    # Build the model with the chosen settings.
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Train the forest using only the training data.
    model.fit(X_train)
    # Reverse the score sign so higher scores mean more unusual readings.
    validation_scores = -model.score_samples(X_validation)
    # The data has no labels showing which readings are unusual.
    # Set the alert limit so about 1% of validation scores are above it.
    threshold = float(np.quantile(validation_scores, 0.99))
    # Score the test readings and flag scores above the alert limit.
    test_scores = -model.score_samples(X_test)
    test_predictions = (test_scores > threshold).astype(int)
    # Save the model, scaler, and alert limit for live predictions.
    joblib.dump(model, model_dir / "model.joblib")
    save_scaler(scaler, model_dir / "scaler.joblib")
    save_json({
        "model": "isolation_forest", "threshold": threshold,
        "n_estimators": n_estimators, "max_samples": max_samples,
        "max_features": max_features, "features": list(scaler.feature_names_in_),
        "threshold_quantile": 0.99,
    }, model_dir / "metadata.json")
    # Save each test score and alert flag with its time.
    # An alert means the reading looks unusual; it may still be normal.
    pd.DataFrame({
        "timestamp": data["test"]["timestamp"],
        "score": test_scores, "is_anomaly": test_predictions,
    }).to_csv(EVALUATION_RESULTS_DIR / "isolation_forest_test_results.csv", index=False)
    print(f"Isolation forest trained; threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
