import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                            recall_score, roc_auc_score)

from config import EVALUATION_RESULTS_DIR
from ml_utils import (load_feature_splits, transform_features, create_sequences,
                      inject_tabular_anomalies, inject_sequence_anomalies,
                      reconstruction_errors)
from model_io import MODEL_NAMES, load_artifacts


def classification_metrics(labels, scores, threshold):
    predictions = (scores > threshold).astype(int)
    return {
        "synthetic_auc": roc_auc_score(labels, scores),
        "synthetic_average_precision": average_precision_score(labels, scores),
        "synthetic_precision": precision_score(labels, predictions, zero_division=0),
        "synthetic_recall": recall_score(labels, predictions, zero_division=0),
        "synthetic_f1": f1_score(labels, predictions, zero_division=0),
    }


def evaluate_model(data, name):
    model, scaler, metadata = load_artifacts(name)
    X = transform_features(data["test"], scaler)
    if name == "lstm_autoencoder":
        X = create_sequences(X, metadata["sequence_length"], data["test"]["timestamp"])
        challenge, labels = inject_sequence_anomalies(X)
    else:
        challenge, labels = inject_tabular_anomalies(X)
    if name == "isolation_forest":
        real_scores = -model.score_samples(X)
        scores = -model.score_samples(challenge)
    else:
        real_scores = reconstruction_errors(model, X)
        scores = reconstruction_errors(model, challenge)
    metrics = classification_metrics(labels, scores, metadata["threshold"])
    metrics.update({
        "model": name,
        "real_test_anomaly_rate": float(np.mean(real_scores > metadata["threshold"])),
        "test_samples": len(X), "threshold": metadata["threshold"],
    })
    return metrics


def evaluate_isolation_forest(data):
    return evaluate_model(data, "isolation_forest")


def main():
    data = load_feature_splits()
    result = pd.DataFrame([evaluate_model(data, name) for name in MODEL_NAMES])
    EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVALUATION_RESULTS_DIR / "model_comparison.csv", index=False)
    print(result.to_string(index=False))
    print("Metrics use synthetic anomalies; they do not establish real-world accuracy.")


if __name__ == "__main__":
    main()
