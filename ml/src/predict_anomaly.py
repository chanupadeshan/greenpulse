# Process live IoT sensor readings and detect anomalies using a trained ML model.
# Return the anomaly score, threshold, and severity level.
import argparse
import json

import numpy as np
import pandas as pd

from config import BASE_FEATURES, MODEL_DIR, MODEL_FEATURES
from feature_engineering import add_features
from ml_utils import load_json, reconstruction_errors
from model_io import MODEL_NAMES, load_artifacts



def severity_from_score(score, threshold):
    """
    Return a severity level for the anomaly score relative to the saved threshold.
    """

    if score <= threshold:
        return "normal"
    return "moderate" if score / max(threshold, 1e-12) < 1.5 else "high"


def load_live_csv(path):
    """
    Load a CSV of live sensor readings and validate its timestamps and numeric values.
    """

    df = pd.read_csv(path)
    required = ["timestamp"] + BASE_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing live columns: {missing}")
    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df.empty or df["timestamp"].isna().any() or not np.isfinite(df[BASE_FEATURES].to_numpy()).all():
        raise ValueError("Live data must contain valid timestamps and finite numeric sensor readings")
    if df["timestamp"].duplicated().any():
        raise ValueError("Live data contains duplicate timestamps")
    return add_features(df.sort_values("timestamp").reset_index(drop=True))


def predict(df, name):
    """
    Return the anomaly score and threshold for the latest sensor reading(s) using the specified model.(name:model name)
    """

    model, scaler, metadata = load_artifacts(name)
    # Apply the training scaler in the saved feature order; do not fit it again on live readings.
    X = scaler.transform(df[MODEL_FEATURES])
    if name == "isolation_forest":
        # Isolation Forest gives lower scores to unusual readings; negate them so higher means more unusual.
        score = float(-model.score_samples(X[-1:])[0])
    else:
        if name == "lstm_autoencoder":
            length = metadata["sequence_length"]
            if len(X) < length:
                raise ValueError(f"LSTM requires at least {length} hourly readings")
            # Require an unbroken hourly window; a gap must not be treated as the next consecutive hour.
            window = df["timestamp"].iloc[-length:]
            if not window.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
                raise ValueError("LSTM requires consecutive hourly readings")
            # Add the batch dimension: one sequence containing the latest required hours.
            X = X[-length:][None, :, :]
        else:
            # Dense autoencoder expects a 2D array of shape (samples, features).
            X = X[-1:]
        # Measure how far the reconstructed values differ from the actual scaled readings.
        score = float(reconstruction_errors(model, X)[0])
    # The caller marks an anomaly only when the score exceeds this training-time cutoff.
    return score, metadata["threshold"]


# Convenience wrappers let other scripts request a specific model without repeating its name.
def predict_isolation_forest(df):
    return predict(df, "isolation_forest")


def predict_autoencoder(df):
    return predict(df, "autoencoder")


def predict_lstm_autoencoder(df):
    return predict(df, "lstm_autoencoder")


def main():
    parser = argparse.ArgumentParser(description="Score the latest sensor reading")
    parser.add_argument("csv", help="CSV with timestamp and six base sensor columns")
    parser.add_argument("--model", choices=MODEL_NAMES)
    args = parser.parse_args()
    # Command-line selection: --model wins, then selected_model.json, then Isolation Forest.
    # The AI-agent platform supplies its configured model directly to predict().
    selected = load_json(MODEL_DIR / "selected_model.json", default={})
    name = args.model or selected.get("selected_model", "isolation_forest")
    df = load_live_csv(args.csv)
    score, threshold = predict(df, name)
    print(json.dumps({
        "model": name, "timestamp": df["timestamp"].iloc[-1].isoformat(),
        "score": score, "threshold": threshold,
        "is_anomaly": score > threshold,
        "severity": severity_from_score(score, threshold),
    }, indent=2))


if __name__ == "__main__":
    main()
