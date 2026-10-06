import argparse
import json

import numpy as np
import pandas as pd

from config import BASE_FEATURES, MODEL_DIR, MODEL_FEATURES
from feature_engineering import add_features
from ml_utils import load_json, reconstruction_errors
from model_io import MODEL_NAMES, load_artifacts


def severity_from_score(score, threshold):
    if score <= threshold:
        return "normal"
    return "moderate" if score / max(threshold, 1e-12) < 1.5 else "high"


def load_live_csv(path):
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
    model, scaler, metadata = load_artifacts(name)
    X = scaler.transform(df[MODEL_FEATURES])
    if name == "isolation_forest":
        score = float(-model.score_samples(X[-1:])[0])
    else:
        if name == "lstm_autoencoder":
            length = metadata["sequence_length"]
            if len(X) < length:
                raise ValueError(f"LSTM requires at least {length} hourly readings")
            window = df["timestamp"].iloc[-length:]
            if not window.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
                raise ValueError("LSTM requires consecutive hourly readings")
            X = X[-length:][None, :, :]
        else:
            X = X[-1:]
        score = float(reconstruction_errors(model, X)[0])
    return score, metadata["threshold"]


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
