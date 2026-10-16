# Shared helpers for scaling, hourly sequences, neural training, scoring, and artifact storage.
# Also generates synthetic anomalies for controlled tuning and evaluation.

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURE_DIR, MODEL_FEATURES, RANDOM_STATE


# Seed Python, NumPy, and PyTorch random generators so repeated experiments are more comparable.
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Use a CUDA GPU when available; otherwise run the same calculations on the CPU.
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the three feature CSVs and reject missing values, invalid times, or overlapping time periods.
def load_feature_splits():
    data = {}
    for split in ("train", "validation", "test"):
        path = FEATURE_DIR / f"{split}_engineered.csv"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing engineered data: {path}; run preprocessing and feature engineering first")
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        missing = [column for column in MODEL_FEATURES if column not in frame.columns]
        if missing or frame.empty:
            raise ValueError(f"Invalid {split} split: empty data or missing features {missing}")
        if not np.isfinite(frame[MODEL_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite feature values in {split} split")
        times = frame["timestamp"]
        if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError(f"{split} timestamps must be valid, unique and sorted")
        data[split] = frame
    # Ensure all training readings precede validation, and all validation readings precede testing.
    if not (data["train"]["timestamp"].max() < data["validation"]["timestamp"].min()
            and data["validation"]["timestamp"].max() < data["test"]["timestamp"].min()):
        raise ValueError("Splits must be chronological and must not overlap")
    return data


# Learn each feature's average and spread from training data only, then standardize that data.
# Return both the fitted scaler and the scaled training matrix.
def fit_scaler(train_df):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[MODEL_FEATURES])

    return scaler, X_train


# Scale validation, test, or live data using the training averages and spreads without relearning them.
def transform_features(df, scaler):
    return scaler.transform(df[MODEL_FEATURES])


# Find the last row of every complete window, excluding windows that cross gaps in hourly readings.
def sequence_end_indices(row_count, sequence_length, timestamps=None):
    if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 1:
        raise ValueError("Sequence length must be a positive integer")
    if row_count < sequence_length:
        raise ValueError(f"Not enough rows for sequence length {sequence_length}")
    ends = np.arange(sequence_length - 1, row_count)
    if timestamps is not None:
        times = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
        if len(times) != row_count or times.isna().any():
            raise ValueError("Timestamps must match rows and contain no missing values")
        breaks = times.diff().ne(pd.Timedelta(hours=1)).to_numpy()
        # Count time breaks up to each row. Equal counts at a window's endpoints mean no internal gap.
        cumulative = np.cumsum(breaks)
        starts = ends - sequence_length + 1
        ends = ends[(cumulative[ends] - cumulative[starts]) == 0]
    if len(ends) == 0:
        raise ValueError("No consecutive hourly windows available")
    return ends


# Turn a 2D table (rows, features) into a 3D array (windows, hours, features).
def create_sequences(X, sequence_length, timestamps=None):
    ends = sequence_end_indices(len(X), sequence_length, timestamps)
    return np.stack([X[end - sequence_length + 1:end + 1] for end in ends])


# Reject empty arrays and invalid settings before creating artificial anomalies.
def _validate_injection(X, fraction, magnitude, dimensions):
    if X.ndim != dimensions or X.shape[0] == 0 or X.shape[-1] == 0:
        raise ValueError("Anomaly injection requires a nonempty array of the expected dimensions")
    if not 0 < fraction <= 1 or not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Fraction must be in (0, 1] and magnitude must be finite and positive")


# Copy selected normal rows and disturb a few scaled features; the original rows stay unchanged.
# Return normal rows plus disturbed copies, with labels 0=assumed normal and 1=artificial anomaly.
def inject_tabular_anomalies(X, fraction=0.5, magnitude=3.0, random_state=RANDOM_STATE):
    """
    Synthetic anomalies are used only for controlled
    tuning/evaluation because the dataset has no anomaly labels.
    """
    _validate_injection(X, fraction, magnitude, 2)
    rng = np.random.default_rng(random_state)

    count = max(1, int(len(X) * fraction))

    indices = rng.choice(len(X), size=count, replace=False)

    anomaly = X[indices].copy()

    for row in range(len(anomaly)):
        feature_count = int(rng.integers(1, min(4, X.shape[1] + 1)))

        features = rng.choice(X.shape[1], size=feature_count, replace=False)

        direction = rng.choice([-1.0, 1.0], size=feature_count)

        anomaly[row, features] += (direction * magnitude * rng.uniform(0.8, 1.3, size=feature_count))

    combined = np.vstack([X, anomaly])

    labels = np.concatenate([np.zeros(len(X), dtype=int), np.ones(len(anomaly), dtype=int)])

    return combined, labels


# Create artificial anomalies by disturbing selected features at one hour inside copied sequences.
# Return the original and disturbed sequences with matching 0/1 labels.
def inject_sequence_anomalies(sequences, fraction=0.5, magnitude=3.0, random_state=RANDOM_STATE):
    _validate_injection(sequences, fraction, magnitude, 3)
    rng = np.random.default_rng(random_state)

    count = max(1, int(len(sequences) * fraction))

    indices = rng.choice(len(sequences), size=count, replace=False)

    anomaly = sequences[indices].copy()

    for i in range(len(anomaly)):
        time_index = int(rng.integers(0, anomaly.shape[1]))

        feature_count = int(rng.integers(1, min(4, anomaly.shape[2] + 1)))

        features = rng.choice(anomaly.shape[2], size=feature_count, replace=False)

        direction = rng.choice([-1.0, 1.0], size=feature_count)

        anomaly[
            i,
            time_index,
            features,
        ] += (
            direction
            * magnitude
            * rng.uniform(
                0.8,
                1.3,
                size=feature_count,
            )
        )

    combined = np.concatenate([sequences, anomaly])

    labels = np.concatenate([np.zeros(len(sequences), dtype=int), np.ones(len(anomaly), dtype=int)])

    return combined, labels


# Train either neural network to reconstruct its own inputs; no anomaly labels are needed.
# Use validation reconstruction error to choose the best weights and stop unproductive training.
def train_autoencoder(model, X_train, X_validation, epochs=60, batch_size=64, learning_rate=1e-3, patience=8):
    set_seed()

    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Mean squared error measures the average squared difference between input and reconstructed values.
    loss_function = nn.MSELoss()

    train_tensor = torch.tensor(X_train, dtype=torch.float32)

    validation_tensor = torch.tensor(X_validation, dtype=torch.float32, device=device)

    # For an autoencoder, the desired output is the input itself: each batch is also its own target.
    loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = None
    waiting = 0

    for epoch in range(epochs):
        model.train()

        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)

            # Clear old gradients, calculate reconstruction error, then update weights to reduce that error.
            optimizer.zero_grad()

            output = model(batch)
            loss = loss_function(output, target)

            loss.backward()
            optimizer.step()

        model.eval()

        # Check validation performance without changing weights or recording gradients.
        with torch.no_grad():
            prediction = model(validation_tensor)

            validation_loss = loss_function(prediction, validation_tensor).item()

        if validation_loss < best_loss:
            best_loss = validation_loss
            # Keep an independent copy of the best weights so later training cannot overwrite them.
            best_state = { k: v.detach().cpu().clone() for k, v in model.state_dict().items() }
            waiting = 0
        else:
            waiting += 1

        # Stop after this many epochs without improvement, then restore the best saved weights.
        if waiting >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss


# Return one reconstruction-error score per row or sequence, in the original input order.
def reconstruction_errors(model, X, batch_size=512):
    device = get_device()
    model = model.to(device)
    model.eval()

    tensor = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(tensor, batch_size=batch_size, shuffle=False)

    errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            reconstruction = model(batch)

            # Average over all non-batch dimensions: features for rows, or hours and features for sequences.
            dims = tuple(range(1, batch.ndim))

            error = (batch - reconstruction).pow(2).mean(dim=dims)

            errors.extend(error.cpu().numpy())

    return np.asarray(errors)


# Save readable settings or metadata, creating the destination directory if necessary.
def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# Read saved settings; return the supplied default if the file does not exist yet.
def load_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Persist the fitted scaler so prediction uses exactly the same scaling as training.
def save_scaler(scaler, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
