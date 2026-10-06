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

from config import (
    FEATURE_DIR,
    MODEL_FEATURES,
    RANDOM_STATE,
)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


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
    if not (data["train"]["timestamp"].max() < data["validation"]["timestamp"].min()
            and data["validation"]["timestamp"].max() < data["test"]["timestamp"].min()):
        raise ValueError("Splits must be chronological and must not overlap")
    return data


def fit_scaler(train_df):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[MODEL_FEATURES]
    )

    return scaler, X_train


def transform_features(df, scaler):
    return scaler.transform(
        df[MODEL_FEATURES]
    )


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
        cumulative = np.cumsum(breaks)
        starts = ends - sequence_length + 1
        ends = ends[(cumulative[ends] - cumulative[starts]) == 0]
    if len(ends) == 0:
        raise ValueError("No consecutive hourly windows available")
    return ends


def create_sequences(X, sequence_length, timestamps=None):
    ends = sequence_end_indices(len(X), sequence_length, timestamps)
    return np.stack([X[end - sequence_length + 1:end + 1] for end in ends])


def _validate_injection(X, fraction, magnitude, dimensions):
    if X.ndim != dimensions or X.shape[0] == 0 or X.shape[-1] == 0:
        raise ValueError("Anomaly injection requires a nonempty array of the expected dimensions")
    if not 0 < fraction <= 1 or not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Fraction must be in (0, 1] and magnitude must be finite and positive")


def inject_tabular_anomalies(
    X,
    fraction=0.5,
    magnitude=3.0,
    random_state=RANDOM_STATE,
):
    """
    Synthetic anomalies are used only for controlled
    tuning/evaluation because the dataset has no anomaly labels.
    """
    _validate_injection(X, fraction, magnitude, 2)
    rng = np.random.default_rng(
        random_state
    )

    count = max(
        1,
        int(len(X) * fraction)
    )

    indices = rng.choice(
        len(X),
        size=count,
        replace=False,
    )

    anomaly = X[indices].copy()

    for row in range(len(anomaly)):
        feature_count = int(
            rng.integers(
                1,
                min(4, X.shape[1] + 1),
            )
        )

        features = rng.choice(
            X.shape[1],
            size=feature_count,
            replace=False,
        )

        direction = rng.choice(
            [-1.0, 1.0],
            size=feature_count,
        )

        anomaly[row, features] += (
            direction
            * magnitude
            * rng.uniform(
                0.8,
                1.3,
                size=feature_count,
            )
        )

    combined = np.vstack([
        X,
        anomaly,
    ])

    labels = np.concatenate([
        np.zeros(len(X), dtype=int),
        np.ones(len(anomaly), dtype=int),
    ])

    return combined, labels


def inject_sequence_anomalies(
    sequences,
    fraction=0.5,
    magnitude=3.0,
    random_state=RANDOM_STATE,
):
    _validate_injection(sequences, fraction, magnitude, 3)
    rng = np.random.default_rng(
        random_state
    )

    count = max(
        1,
        int(len(sequences) * fraction)
    )

    indices = rng.choice(
        len(sequences),
        size=count,
        replace=False,
    )

    anomaly = sequences[indices].copy()

    for i in range(len(anomaly)):
        time_index = int(
            rng.integers(
                0,
                anomaly.shape[1],
            )
        )

        feature_count = int(
            rng.integers(
                1,
                min(4, anomaly.shape[2] + 1),
            )
        )

        features = rng.choice(
            anomaly.shape[2],
            size=feature_count,
            replace=False,
        )

        direction = rng.choice(
            [-1.0, 1.0],
            size=feature_count,
        )

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

    combined = np.concatenate([
        sequences,
        anomaly,
    ])

    labels = np.concatenate([
        np.zeros(
            len(sequences),
            dtype=int,
        ),
        np.ones(
            len(anomaly),
            dtype=int,
        ),
    ])

    return combined, labels


def train_autoencoder(
    model,
    X_train,
    X_validation,
    epochs=60,
    batch_size=64,
    learning_rate=1e-3,
    patience=8,
):
    set_seed()

    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.MSELoss()

    train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    validation_tensor = torch.tensor(
        X_validation,
        dtype=torch.float32,
        device=device,
    )

    loader = DataLoader(
        TensorDataset(
            train_tensor,
            train_tensor,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_loss = float("inf")
    best_state = None
    waiting = 0

    for epoch in range(epochs):
        model.train()

        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(batch)
            loss = loss_function(
                output,
                target,
            )

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            prediction = model(
                validation_tensor
            )

            validation_loss = (
                loss_function(
                    prediction,
                    validation_tensor,
                )
                .item()
            )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1

        if waiting >= patience:
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model, best_loss


def reconstruction_errors(
    model,
    X,
    batch_size=512,
):
    device = get_device()
    model = model.to(device)
    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    loader = DataLoader(
        tensor,
        batch_size=batch_size,
        shuffle=False,
    )

    errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            reconstruction = model(
                batch
            )

            dims = tuple(
                range(
                    1,
                    batch.ndim,
                )
            )

            error = (
                (batch - reconstruction)
                .pow(2)
                .mean(dim=dims)
            )

            errors.extend(
                error.cpu().numpy()
            )

    return np.asarray(errors)


def save_json(data, path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def load_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_scaler(scaler, path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        scaler,
        path,
    )
