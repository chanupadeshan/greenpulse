import argparse
import numpy as np
import pandas as pd
import torch

from config import (
    MODEL_DIR,
    EVALUATION_RESULTS_DIR,
)
from ml_utils import (
    load_feature_splits,
    fit_scaler,
    transform_features,
    load_json,
    save_json,
    save_scaler,
    train_autoencoder,
    reconstruction_errors,
    set_seed,
)
from models import DenseAutoencoder


def main(epochs=None):
    set_seed()

    model_dir = (
        MODEL_DIR / "autoencoder"
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
        / "autoencoder_best_params.json",
        default={},
    )

    hidden_dim = int(
        params.get(
            "hidden_dim",
            32,
        )
    )

    latent_dim = int(
        params.get(
            "latent_dim",
            8,
        )
    )

    dropout = float(
        params.get(
            "dropout",
            0.1,
        )
    )

    learning_rate = float(
        params.get(
            "learning_rate",
            1e-3,
        )
    )

    model = DenseAutoencoder(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    model, best_validation_loss = (
        train_autoencoder(
            model,
            X_train,
            X_validation,
            epochs=epochs or 80,
            batch_size=64,
            learning_rate=learning_rate,
            patience=10,
        )
    )

    validation_errors = (
        reconstruction_errors(
            model,
            X_validation,
        )
    )

    threshold = float(
        np.quantile(
            validation_errors,
            0.99,
        )
    )

    test_errors = (
        reconstruction_errors(
            model,
            X_test,
        )
    )
    torch.save(model.cpu().state_dict(), model_dir / "model.pt")
    save_scaler(scaler, model_dir / "scaler.joblib")
    metadata = {
        "model": "autoencoder", "input_dim": X_train.shape[1],
        "hidden_dim": hidden_dim, "latent_dim": latent_dim,
        "learning_rate": learning_rate, "threshold": threshold,
        "best_validation_loss": best_validation_loss, "max_epochs": epochs or 80,
        "features": list(scaler.feature_names_in_), "threshold_quantile": 0.99,
    }
    metadata["dropout"] = dropout
    save_json(metadata, model_dir / "metadata.json")
    pd.DataFrame({
        "timestamp": data["test"]["timestamp"],
        "score": test_errors, "is_anomaly": (test_errors > threshold).astype(int),
    }).to_csv(EVALUATION_RESULTS_DIR / "autoencoder_test_results.csv", index=False)
    print(f"autoencoder trained; threshold={threshold:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    main(args.epochs)
