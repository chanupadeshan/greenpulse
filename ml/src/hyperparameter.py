import argparse
from itertools import product

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from config import (
    MODEL_DIR,
    TUNING_RESULTS_DIR,
    RANDOM_STATE,
)
from ml_utils import (
    load_feature_splits,
    fit_scaler,
    transform_features,
    inject_tabular_anomalies,
    create_sequences,
    inject_sequence_anomalies,
    train_autoencoder,
    reconstruction_errors,
    save_json,
    set_seed,
)
from models import (
    DenseAutoencoder,
    LSTMAutoencoder,
)


def tune_isolation_forest(
    X_train,
    X_validation,
    quick=False,
):
    if quick:
        candidates = [
            (200, 0.75, 1.0),
            (300, 1.0, 1.0),
        ]
    else:
        candidates = list(product(
            [200, 400],
            [0.75, 1.0],
            [0.75, 1.0],
        ))

    X_challenge, y = (
        inject_tabular_anomalies(
            X_validation,
            fraction=0.50,
            magnitude=3.0,
        )
    )

    rows = []

    for (
        n_estimators,
        max_samples,
        max_features,
    ) in candidates:

        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train)

        scores = -model.score_samples(
            X_challenge
        )

        auc = roc_auc_score(
            y,
            scores,
        )

        rows.append({
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "synthetic_auc": auc,
        })

    result = pd.DataFrame(
        rows
    ).sort_values(
        "synthetic_auc",
        ascending=False,
    )

    best = result.iloc[0].to_dict()

    return result, best


def tune_neural(X_train, X_validation, candidates, lstm=False, quick=False,
                train_timestamps=None, validation_timestamps=None):
    rows = []
    for params in candidates:
        set_seed()
        if lstm:
            train = create_sequences(X_train, params["sequence_length"], train_timestamps)
            validation = create_sequences(X_validation, params["sequence_length"], validation_timestamps)
            challenge, labels = inject_sequence_anomalies(validation)
            model = LSTMAutoencoder(X_train.shape[1], params["hidden_dim"], params["latent_dim"])
        else:
            train, validation = X_train, X_validation
            challenge, labels = inject_tabular_anomalies(validation)
            model = DenseAutoencoder(X_train.shape[1], params["hidden_dim"], params["latent_dim"], params["dropout"])
        model, loss = train_autoencoder(
            model, train, validation, epochs=3 if quick else 60,
            learning_rate=params["learning_rate"], patience=8,
        )
        auc = roc_auc_score(labels, reconstruction_errors(model, challenge))
        rows.append({**params, "synthetic_auc": float(auc), "validation_loss": loss})
    result = pd.DataFrame(rows).sort_values("synthetic_auc", ascending=False)
    return result, result.iloc[0].to_dict()


def tune_dense_autoencoder(X_train, X_validation, quick=False):
    candidates = [
        dict(hidden_dim=32, latent_dim=8, dropout=0.1, learning_rate=1e-3),
        dict(hidden_dim=64, latent_dim=16, dropout=0.1, learning_rate=5e-4),
    ]
    if not quick:
        candidates += [dict(hidden_dim=32, latent_dim=8, dropout=0.0, learning_rate=1e-3)]
    return tune_neural(X_train, X_validation, candidates, quick=quick)


def tune_lstm_autoencoder(X_train, X_validation, quick=False,
                          train_timestamps=None, validation_timestamps=None):
    candidates = [
        dict(sequence_length=length, hidden_dim=32 if quick else 64,
             latent_dim=16, learning_rate=1e-3)
        for length in ([12, 24] if quick else [12, 24, 48])
    ]
    return tune_neural(X_train, X_validation, candidates, lstm=True, quick=quick,
                      train_timestamps=train_timestamps, validation_timestamps=validation_timestamps)


def main():
    parser = argparse.ArgumentParser(description="Tune using validation data and synthetic anomalies")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    data = load_feature_splits()
    scaler, X_train = fit_scaler(data["train"])
    X_validation = transform_features(data["validation"], scaler)
    functions = {
        "isolation_forest": lambda: tune_isolation_forest(X_train, X_validation, args.quick),
        "autoencoder": lambda: tune_dense_autoencoder(X_train, X_validation, args.quick),
        "lstm_autoencoder": lambda: tune_lstm_autoencoder(
            X_train, X_validation, args.quick,
            data["train"]["timestamp"], data["validation"]["timestamp"]),
    }
    TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, tune in functions.items():
        result, best = tune()
        result.to_csv(TUNING_RESULTS_DIR / f"{name}_tuning.csv", index=False)
        save_json(best, MODEL_DIR / "tuning" / f"{name}_best_params.json")
        print(f"{name}: {best}")


if __name__ == "__main__":
    main()
