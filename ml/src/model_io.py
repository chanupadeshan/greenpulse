# Load trained models, scalers, and metadata for evaluation and live anomaly detection.

import joblib
import torch

from config import MODEL_DIR, MODEL_FEATURES
from ml_utils import load_json
from models import DenseAutoencoder, LSTMAutoencoder

MODEL_NAMES = "isolation_forest", "autoencoder", "lstm_autoencoder"


def load_artifacts(name):
    """
    Load a trained model, its scaler, and its metadata one at a time.
    """

    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    directory = MODEL_DIR / name
    metadata = load_json(directory / "metadata.json")
    if not metadata or "threshold" not in metadata:
        raise ValueError(f"No trained metadata for {name}; run train_{name}.py first")
    # Reject mismatched feature names or ordering rather than score readings with incompatible inputs.
    if metadata.get("features") != MODEL_FEATURES:
        raise ValueError(f"Feature schema mismatch for {name}; retrain the model")
    # Reuse training-time scaling; fitting a new scaler would change what the saved model sees.
    scaler = joblib.load(directory / "scaler.joblib")
    if name == "isolation_forest":
        model = joblib.load(directory / "model.joblib")
    else:
        # PyTorch saves weights separately from the architecture, so rebuild the matching network first.
        cls = LSTMAutoencoder if name == "lstm_autoencoder" else DenseAutoencoder
        params = {key: metadata[key] for key in ("input_dim", "hidden_dim", "latent_dim")}
        if name == "autoencoder":
            params["dropout"] = metadata["dropout"]
        model = cls(**params)
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        # Disable training behavior such as dropout before making predictions.
        model.eval()
    return model, scaler, metadata
