import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest

import model_io
from config import BASE_FEATURES, MODEL_FEATURES, COLUMN_MAP
from feature_engineering import add_features
from ml_utils import (create_sequences, fit_scaler, inject_tabular_anomalies,
                      reconstruction_errors, save_json, sequence_end_indices)
from models import DenseAutoencoder, LSTMAutoencoder
from predict_anomaly import load_live_csv, predict
from preprocess import clean_data, chronological_split


def readings(n=48):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({column: rng.uniform(10, 30, n) for column in BASE_FEATURES})
    frame.insert(0, "timestamp", pd.date_range("2025-01-01", periods=n, freq="h"))
    return frame


class PipelineTests(unittest.TestCase):
    def test_cleaning_and_chronological_split(self):
        frame = readings()
        frame.loc[0, "humidity"] = 101
        frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
        raw = frame.rename(columns={value: key for key, value in COLUMN_MAP.items()})
        cleaned = clean_data(raw)
        self.assertEqual(len(cleaned), 47)
        splits = chronological_split(cleaned)
        self.assertEqual(sum(map(len, splits.values())), len(cleaned))
        self.assertLess(splits["train"].timestamp.max(), splits["validation"].timestamp.min())
        self.assertLess(splits["validation"].timestamp.max(), splits["test"].timestamp.min())

    def test_features_and_scaling_do_not_use_future_data(self):
        frame = readings()
        engineered = add_features(frame)
        pd.testing.assert_frame_equal(engineered.iloc[:10], add_features(frame.iloc[:10]))
        scaler, transformed = fit_scaler(engineered.iloc[:30])
        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
        original_mean = scaler.mean_.copy()
        scaler.transform(engineered.iloc[30:][MODEL_FEATURES])
        np.testing.assert_array_equal(scaler.mean_, original_mean)

    def test_sequences_exclude_gaps(self):
        times = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00",
                                "2025-01-01 05:00", "2025-01-01 06:00"])
        X = np.arange(8).reshape(4, 2)
        result = create_sequences(X, 2, times)
        self.assertEqual(result.shape, (2, 2, 2))
        np.testing.assert_array_equal(sequence_end_indices(4, 2, times), [1, 3])
        with self.assertRaises(ValueError):
            create_sequences(X, 3, times)
        with self.assertRaises(ValueError):
            create_sequences(X, 0)

    def test_synthetic_anomalies_are_separate_and_reproducible(self):
        X = np.zeros((10, 3))
        challenge, labels = inject_tabular_anomalies(X)
        self.assertEqual(challenge.shape, (15, 3))
        np.testing.assert_array_equal(X, np.zeros_like(X))
        np.testing.assert_array_equal(challenge[:10], X)
        self.assertEqual(labels.sum(), 5)
        np.testing.assert_array_equal(challenge, inject_tabular_anomalies(X)[0])

    def test_saved_artifacts_produce_same_predictions(self):
        frame = add_features(readings())
        scaler, X = fit_scaler(frame)
        with tempfile.TemporaryDirectory() as tmp, patch.object(model_io, "MODEL_DIR", Path(tmp)):
            for name in model_io.MODEL_NAMES:
                directory = Path(tmp) / name
                directory.mkdir()
                metadata = {"threshold": 1.0, "features": MODEL_FEATURES,
                            "input_dim": len(MODEL_FEATURES), "hidden_dim": 8,
                            "latent_dim": 3, "dropout": 0.0, "sequence_length": 4}
                joblib.dump(scaler, directory / "scaler.joblib")
                if name == "isolation_forest":
                    model = IsolationForest(n_estimators=10, random_state=42).fit(X)
                    joblib.dump(model, directory / "model.joblib")
                    expected = -model.score_samples(X[-1:])[0]
                else:
                    model = (LSTMAutoencoder(len(MODEL_FEATURES), 8, 3) if name == "lstm_autoencoder"
                             else DenseAutoencoder(len(MODEL_FEATURES), 8, 3, 0.0))
                    torch.save(model.state_dict(), directory / "model.pt")
                    expected = reconstruction_errors(model, X[-4:][None] if name == "lstm_autoencoder" else X[-1:])[0]
                save_json(metadata, directory / "metadata.json")
                score, threshold = predict(frame, name)
                self.assertAlmostEqual(score, float(expected), places=6)
                self.assertEqual(threshold, 1.0)

    def test_live_input_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.csv"
            frame = readings()
            frame.loc[2, "humidity"] = np.inf
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_live_csv(path)
            readings().drop(columns="timestamp").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing live columns"):
                load_live_csv(path)


if __name__ == "__main__":
    unittest.main()
