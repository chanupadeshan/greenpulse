import sys
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

# The existing ML scripts use direct imports from ml/src. Keep a single bridge here
# so the agent layer reuses their feature engineering, scaler and saved weights.
ML_SOURCE = PROJECT_ROOT / "ml" / "src"
if str(ML_SOURCE) not in sys.path:
    sys.path.insert(0, str(ML_SOURCE))
from feature_engineering import add_features
from predict_anomaly import predict, severity_from_score


class MLClient:
    def __init__(self, settings):
        self.settings = settings

    def analyze(self, readings):
        from .dynamodb import SENSOR_FIELDS
        missing = [name for name in SENSOR_FIELDS if readings[-1].get(name) is None]
        if missing:
            return {"status": "blocked", "model": self.settings.ml_model, "missing_fields": missing,
                    "error": "Prediction needs all six measured inputs. Restore missing sensor values; no imputation is enabled."}
        frame = pd.DataFrame(readings)
        # Earlier incomplete records cannot enter model scaling or inference.
        frame = frame.dropna(subset=list(SENSOR_FIELDS)).copy()
        # Hour-of-day features must use local sensor time, not UTC hour.
        times = pd.to_datetime(frame["timestamp"], utc=True)
        frame["timestamp"] = times.dt.tz_convert(self.settings.sensor_timezone).dt.tz_localize(None)
        frame = add_features(frame)
        score, trained_threshold = predict(frame, self.settings.ml_model)
        override = self.settings.ml_threshold_override
        threshold = trained_threshold if override is None else override
        limitations = ["Model evaluation used synthetic anomalies, not labeled incidents"]
        if override is not None:
            limitations.append("The custom threshold requires validation using local observations")
        return {"status": "ok", "model": self.settings.ml_model,
                "timestamp": readings[-1]["timestamp"], "score": score, "threshold": threshold,
                "trained_threshold": trained_threshold,
                "threshold_source": "trained" if override is None else "override",
                "is_anomaly": bool(score > threshold), "severity": severity_from_score(score, threshold),
                "scope": "sensor-pattern anomaly, not a plant disease diagnosis",
                "limitations": limitations}
