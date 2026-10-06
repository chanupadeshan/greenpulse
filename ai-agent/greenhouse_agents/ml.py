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
        frame = pd.DataFrame(readings)
        # Hour-of-day features must use local sensor time, not UTC hour.
        times = pd.to_datetime(frame["timestamp"], utc=True)
        frame["timestamp"] = times.dt.tz_convert(self.settings.sensor_timezone).dt.tz_localize(None)
        frame = add_features(frame)
        score, threshold = predict(frame, self.settings.ml_model)
        return {"status": "ok", "model": self.settings.ml_model,
                "timestamp": readings[-1]["timestamp"], "score": score, "threshold": threshold,
                "is_anomaly": bool(score > threshold), "severity": severity_from_score(score, threshold),
                "scope": "sensor-pattern anomaly, not a plant disease diagnosis",
                "limitations": ["Thresholds were validated using synthetic anomalies, not labeled incidents"]}
