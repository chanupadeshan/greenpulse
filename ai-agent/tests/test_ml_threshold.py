import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenhouse_agents.config import Settings
from greenhouse_agents.ml import MLClient


class ThresholdTests(unittest.TestCase):
    def reading(self):
        return {"timestamp": "2026-10-15T15:10:40+00:00", "air_temp_c": 27.3,
                "humidity": 87.2, "soil_moisture_l1": 50.21, "soil_temp_l1_c": 27.06,
                "soil_moisture_l2": 42.57, "soil_temp_l2_c": 27.37}

    def test_override_controls_anomaly_and_severity_and_preserves_trained_cutoff(self):
        for override, anomaly, severity, source in (
            (None, True, "high", "trained"),
            (0.5, True, "moderate", "override"),
            (0.7, False, "normal", "override"),
            (0.606566, False, "normal", "override"),
        ):
            with self.subTest(override=override):
                with patch("greenhouse_agents.ml.predict", return_value=(0.606566, 0.178632)):
                    report = MLClient(Settings(ml_threshold_override=override)).analyze([self.reading()])
                self.assertEqual(report["score"], 0.606566)
                self.assertEqual(report["trained_threshold"], 0.178632)
                self.assertEqual(report["threshold"], 0.178632 if override is None else override)
                self.assertEqual(report["is_anomaly"], anomaly)
                self.assertEqual(report["severity"], severity)
                self.assertEqual(report["threshold_source"], source)
                self.assertEqual(any("custom threshold" in note for note in report["limitations"]),
                                 override is not None)

    def test_override_does_not_bypass_missing_measurements(self):
        reading = self.reading()
        reading["humidity"] = None
        with patch("greenhouse_agents.ml.predict") as predict:
            report = MLClient(Settings(ml_threshold_override=0.7)).analyze([reading])
        self.assertEqual(report["status"], "blocked")
        predict.assert_not_called()

    def test_threshold_from_dotenv_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ML_THRESHOLD_OVERRIDE=0.7\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(Settings.from_env(path).ml_threshold_override, 0.7)
            with patch.dict(os.environ, {"ML_THRESHOLD_OVERRIDE": "1.0"}, clear=True):
                self.assertEqual(Settings.from_env(path).ml_threshold_override, 1.0)
            path.write_text("ML_THRESHOLD_OVERRIDE=\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(Settings.from_env(path).ml_threshold_override)

    def test_invalid_cutoffs_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), float("-inf"), "invalid"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Settings(ml_threshold_override=value)


if __name__ == "__main__":
    unittest.main()
