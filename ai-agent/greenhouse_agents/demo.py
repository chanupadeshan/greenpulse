"""Explicitly simulated inputs for local workflow checks without hardware/API keys."""
import csv
import json
from datetime import datetime, timezone

from .config import AGENT_ROOT
from .sensors import SensorClient
from .weather import WeatherClient


class DemoSensorClient:
    def __init__(self, settings):
        self.settings = settings

    def fetch(self):
        with (AGENT_ROOT / "examples" / "sensors.csv").open() as file:
            rows = list(csv.DictReader(file))
        rows[0]["timestamp"] = datetime.now(timezone.utc).isoformat()
        report = SensorClient(self.settings).normalize(rows, "example")
        report["warnings"].append("Simulated sensor example; not a live greenhouse measurement")
        return report


class DemoWeatherClient:
    def fetch(self):
        payload = json.loads((AGENT_ROOT / "examples" / "weather.json").read_text())
        payload["dt"] = int(datetime.now(timezone.utc).timestamp())
        report = WeatherClient.normalize(payload, "example")
        report["warnings"].append("Simulated weather example; OpenWeather was not called")
        return report
