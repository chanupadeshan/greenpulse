import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from .schemas import SensorReading


class SensorClient:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client

    def fetch(self):
        if self.settings.sensor_source == "csv":
            with self.settings.sensor_csv_path.open(newline="", encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            source = "csv"
        else:
            if not self.settings.sensor_api_url:
                raise ValueError("SENSOR_API_URL is required for HTTP sensors")
            headers = {}
            key = self.settings.sensor_api_key.get_secret_value()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            def request(client):
                response = client.get(self.settings.sensor_api_url, headers=headers)
                response.raise_for_status()
                return response.json()
            if self.client is None:
                with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
                    payload = request(client)
            else:
                payload = request(self.client)
            rows = payload.get("readings", [payload]) if isinstance(payload, dict) else payload
            source = "http"
        return self.normalize(rows, source)

    def normalize(self, rows, source):
        if not isinstance(rows, list) or not rows:
            raise ValueError("Sensors must supply at least one complete reading")
        readings = []
        for row in rows:
            reading = SensorReading.model_validate(row)
            if reading.timestamp.tzinfo is None:
                reading.timestamp = reading.timestamp.replace(tzinfo=ZoneInfo(self.settings.sensor_timezone))
            reading.timestamp = reading.timestamp.astimezone(timezone.utc)
            readings.append(reading)
        readings.sort(key=lambda item: item.timestamp)
        if len({item.timestamp for item in readings}) != len(readings):
            raise ValueError("Duplicate sensor timestamps")
        now = datetime.now(timezone.utc)
        latest = readings[-1]
        age = (now - latest.timestamp).total_seconds() / 60
        warnings = []
        if age > self.settings.max_sensor_age_minutes:
            warnings.append("Sensor readings are stale; collect fresh readings before acting")
        if age < -5:
            warnings.append("Sensor timestamp is in the future; verify device clock")
        ranges = {"humidity": (0, 100), "soil_moisture_l1": (0, 100), "soil_moisture_l2": (0, 100)}
        for name, (low, high) in ranges.items():
            if not low <= getattr(latest, name) <= high:
                warnings.append(f"{name} is outside 0–100%; verify sensor calibration")
        return {
            "status": "ok", "source": source,
            "readings": [item.model_dump(mode="json") for item in readings],
            "latest": latest.model_dump(mode="json"), "row_count": len(readings),
            "age_minutes": round(age, 2), "warnings": warnings,
            "units": {"temperature": "°C", "humidity": "%", "soil_moisture": "%"},
        }
