from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, ConfigDict, Field


class CurrentConditions(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    air_temp_c: float
    humidity: float = Field(ge=0, le=100)
    wind_speed_m_s: float = Field(ge=0)
    rain_last_hour_mm: float = Field(default=0, ge=0)
    observed_at: datetime
    description: str


class WeatherClient:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client

    def fetch(self):
        key = self.settings.weather_api_key.get_secret_value()
        if not key:
            raise ValueError("OPENWEATHER_API_KEY is required")
        if self.settings.latitude is None or self.settings.longitude is None:
            raise ValueError("GREENHOUSE_LATITUDE and GREENHOUSE_LONGITUDE are required")
        params = {"lat": self.settings.latitude, "lon": self.settings.longitude,
                  "appid": key, "units": "metric"}
        def request(client):
            response = client.get("https://api.openweathermap.org/data/2.5/weather", params=params)
            response.raise_for_status()
            return response.json()
        if self.client is None:
            with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
                payload = request(client)
        else:
            payload = request(self.client)
        return self.normalize(payload)

    @staticmethod
    def normalize(payload, source="openweather"):
        conditions = CurrentConditions(
            air_temp_c=payload["main"]["temp"], humidity=payload["main"]["humidity"],
            wind_speed_m_s=payload["wind"]["speed"],
            rain_last_hour_mm=payload.get("rain", {}).get("1h", 0),
            observed_at=datetime.fromtimestamp(payload["dt"], tz=timezone.utc),
            description=payload["weather"][0]["description"],
        )
        warnings = []
        age = (datetime.now(timezone.utc) - conditions.observed_at).total_seconds() / 60
        if age > 120 or age < -5:
            warnings.append("Weather observation is stale or has an invalid clock")
        return {"status": "ok", "source": source, "conditions": conditions.model_dump(mode="json"),
                "scope": "outdoor current weather, not an indoor measurement or forecast",
                "warnings": warnings}
