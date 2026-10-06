from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "ai-agent"


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-120b"
    weather_api_key: SecretStr = SecretStr("")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    sensor_source: Literal["csv", "http"] = "csv"
    sensor_csv_path: Path = PROJECT_ROOT / "readings.csv"
    sensor_api_url: str = ""
    sensor_api_key: SecretStr = SecretStr("")
    sensor_timezone: str = "Asia/Colombo"
    max_sensor_age_minutes: int = Field(default=120, gt=0)
    ml_model: Literal["autoencoder", "isolation_forest", "lstm_autoencoder"] = "autoencoder"
    crop_name: str = "unspecified"
    growth_stage: str = "unspecified"
    growing_medium: str = "unspecified"
    request_timeout_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_timezone(self):
        ZoneInfo(self.sensor_timezone)
        return self

    @classmethod
    def from_env(cls, env_file=None):
        # Process variables override .env; the file is parsed without exporting secrets.
        values = {**dotenv_values(env_file or PROJECT_ROOT / ".env"), **os.environ}
        mapping = {
            "groq_api_key": "GROQ_API_KEY", "groq_model": "GROQ_MODEL",
            "weather_api_key": "OPENWEATHER_API_KEY", "latitude": "GREENHOUSE_LATITUDE",
            "longitude": "GREENHOUSE_LONGITUDE", "sensor_source": "SENSOR_SOURCE",
            "sensor_csv_path": "SENSOR_CSV_PATH", "sensor_api_url": "SENSOR_API_URL",
            "sensor_api_key": "SENSOR_API_KEY", "sensor_timezone": "SENSOR_TIMEZONE",
            "max_sensor_age_minutes": "MAX_SENSOR_AGE_MINUTES", "ml_model": "ML_MODEL",
            "crop_name": "CROP_NAME", "growth_stage": "GROWTH_STAGE",
            "growing_medium": "GROWING_MEDIUM", "request_timeout_seconds": "REQUEST_TIMEOUT_SECONDS",
        }
        data = {field: values[name] for field, name in mapping.items() if values.get(name)}
        if "sensor_csv_path" in data:
            path = Path(data["sensor_csv_path"])
            data["sensor_csv_path"] = path if path.is_absolute() else PROJECT_ROOT / path
        return cls(**data)
