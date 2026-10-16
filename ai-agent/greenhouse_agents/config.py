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
    sensor_source: Literal["csv", "http", "dynamodb"] = "csv"
    aws_region: str = "us-east-1"
    dynamodb_table: str = "GreenPluse_Sensor_Data"
    dynamodb_device_id: str = ""
    dynamodb_read_limit: int = Field(default=120, ge=1, le=1000)
    dynamodb_timestamp_unit: Literal["s", "ms"] = "s"
    sensor_field_map: dict[str, str] = Field(default_factory=lambda: {
        "air_temp_c": "air_temperature_c", "humidity": "humidity_percent",
        "soil_moisture_l1": "moisture_l1_percent", "soil_temp_l1_c": "soil_temperature_l1_c",
        "soil_moisture_l2": "moisture_l2_percent", "soil_temp_l2_c": "soil_temperature_l2_c",
    })
    sensor_csv_path: Path = PROJECT_ROOT / "readings.csv"
    sensor_api_url: str = ""
    sensor_api_key: SecretStr = SecretStr("")
    sensor_timezone: str = "Asia/Colombo"
    max_sensor_age_minutes: int = Field(default=120, gt=0)
    ml_model: Literal["autoencoder", "isolation_forest", "lstm_autoencoder"] = "autoencoder"
    ml_threshold_override: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    crop_name: str = "unspecified"
    growth_stage: str = "unspecified"
    growing_medium: str = "unspecified"
    request_timeout_seconds: float = Field(default=30, gt=0)
    prediction_interval_seconds: int = Field(default=60, ge=30, le=86400)
    auto_assess: bool = False

    @model_validator(mode="after")
    def validate_timezone(self):
        ZoneInfo(self.sensor_timezone)
        return self

    @classmethod
    def from_env(cls, env_file=None, *, prefer_file_keys=()):
        # Process variables override .env; the file is parsed without exporting secrets.
        file_values = dotenv_values(env_file or PROJECT_ROOT / ".env")
        values = {**file_values, **os.environ}
        # The interactive dashboard reloads the explicitly managed project Groq key.
        for name in prefer_file_keys:
            if name in file_values:
                values[name] = file_values[name]
        mapping = {
            "groq_api_key": "GROQ_API_KEY", "groq_model": "GROQ_MODEL",
            "weather_api_key": "OPENWEATHER_API_KEY", "latitude": "GREENHOUSE_LATITUDE",
            "longitude": "GREENHOUSE_LONGITUDE", "sensor_source": "SENSOR_SOURCE",
            "sensor_csv_path": "SENSOR_CSV_PATH", "sensor_api_url": "SENSOR_API_URL",
            "sensor_api_key": "SENSOR_API_KEY", "sensor_timezone": "SENSOR_TIMEZONE",
            "max_sensor_age_minutes": "MAX_SENSOR_AGE_MINUTES", "ml_model": "ML_MODEL",
            "ml_threshold_override": "ML_THRESHOLD_OVERRIDE",
            "crop_name": "CROP_NAME", "growth_stage": "GROWTH_STAGE",
            "growing_medium": "GROWING_MEDIUM", "request_timeout_seconds": "REQUEST_TIMEOUT_SECONDS",
            "aws_region": "AWS_DEFAULT_REGION", "dynamodb_table": "DYNAMODB_TABLE",
            "dynamodb_device_id": "DYNAMODB_DEVICE_ID", "dynamodb_read_limit": "DYNAMODB_READ_LIMIT",
            "dynamodb_timestamp_unit": "DYNAMODB_TIMESTAMP_UNIT",
            "prediction_interval_seconds": "PREDICTION_INTERVAL_SECONDS", "auto_assess": "AUTO_ASSESS",
        }
        data = {field: values[name] for field, name in mapping.items() if values.get(name)}
        if values.get("SENSOR_FIELD_MAP"):
            import json
            data["sensor_field_map"] = json.loads(values["SENSOR_FIELD_MAP"])
        if "sensor_csv_path" in data:
            path = Path(data["sensor_csv_path"])
            data["sensor_csv_path"] = path if path.is_absolute() else PROJECT_ROOT / path
        return cls(**data)
