from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    timestamp: datetime
    air_temp_c: float
    humidity: float
    soil_moisture_l1: float
    soil_temp_l1_c: float
    soil_moisture_l2: float
    soil_temp_l2_c: float


class Recommendation(BaseModel):
    priority: Literal["low", "medium", "high"]
    action: str = Field(min_length=1, description="Concrete action for the grower")
    reason: str = Field(min_length=1, description="Evidence and reasoning supporting the action")
    evidence_sources: list[Literal["sensors", "ml", "weather", "crop_context"]] = Field(min_length=1)


class PlantAdvice(BaseModel):
    summary: str = Field(min_length=1)
    recommendations: list[Recommendation]
    missing_information: list[str]
    cautions: list[str]
