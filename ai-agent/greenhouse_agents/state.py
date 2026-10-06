import operator
from typing import Annotated, TypedDict


class GreenhouseState(TypedDict, total=False):
    crop_context: dict
    sensor_report: dict
    ml_report: dict
    weather_report: dict
    master_report: dict
    history: Annotated[list[str], operator.add]
