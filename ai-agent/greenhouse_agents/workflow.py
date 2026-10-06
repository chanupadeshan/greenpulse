from langgraph.graph import END, START, StateGraph

from .agents import build_nodes
from .master import MasterClient
from .ml import MLClient
from .sensors import SensorClient
from .state import GreenhouseState
from .weather import WeatherClient


def build_workflow(settings, *, sensor_client=None, ml_client=None, weather_client=None,
                   master_client=None, dry_run=False):
    nodes = build_nodes(
        sensor_client if sensor_client is not None else SensorClient(settings),
        ml_client if ml_client is not None else MLClient(settings),
        weather_client if weather_client is not None else WeatherClient(settings),
        master_client if master_client is not None else MasterClient(settings, dry_run=dry_run),
    )
    # Define state -> nodes -> edges -> compile -> invoke, as in the learning labs.
    builder = StateGraph(GreenhouseState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "sensor_agent")
    builder.add_edge(START, "weather_agent")
    builder.add_edge("sensor_agent", "ml_agent")
    # Explicit barrier: master runs once after both completed branches.
    builder.add_edge(["ml_agent", "weather_agent"], "master_agent")
    builder.add_edge("master_agent", END)
    return builder.compile()


def initial_state(settings):
    return {"crop_context": {"country": "Sri Lanka", "crop": settings.crop_name,
                             "growth_stage": settings.growth_stage, "growing_medium": settings.growing_medium,
                             "latitude": settings.latitude, "longitude": settings.longitude},
            "history": []}
