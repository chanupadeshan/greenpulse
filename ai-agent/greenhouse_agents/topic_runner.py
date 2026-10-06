import argparse
import json

from .agents import build_nodes
from .config import Settings
from .demo import DemoSensorClient, DemoWeatherClient
from .master import MasterClient
from .ml import MLClient
from .sensors import SensorClient
from .weather import WeatherClient
from .workflow import initial_state


def run_topic(name):
    parser = argparse.ArgumentParser(description=f"Standalone {name} example")
    parser.add_argument("--demo", action="store_true", help="Simulated sensors/weather; real ML")
    parser.add_argument("--dry-run", action="store_true", help="No Groq call")
    args = parser.parse_args()
    settings = Settings.from_env()
    nodes = build_nodes(
        DemoSensorClient(settings) if args.demo else SensorClient(settings), MLClient(settings),
        DemoWeatherClient() if args.demo else WeatherClient(settings),
        MasterClient(settings, dry_run=args.dry_run),
    )
    state = initial_state(settings)
    dependencies = {
        "sensor_agent": ["sensor_agent"], "ml_agent": ["sensor_agent", "ml_agent"],
        "weather_agent": ["weather_agent"],
        "master_agent": ["sensor_agent", "ml_agent", "weather_agent", "master_agent"],
    }
    for node in dependencies[name]:
        state.update(nodes[node](state))
    report_name = {"sensor_agent": "sensor_report", "ml_agent": "ml_report",
                   "weather_agent": "weather_report", "master_agent": "master_report"}[name]
    print(json.dumps(state[report_name], indent=2, ensure_ascii=False, allow_nan=False))
