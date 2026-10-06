import argparse
import json
from pathlib import Path

from .config import Settings
from .demo import DemoSensorClient, DemoWeatherClient
from .workflow import build_workflow, initial_state


def main():
    parser = argparse.ArgumentParser(description="GreenPulse four-agent greenhouse assessment")
    parser.add_argument("--env-file", type=Path, help="Default: project-root .env")
    parser.add_argument("--sensor-csv", type=Path, help="Use this CSV instead of configured sensor source")
    parser.add_argument("--model", choices=["autoencoder", "isolation_forest", "lstm_autoencoder"])
    parser.add_argument("--demo", action="store_true", help="Simulated sensor/weather inputs; ML stays real")
    parser.add_argument("--dry-run", action="store_true", help="Assemble evidence without calling Groq")
    parser.add_argument("--output", type=Path, help="Optional output JSON report")
    parser.add_argument("--graph", type=Path, help="Optional Mermaid graph file (no external renderer)")
    args = parser.parse_args()
    if args.demo and args.sensor_csv:
        parser.error("--demo and --sensor-csv cannot be combined")
    try:
        settings = Settings.from_env(args.env_file)
    except Exception:
        parser.error("Invalid .env configuration; check field types, coordinate ranges and timezone")
    updates = {}
    if args.sensor_csv:
        updates.update(sensor_source="csv", sensor_csv_path=args.sensor_csv.resolve())
    if args.model:
        updates["ml_model"] = args.model
    settings = settings.model_copy(update=updates)
    graph = build_workflow(settings, dry_run=args.dry_run,
                           sensor_client=DemoSensorClient(settings) if args.demo else None,
                           weather_client=DemoWeatherClient() if args.demo else None)
    if args.graph:
        args.graph.parent.mkdir(parents=True, exist_ok=True)
        args.graph.write_text(graph.get_graph().draw_mermaid() + "\n")
    result = graph.invoke(initial_state(settings))
    # Persist summaries, not the potentially long full sensor history.
    result["sensor_report"].pop("readings", None)
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if result["sensor_report"]["status"] != "ok" or result["ml_report"]["status"] != "ok":
        return 1
    if result["weather_report"]["status"] != "ok" or result["master_report"]["status"] not in ("ok", "dry_run"):
        return 1
    return 0
