# Sensor data agent

Read the CSV or HTTP JSON source, validate six numeric sensor fields and timestamps, normalize timestamps to UTC, and report stale or suspicious data. No LLM or weather key is needed.

Run from the greenhouse project root:

```bash
ml/.venv/bin/python ai-agent/topics/02-sensor-agent/example.py --demo
```

Expected output: a JSON report with explicit source and status. Dry-run master output
contains assembled evidence and no generated recommendations.

Experiment: Set SENSOR_SOURCE=http and configure your device endpoint. Try a duplicate timestamp to see a validation failure.

See [setup and input contracts](../../README.md) for configuration and units.
