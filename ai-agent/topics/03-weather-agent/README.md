# Weather API agent

Request current outdoor observations from OpenWeather with metric units and exact greenhouse coordinates. Real mode requires OPENWEATHER_API_KEY, GREENHOUSE_LATITUDE and GREENHOUSE_LONGITUDE.

Run from the greenhouse project root:

```bash
ml/.venv/bin/python ai-agent/topics/03-weather-agent/example.py --demo
```

Expected output: a JSON report with explicit source and status. Dry-run master output
contains assembled evidence and no generated recommendations.

Experiment: Run without --demo after entering your key and greenhouse coordinates to fetch real outdoor conditions.

See [setup and input contracts](../../README.md) for configuration and units.
