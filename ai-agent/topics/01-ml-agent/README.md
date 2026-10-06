# ML model agent

Fetch valid sensor readings, calculate the same 13 model features used during training, apply the saved scaler, and score the latest reading. The default model is the dense autoencoder. No Groq key is needed.

Run from the greenhouse project root:

```bash
ml/.venv/bin/python ai-agent/topics/01-ml-agent/example.py --demo
```

Expected output: a JSON report with explicit source and status. Dry-run master output
contains assembled evidence and no generated recommendations.

Experiment: Change ML_MODEL in .env to isolation_forest and compare scores. Scores across different models are not on the same scale.

See [setup and input contracts](../../README.md) for configuration and units.
