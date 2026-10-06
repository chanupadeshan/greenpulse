# Complete LangGraph workflow

Define TypedDict state, register four node functions, fan out sensor/weather collection, run ML after sensors, join both branches, and invoke the master exactly once.

Run from the greenhouse project root:

```bash
ml/.venv/bin/python ai-agent/topics/05-greenhouse-workflow/example.py --demo --dry-run
```

Expected output: a JSON report with explicit source and status. Dry-run master output
contains assembled evidence and no generated recommendations.

Experiment: Export a graph source with --graph ai-agent/results/workflow.mmd, or replace demo inputs with a real sensor CSV.

See [setup and input contracts](../../README.md) for configuration and units.
