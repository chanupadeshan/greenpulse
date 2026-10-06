# Master plant specialist agent

Gather all three reports and crop context, then run a Groq prompt chain with a validated PlantAdvice schema. The master requests missing crop information and distinguishes sensor-pattern alerts from diagnoses.

Run from the greenhouse project root:

```bash
ml/.venv/bin/python ai-agent/topics/04-master-agent/example.py --demo --dry-run
```

Expected output: a JSON report with explicit source and status. Dry-run master output
contains assembled evidence and no generated recommendations.

Experiment: Remove --dry-run after setting GROQ_API_KEY. Demo inputs remain explicitly simulated.

See [setup and input contracts](../../README.md) for configuration and units.
