# GreenPulse Node-RED dashboard

Local FlowFuse Dashboard 2.0 connected to the real saved anomaly model and the
plant assistant. Nothing is published to an external hosting service.

Two tabs separate the jobs:

- **Prediction:** http://127.0.0.1:1880/dashboard/greenhouse — continuous sensor/model monitoring.
- **AI assistant:** http://127.0.0.1:1880/dashboard/assistant — plant conversation on submitted questions.

## Start

From the project root:

```bash
bash scripts/start-dashboard.sh --demo
```

Open **http://127.0.0.1:1880/dashboard/greenhouse**. Demo mode is prominently labeled;
it uses simulated sensors/weather, the real trained model, and a dry-run master agent.
It never calls AWS, OpenWeather, or Groq.

To run against your live device, stop demo with Ctrl+C and start:

```bash
bash scripts/start-dashboard.sh
```

The editor is at **http://127.0.0.1:1880/editor/**. The Python API is at
**http://127.0.0.1:8010/docs**. Both servers bind only to localhost. Ctrl+C stops both.
Use `NODE_RED_PORT` and `GREENPULSE_API_PORT` to change occupied ports.
The launcher checks both ports before starting. If a dashboard is already running,
use that instance or stop it with Ctrl+C before launching another copy. To run a
second instance, choose distinct free ports for both services, for example
`NODE_RED_PORT=1881 GREENPULSE_API_PORT=8011 bash scripts/start-dashboard.sh`.

The launcher uses `ml/.venv` by default because it already contains the model's
validated CPU PyTorch dependencies. Your root `.venv` still contains AWS CLI/Boto3.
To use a different complete Python environment, set `GREENPULSE_PYTHON`.

On a fresh checkout, install dependencies first:

```bash
source .venv/bin/activate
pip install -r ai-agent/requirements-dashboard.txt
npm ci --prefix node-red
GREENPULSE_PYTHON="$PWD/.venv/bin/python" bash scripts/start-dashboard.sh --demo
```

## Live settings

The ignored `.env.aws` loads the local AWS profile, region, table, and device ID.
It is shell setup, not a dotenv file. AWS keys stay in the ignored, mode-600
`.aws-local/credentials`. No AWS or provider keys are embedded in the flow or browser.

The configured device is `ESP32_GREENPULSE_AI`, table `GreenPluse_Sensor_Data`,
region `us-east-1`. A descending DynamoDB **Query**, scoped to the device partition,
retrieves up to `DYNAMODB_READ_LIMIT` records (default 120) per prediction run.
The source is queried every `PREDICTION_INTERVAL_SECONDS` (default 60, minimum 30),
plus on manual requests. The dashboard refreshes cached API state every five seconds;
those UI polls do not read DynamoDB. Querying uses AWS read capacity.

These DynamoDB fields map to the six trained inputs:

| DynamoDB attribute | Model input |
| --- | --- |
| `air_temperature_c` | `air_temp_c` |
| `humidity_percent` | `humidity` |
| `moisture_l1_percent` | `soil_moisture_l1` |
| `soil_temperature_l1_c` | `soil_temp_l1_c` |
| `moisture_l2_percent` | `soil_moisture_l2` |
| `soil_temperature_l2_c` | `soil_temp_l2_c` |

`timestamp` is a numeric Unix epoch in seconds; configure `DYNAMODB_TIMESTAMP_UNIT=ms`
only for devices that send milliseconds. Values must use °C and calibrated percentages.
`SENSOR_FIELD_MAP` optionally overrides mappings using a JSON object in `.env`.

The inspected live records contain **null soil temperatures at both layers**.
The dashboard displays unavailable values and blocks prediction until all six latest
inputs are measured. It never substitutes air temperature, zeros, or historic values.
It also flags stale timestamps and moisture readings at 100% for calibration review.
Fix the soil-temperature measurements at the device before relying on this model.

The default dense autoencoder needs one complete reading. The LSTM also needs its
saved consecutive hourly window; high-frequency raw readings do not satisfy it.
Use the dense autoencoder for the current device stream.

## Plant assistant and configuration

Prediction monitoring runs sensor validation and the saved ML model every minute.
It never calls Groq. The separate AI tab has a question input, conversation, and two modes:

- **General plant knowledge:** answer plant-care and growing questions without attaching device data.
- **Greenhouse insights:** attach the latest cached sensor summary, model report,
  crop context, and any available previously collected weather for anomaly suggestions.

Each question uses a separate worker, so an AI response cannot block scheduled prediction.
Chat does not trigger another DynamoDB query or weather request. AI answers run only when
requested, even if a legacy `AUTO_ASSESS=true` setting remains in the environment.
The original structured five-agent assessment remains available through the CLI and
legacy `/api/run` assessment endpoint for explicit requests.

Edit your existing `.env` to configure:

- `GROQ_API_KEY`: needed for AI answers. The dashboard reloads the project `.env` value on
  each question and gives it precedence over an older shell key. Keys never enter the browser.
- `OPENWEATHER_API_KEY`, `GREENHOUSE_LATITUDE`, `GREENHOUSE_LONGITUDE`: needed for
  current outdoor weather. The key is present, but coordinates are currently missing.
- `CROP_NAME`, `GROWTH_STAGE`, `GROWING_MEDIUM`: optional. Leave unspecified when
  unknown; the master must not invent crop-specific targets.
- `ML_MODEL=autoencoder`: selected trained model.
- `ML_THRESHOLD_OVERRIDE`: optional positive, finite cutoff for the selected model.
  For example, `ML_THRESHOLD_OVERRIDE=0.7` classifies a score of `0.6066` as normal.
  Blank or unset uses the model's trained threshold. Both anomaly status and severity
  use the active cutoff, which the dashboard displays. Model reports retain the
  trained cutoff for reference. Saved weights, scalers, and metadata
  are unchanged. Higher cutoffs reduce alerts and can miss real anomalies; validate
  custom values against local observations. Cutoffs are specific to the selected
  model and must be reconsidered when changing `ML_MODEL`. Restart after editing.
- `PREDICTION_INTERVAL_SECONDS=60`: sensor/ML monitoring interval.

Restart after sensor/model configuration changes. Groq key edits take effect on the next
question. A question sends its text and up to ten prior conversation messages to Groq.
Stored-reading mode also sends a summary, not the full reading history. General mode
does not send greenhouse measurements. The chat is kept in browser memory, is not saved
in report files, and clears on reload. Each reply is routed to the requesting client.
The separate CLI's weather requests send configured coordinates to OpenWeather.

Unavailable providers produce explicit statuses. Sensor monitoring and prediction remain
available independently. This workflow has **no actuator tools or automatic irrigation**.
An anomaly score is not a probability, a yield forecast, or a plant disease diagnosis.

Current reports are saved in ignored `ai-agent/results/dashboard/latest.json`, with
30 recent run summaries. These files are not a second sensor database; they are bounded
workflow output snapshots. Live history starts afresh on server restart. Demo reports
use a separate `dashboard-demo` directory.

## Source and checks

- `dashboard.vue`: editable UI template.
- `assistant.vue`: question input, conversation, and optional evidence context.
- `build-flow.py`: creates the importable `flows.json` from the UI template.
- `flows.json`: complete Node-RED flow with polling, dashboard, run controls, and errors.
- `settings.js`: isolated project runtime configuration.

After editing the UI:

```bash
python3 node-red/build-flow.py
```

Stop/restart Node-RED to reload the generated flow. Editing flows in the Node-RED editor
is supported, but running the generator replaces those edits; update the generator to
retain structural changes.

Checks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ml/.venv/bin/python -m unittest discover -s ai-agent/tests -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ml/.venv/bin/python -m unittest discover -s ml/tests -v
# With the demo dashboard running:
node node-red/verify-flow.mjs
# Template compilation and rendering checks:
node node-red/verify-ui.cjs
```

The interface uses FlowFuse's documented [Vue template node](https://dashboard.flowfuse.com/nodes/widgets/ui-template.html)
and Node-RED's [runtime settings](https://nodered.org/docs/user-guide/runtime/configuration).
