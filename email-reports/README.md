# GreenPulse email reports

For a clearly labeled example without a live device, see [sample/README.md](sample/README.md).

Separate Python sender for current outdoor weather plus the latest cached greenhouse
sensor/model result. Checks every 300 seconds (five minutes) by default and sends only when a new
completed prediction exists. Reports include all six sensor values, an explicit
"Anomaly detected: Yes/No" result (or unavailable), and current outdoor weather.
Emails have a formatted HTML layout: a short situation summary, six measurement
cards, an anomaly assessment, weather details, and data-quality checks. A plain-text
alternative is included. Summaries are built from supplied measurements without an
additional LLM request. HTML uses inline styles and tables with no remote images or scripts.
This is a snapshot, not a five-minute average. It does
not run the LLM or change the existing dashboard, monitoring, or AWS settings.

## 1. Install

Run from the project root (Linux/macOS; the sender uses a local process lock):

```bash
python3 -m venv email-reports/.venv
email-reports/.venv/bin/python -m pip install -r email-reports/requirements.txt
cp email-reports/.env.example email-reports/.env
```

Edit `email-reports/.env`:

```dotenv
GREENHOUSE_LATITUDE=6.9271
GREENHOUSE_LONGITUDE=79.8612
OPENWEATHER_API_KEY=your_actual_key
EMAIL_TO=your-address@gmail.com
EMAIL_INTERVAL_SECONDS=300
GREENPULSE_API_URL=http://127.0.0.1:8010
```

Coordinates above are an example for Colombo; enter your greenhouse location.
Latitude must be -90 to 90 and longitude -180 to 180.
Root `.env` supplies fallback weather settings; nonblank settings in this folder
override them, and shell variables override both. CLI coordinates override all files.
These coordinates affect emailed weather only; they do not change the dashboard's
configuration. Secret and runtime files in this folder are ignored by Git.

## 2. Set up Gmail

1. Create/select a project in [Google Cloud Console](https://console.cloud.google.com/).
2. Enable Gmail API.
3. Configure Google Auth Platform branding, audience, and data access. For a personal
   Gmail prototype, use an External audience and add your sending account as a test user
   while the app is in Testing.
4. Add the `https://www.googleapis.com/auth/gmail.send` scope.
5. Create an OAuth client with application type **Desktop app**. If you already created
   a **Web application** client, it is also supported: add exactly
   `http://localhost:8080/` under **Authorized redirect URIs** in Google Cloud and save.
   Keep the trailing slash. No JavaScript origin is needed for this Python flow.
   If changing the port, also set `GMAIL_REDIRECT_URI` in `email-reports/.env`.
6. Download its JSON into `email-reports/`. You can keep Google's
   `client_secret_...json` filename; the code finds it automatically when there is
   exactly one downloaded file. A file named `credentials.json` takes precedence;
   `GMAIL_CREDENTIALS_PATH` can explicitly select a file. Restrict file access:

```bash
chmod 600 email-reports/client_secret_*.json
email-reports/.venv/bin/python email-reports/main.py --authorize
```

Choose the Gmail account that should send the reports and approve sending access.
If port 8080 is already used, set `GMAIL_REDIRECT_URI=http://localhost:8765/`
in `email-reports/.env` and add that exact URI to the Google Cloud OAuth client.
Then run authorization again. Keep existing services running; no need to stop them.
This saves a private `token.json`; authorization sends no email. No Gmail password
is needed. Token refresh is handled automatically when permitted by Google. Testing
mode refresh tokens can expire; if access is revoked/expired, authorize again.
This local browser OAuth setup is for a prototype; choose an appropriate server OAuth
deployment before making this a hosted multi-user feature.

Official references: [Python OAuth setup](https://developers.google.com/workspace/gmail/api/quickstart/python),
[sending messages](https://developers.google.com/workspace/gmail/api/guides/sending),
[OAuth token expiration](https://developers.google.com/identity/protocols/oauth2#expiration).

## 3. Start monitoring

Use the existing dashboard environment and start the dashboard in another terminal:

```bash
bash scripts/start-dashboard.sh
```

It supplies `/api/status` at `http://127.0.0.1:8010`. The email process does not query
AWS directly or trigger a new prediction. It reads the cached prediction generated
by your existing one-minute monitor. If that service is stopped, no report is sent.
For a demo dashboard, reports clearly label the simulated greenhouse readings;
weather is still fetched from the real API.

## 4. Preview without sending

```bash
email-reports/.venv/bin/python email-reports/main.py --dry-run
```

Preview calls the local prediction API and OpenWeather, prints one report, and exits.
It never authorizes or calls Gmail and needs no Gmail credentials or recipient.

You can supply a location directly:

```bash
email-reports/.venv/bin/python email-reports/main.py --dry-run --latitude 6.9271 --longitude 79.8612
```

Save the formatted version to `email-reports/runtime/preview.html` without sending:

```bash
email-reports/.venv/bin/python email-reports/main.py --preview-html
```

Open the saved file in your browser. It uses the current cached prediction and a
weather request; Gmail authorization is not required for previews. Report timestamps
use the dashboard timezone (Asia/Colombo by default); original timestamps remain in
the plain-text alternative. Email client appearance may vary slightly.

## 5. Send

Send at most one report:

```bash
email-reports/.venv/bin/python email-reports/main.py --once
```

Keep sending new results every five minutes (Ctrl+C stops):

```bash
email-reports/.venv/bin/python email-reports/main.py
```

Override recipient or interval:

```bash
email-reports/.venv/bin/python email-reports/main.py --to your-address@gmail.com --interval 600
```

## Behavior and limits

- The first check happens immediately. Subsequent checks are 300 seconds apart by default
  under normal operation; network work can delay a cycle.
- New prediction run IDs are emailed even if measured values are unchanged. Already
  sent run IDs are skipped across restarts using `runtime/last_sent.json`.
- One process per folder is allowed. Run a single sender instance, not separate copies
  on several machines. Delivery state is local and applies across recipient changes.
- Missing sensor fields stay unavailable. Blocked model results and stale reading
  warnings remain visible. Original sensor and weather timestamps are included.
- Weather failures produce an unavailable section while available prediction data is
  still sent. Current weather is not necessarily updated each minute by the provider.
- API failures are retried on the next cycle. Gmail/auth/state-write failures stop
  sending. Inspect Gmail Sent before restarting after an uncertain delivery: Gmail
  sending and local state writes are not one transaction, so exactly-once delivery
  cannot be guaranteed after a lost response or crash.
- Every five minutes can mean 288 emails per day. Adjust the interval to your account's
  sending limits. Keep the API URL on a trusted network; this tool adds no API auth.

## Offline checks

```bash
email-reports/.venv/bin/python -m unittest discover -s email-reports/tests -v
```

Tests use mocked API/weather/Gmail behavior; they send no messages.
