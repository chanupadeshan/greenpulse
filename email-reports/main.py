"""Send the latest GreenPulse result and current weather at a chosen interval."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from email.utils import parseaddr

import httpx
from dotenv import dotenv_values

from gmail_sender import authorize, find_credentials, save_private, send_report
from reports import report_body, report_html

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ai-agent"))
from greenhouse_agents.config import Settings
from greenhouse_agents.weather import WeatherClient


def settings_from_args(args):
    # Local email settings override root settings; shell settings take precedence.
    values = dict(dotenv_values(ROOT / ".env"))
    values.update({key: value for key, value in dotenv_values(HERE / ".env").items() if value})
    values.update(os.environ)
    def setting(name, default=None):
        return values.get(name) or default
    latitude = args.latitude if args.latitude is not None else setting("GREENHOUSE_LATITUDE")
    longitude = args.longitude if args.longitude is not None else setting("GREENHOUSE_LONGITUDE")
    if latitude is None or longitude is None:
        raise ValueError("Set latitude and longitude in email-reports/.env or with --latitude and --longitude")
    weather_settings = Settings(latitude=latitude, longitude=longitude,
        weather_api_key=setting("OPENWEATHER_API_KEY", ""), request_timeout_seconds=15)
    if not weather_settings.weather_api_key.get_secret_value():
        raise ValueError("Set OPENWEATHER_API_KEY")
    interval = args.interval if args.interval is not None else int(setting("EMAIL_INTERVAL_SECONDS", "300"))
    if interval < 60:
        raise ValueError("The email interval must be at least 60 seconds")
    recipient = args.to or setting("EMAIL_TO", "")
    if not (args.dry_run or getattr(args, "preview_html", False)):
        display, address = parseaddr(recipient)
        if display or address != recipient or "@" not in address or any(c in recipient for c in "\r\n,;"):
            raise ValueError("Set EMAIL_TO to one plain email address")
    api_url = setting("GREENPULSE_API_URL", "http://127.0.0.1:8010").rstrip("/")
    url = httpx.URL(api_url)
    if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("GREENPULSE_API_URL must be an HTTP(S) URL without credentials or query parameters")
    return values, weather_settings, interval, recipient, api_url


def collect_report(client, api_url, weather_client):
    response = client.get(api_url + "/api/status")
    response.raise_for_status()
    snapshot = response.json()
    prediction = snapshot.get("prediction")
    if not isinstance(prediction, dict) or not prediction.get("id") or not prediction.get("completed_at"):
        return snapshot, None
    try:
        weather = weather_client.fetch()
    except Exception:
        # Do not expose request URLs, API keys, or provider response bodies.
        weather = {"status": "unavailable", "error": "Current weather could not be retrieved. Check weather settings and access."}
    return snapshot, weather


def run(args):
    if args.authorize:
        values = {**dotenv_values(HERE / ".env"), **os.environ}
        credentials_path = find_credentials(HERE, values.get("GMAIL_CREDENTIALS_PATH"))
        token_path = HERE / (values.get("GMAIL_TOKEN_PATH") or "token.json")
        authorize(credentials_path, token_path, values.get("GMAIL_REDIRECT_URI") or "http://localhost:8080/")
        print("Gmail authorized. No email was sent.")
        return 0
    values, weather_settings, interval, recipient, api_url = settings_from_args(args)
    preview_html = getattr(args, "preview_html", False)
    token_path = HERE / (values.get("GMAIL_TOKEN_PATH") or "token.json")
    if not (args.dry_run or preview_html) and not token_path.exists():
        raise ValueError("Run main.py --authorize first")
    state_path = HERE / "runtime" / "last_sent.json"
    last_id = json.loads(state_path.read_text())["run_id"] if state_path.exists() else None
    weather_client = WeatherClient(weather_settings)
    with httpx.Client(timeout=15) as client:
        while True:
            start = time.monotonic()
            try:
                snapshot, weather = collect_report(client, api_url, weather_client)
            except (httpx.HTTPError, ValueError):
                print("Prediction API unavailable or invalid; no email sent.", file=sys.stderr)
                if args.once or args.dry_run or preview_html:
                    return 1
            else:
                if weather is None:
                    print("Waiting for a completed prediction; no email sent.")
                    if args.once or args.dry_run or preview_html:
                        return 1
                else:
                    prediction = snapshot["prediction"]
                    body = report_body(snapshot, weather, weather_settings.latitude, weather_settings.longitude)
                    html_body = report_html(snapshot, weather, weather_settings.latitude, weather_settings.longitude)
                    if preview_html:
                        preview_path = HERE / "runtime" / "preview.html"
                        save_private(preview_path, html_body)
                        print(f"HTML preview saved to {preview_path}. No email sent.")
                        return 0
                    if args.dry_run:
                        print(body)
                        return 0
                    if prediction["id"] == last_id:
                        print("This prediction was already emailed; waiting for a new result.")
                    else:
                        subject = "GreenPulse - current greenhouse situation"
                        if snapshot.get("config", {}).get("demo"):
                            subject = "[DEMO] " + subject
                        # Failures stop this process so an uncertain send is not automatically repeated.
                        message_id = send_report(token_path, recipient, subject, body, html_body=html_body)
                        save_private(state_path, json.dumps({"run_id": prediction["id"], "message_id": message_id}))
                        last_id = prediction["id"]
                        print(f"Report sent. Gmail message ID: {message_id}")
            if args.once:
                return 0
            time.sleep(max(0, interval - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--authorize", action="store_true", help="Authorize Gmail without sending")
    modes.add_argument("--dry-run", action="store_true", help="Preview one report without Gmail access")
    modes.add_argument("--preview-html", action="store_true", help="Save an HTML report preview without sending")
    modes.add_argument("--once", action="store_true", help="Send at most one report, then exit")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--to", help="Recipient email address")
    parser.add_argument("--interval", type=int, help="Seconds between checks (default 300, minimum 60)")
    args = parser.parse_args()
    try:
        # Prevent two local sender processes from emailing the same result.
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True, mode=0o700)
        with (runtime / "sender.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another email sender is already running.", file=sys.stderr)
                return 1
            return run(args)
    except KeyboardInterrupt:
        print("Email sender stopped.")
        return 0
    except Exception as error:
        # Provider exceptions can contain secrets; only show setup errors verbatim.
        detail = str(error) if type(error) is ValueError else type(error).__name__
        if args.authorize:
            print(f"Gmail authorization stopped: {detail}. No email was sent.", file=sys.stderr)
        else:
            print(f"Email sender stopped: {detail}. Check setup; delivery may be uncertain after a send error.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
