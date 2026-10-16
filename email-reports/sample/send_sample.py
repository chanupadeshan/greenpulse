"""Preview or send one sample report with clearly labeled simulated data."""

import argparse
from datetime import datetime, timezone
from email.utils import parseaddr
import json
import math
import os
from pathlib import Path
import sys
import uuid

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
EMAIL_ROOT = HERE.parent
sys.path.insert(0, str(EMAIL_ROOT))
from gmail_sender import save_private, send_report
from reports import report_body, report_html


def sample_data(scenario):
    # These values demonstrate the report format. They are not live measurements.
    now = datetime.now(timezone.utc).isoformat()
    anomaly = scenario == "anomaly"
    snapshot = {
        "config": {"demo": True, "timezone": "Asia/Colombo"},
        "prediction": {
            "id": "sample-" + str(uuid.uuid4()), "completed_at": now,
            "sensor_report": {
                "status": "ok", "source": "sample",
                "latest": {"timestamp": now, "air_temp_c": 31.2 if anomaly else 28.5,
                           "humidity": 76.4, "soil_moisture_l1": 23.5 if anomaly else 43.9,
                           "soil_moisture_l2": 39.6, "soil_temp_l1_c": 27.1, "soil_temp_l2_c": 26.8},
                "warnings": ["Sample only: sensor values and model results are simulated, not computed from your device."]},
            "ml_report": {"status": "ok", "model": "autoencoder (simulated result)",
                          "score": 0.42 if anomaly else 0.12, "threshold": 0.1786,
                          "is_anomaly": anomaly, "severity": "high" if anomaly else "normal"},
        },
    }
    weather = {"status": "ok", "source": "sample",
        "conditions": {"air_temp_c": 29.1, "humidity": 82, "wind_speed_m_s": 2.4,
                       "rain_last_hour_mm": 0, "description": "Scattered clouds (simulated)", "observed_at": now},
        "warnings": ["Sample only: weather information is simulated; OpenWeather was not called."]}
    return snapshot, weather


def run(args):
    values = {**dotenv_values(EMAIL_ROOT / ".env"), **os.environ}
    recipient = values.get("EMAIL_TO") or ""
    latitude = float(values.get("GREENHOUSE_LATITUDE") or "6.98691667")
    longitude = float(values.get("GREENHOUSE_LONGITUDE") or "80.01619444")
    if not math.isfinite(latitude) or not -90 <= latitude <= 90 or not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Set valid latitude and longitude in email-reports/.env")
    snapshot, weather = sample_data(args.scenario)
    body = report_body(snapshot, weather, latitude, longitude)
    html_body = report_html(snapshot, weather, latitude, longitude)
    save_private(HERE / "report.txt", body)
    save_private(HERE / "report.html", html_body)
    print(f"Sample preview saved: {HERE / 'report.html'}")
    if not args.send:
        print("Preview only. No email sent. Use --send to send one sample.")
        return 0
    display, address = parseaddr(recipient)
    if display or address != recipient or "@" not in address or any(c in recipient for c in "\r\n,;"):
        raise ValueError("Set EMAIL_TO to one plain email address in email-reports/.env")
    token_path = EMAIL_ROOT / (values.get("GMAIL_TOKEN_PATH") or "token.json")
    subject = "[SAMPLE] GreenPulse - five-minute situation report"
    # Send exactly one API request; do not start the recurring sender or change its state.
    message_id = send_report(token_path, recipient, subject, body, html_body=html_body)
    save_private(HERE / "delivery.json", json.dumps({
        "gmail_message_id": message_id, "recipient": recipient,
        "sent_at": datetime.now(timezone.utc).isoformat(), "scenario": args.scenario,
        "sample_run_id": snapshot["prediction"]["id"],
    }, indent=2))
    print(f"Sample sent to {recipient}. Gmail message ID: {message_id}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", help="Send one sample to EMAIL_TO")
    parser.add_argument("--scenario", choices=("normal", "anomaly"), default="anomaly")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        detail = str(error) if type(error) is ValueError else type(error).__name__
        print(f"Sample stopped: {detail}. Check Gmail Sent before repeating an uncertain delivery.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
