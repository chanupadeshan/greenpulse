# Sample situation email

One formatted example with six sensor values, anomaly status, weather, and a short
summary. All readings, model scores, and weather are explicitly simulated; this
does not query AWS, run the model, or call OpenWeather. Coordinates and the recipient
come from `email-reports/.env`. It reuses the existing authorized Gmail token.

From the project root, generate `sample/report.html` and `sample/report.txt` without sending:

```bash
email-reports/.venv/bin/python email-reports/sample/send_sample.py
```

Send **one** sample to your configured `EMAIL_TO`:

```bash
email-reports/.venv/bin/python email-reports/sample/send_sample.py --send
```

The default scenario demonstrates an anomaly. For a normal-result example:

```bash
email-reports/.venv/bin/python email-reports/sample/send_sample.py --scenario normal
```

Add `--send` to email that example too. Sent samples have a `[SAMPLE]` subject and
an in-message sample label. Successful Gmail API responses are recorded in the
private, Git-ignored `sample/delivery.json`; this confirms API acceptance, not inbox
placement. There is no automatic retry. Check Gmail Sent before retrying after a
network error. The script does not start or change the recurring five-minute sender.
