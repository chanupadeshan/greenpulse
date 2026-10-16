import argparse
import base64
import errno
from email import policy
from email.parser import BytesParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from gmail_sender import authorize, authorization_options, create_message, find_credentials
from reports import report_body, report_html, situation_summary


def snapshot():
    return {"config": {"demo": True}, "prediction": {
        "id": "run-123", "completed_at": "2026-01-01T00:00:00+00:00",
        "sensor_report": {"status": "ok", "latest": {
            "timestamp": "2026-01-01T00:00:00+00:00", "air_temp_c": 28,
            "soil_temp_l1_c": None}, "warnings": ["Soil probe missing"]},
        "ml_report": {"status": "blocked", "error": "Measured soil temperature required"}}}


class ReportTests(unittest.TestCase):
    def test_occupied_oauth_port_reports_a_clear_error_without_saving_token(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            credentials = folder / "credentials.json"
            credentials.write_text(json.dumps({"web": {}}))
            token = folder / "token.json"
            with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file") as factory:
                factory.return_value.run_local_server.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
                with self.assertRaisesRegex(ValueError, "8765 is already in use"), patch("builtins.print"):
                    authorize(credentials, token, "http://localhost:8765/")
                self.assertEqual(factory.return_value.run_local_server.call_args.kwargs["port"], 8765)
                self.assertFalse(token.exists())

    def test_downloaded_web_credentials_use_fixed_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            downloaded = folder / "client_secret_example.json"
            downloaded.write_text(json.dumps({"web": {"client_id": "example", "client_secret": "test"}}))
            self.assertEqual(find_credentials(folder), downloaded)
            self.assertEqual(authorization_options(downloaded, "http://localhost:8080/"),
                             {"host": "localhost", "port": 8080})
            with self.assertRaisesRegex(ValueError, "loopback"):
                authorization_options(downloaded, "https://example.com/")

    def test_explicit_credentials_and_desktop_clients_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            desktop = folder / "credentials.json"
            desktop.write_text(json.dumps({"installed": {"client_id": "example"}}))
            self.assertEqual(find_credentials(folder, "credentials.json"), desktop)
            self.assertEqual(authorization_options(desktop, "http://localhost:8080/"),
                             {"host": "localhost", "port": 0})
            desktop.unlink()
            for name in ("client_secret_one.json", "client_secret_two.json"):
                (folder / name).write_text("{}")
            with self.assertRaisesRegex(ValueError, "Keep one"):
                find_credentials(folder)

    def test_blank_local_weather_uses_root_settings_and_zero_coordinates_work(self):
        args = argparse.Namespace(latitude=0.0, longitude=0.0, interval=None, to=None, dry_run=True)
        root = {"GREENHOUSE_LATITUDE": "6.9", "GREENHOUSE_LONGITUDE": "79.8", "OPENWEATHER_API_KEY": "test"}
        local = {"GREENHOUSE_LATITUDE": "", "OPENWEATHER_API_KEY": ""}
        with patch.dict(main.os.environ, {}, clear=True), \
             patch.object(main, "dotenv_values", side_effect=[root, local]):
            _, weather, interval, _, _ = main.settings_from_args(args)
        self.assertEqual((weather.latitude, weather.longitude), (0.0, 0.0))
        self.assertEqual(weather.weather_api_key.get_secret_value(), "test")
        self.assertEqual(interval, 300)

    def test_rejects_short_interval_and_multiple_recipients(self):
        args = argparse.Namespace(latitude=6.9, longitude=79.8, interval=1, to="a@example.com", dry_run=False)
        with patch.dict(main.os.environ, {"OPENWEATHER_API_KEY": "test"}, clear=True), \
             patch.object(main, "dotenv_values", return_value={}):
            with self.assertRaisesRegex(ValueError, "at least 60"):
                main.settings_from_args(args)
            args.interval = 60
            args.to = "a@example.com,b@example.com"
            with self.assertRaisesRegex(ValueError, "one plain email"):
                main.settings_from_args(args)

    def test_missing_data_stays_missing_and_demo_is_visible(self):
        body = report_body(snapshot(), {"status": "unavailable", "error": "Weather unavailable"}, 6.9, 79.8)
        for text in ("DEMO", "Soil temperature L1: unavailable", "Measured soil temperature required",
                     "Weather unavailable", "older than", "not an average", "Soil probe missing"):
            self.assertIn(text, body)

    def test_mime_round_trip_preserves_text_and_recipient(self):
        raw = create_message("grower@example.com", "GreenPulse", "Temperature: 28 °C")
        message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw["raw"]))
        self.assertEqual(message["To"], "grower@example.com")
        self.assertIn("28 °C", message.get_content())
        self.assertIsNone(message["Bcc"])

    def test_html_email_contains_both_alternatives(self):
        raw = create_message("grower@example.com", "GreenPulse", "Plain report", "<h1>HTML report</h1>")
        message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw["raw"]))
        self.assertEqual(message.get_content_type(), "multipart/alternative")
        self.assertEqual(message.get_body(preferencelist=("plain",)).get_content().strip(), "Plain report")
        self.assertIn("HTML report", message.get_body(preferencelist=("html",)).get_content())

    def test_html_escapes_untrusted_text_and_preserves_missing_data(self):
        weather = {"status": "ok", "conditions": {"description": '<script>alert("test")</script>'}}
        html = report_html(snapshot(), weather, 6.9, 79.8)
        for text in ("DEMO REPORT", "Prediction unavailable", "Missing measurements", "unavailable", "Asia/Colombo"):
            self.assertIn(text, html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_summary_distinguishes_normal_anomaly_and_unavailable(self):
        report = snapshot()
        self.assertEqual(situation_summary(report, {})[0], "Prediction unavailable")
        report["prediction"]["ml_report"] = {"status": "ok", "is_anomaly": False}
        self.assertEqual(situation_summary(report, {})[0], "No anomaly detected by the model")
        report["prediction"]["ml_report"]["is_anomaly"] = True
        self.assertEqual(situation_summary(report, {})[0], "Unusual sensor pattern detected")

    def test_weather_failure_does_not_discard_prediction_or_expose_key(self):
        client = Mock()
        client.get.return_value.json.return_value = snapshot()
        weather = Mock()
        weather.fetch.side_effect = RuntimeError("secret-provider-key")
        result, conditions = main.collect_report(client, "http://localhost:8010", weather)
        self.assertEqual(result["prediction"]["id"], "run-123")
        self.assertEqual(conditions["status"], "unavailable")
        self.assertNotIn("secret-provider-key", str(conditions))

    def test_no_completed_prediction_does_not_fetch_weather(self):
        client, weather = Mock(), Mock()
        client.get.return_value.json.return_value = {"prediction": None}
        self.assertIsNone(main.collect_report(client, "http://localhost", weather)[1])
        weather.fetch.assert_not_called()

    def test_send_deduplicates_across_restarts_and_preview_never_sends(self):
        settings = Mock(latitude=6.9, longitude=79.8)
        args = argparse.Namespace(authorize=False, dry_run=False, once=True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "token.json").write_text("test")
            with patch.object(main, "HERE", folder), \
                 patch.object(main, "settings_from_args", return_value=({}, settings, 60, "grower@example.com", "http://localhost")), \
                 patch.object(main, "WeatherClient"), patch.object(main.httpx, "Client"), \
                 patch.object(main, "collect_report", return_value=(snapshot(), {"status": "unavailable"})), \
                 patch.object(main, "send_report", return_value="gmail-123") as send:
                self.assertEqual(main.run(args), 0)
                self.assertEqual(main.run(args), 0)
                send.assert_called_once()
                self.assertIn("SITUATION SUMMARY", send.call_args.kwargs["html_body"])
                state = folder / "runtime/last_sent.json"
                self.assertEqual(json.loads(state.read_text())["run_id"], "run-123")
                self.assertEqual(state.stat().st_mode & 0o777, 0o600)
                args.dry_run = True
                with patch("builtins.print"):
                    self.assertEqual(main.run(args), 0)
                send.assert_called_once()
                args.dry_run = False
                args.preview_html = True
                (folder / "token.json").unlink()
                with patch("builtins.print"):
                    self.assertEqual(main.run(args), 0)
                self.assertIn("SITUATION SUMMARY", (folder / "runtime/preview.html").read_text())
                send.assert_called_once()

    def test_gmail_failure_does_not_mark_result_sent(self):
        args = argparse.Namespace(authorize=False, dry_run=False, once=True)
        settings = Mock(latitude=6.9, longitude=79.8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "token.json").write_text("test")
            with patch.object(main, "HERE", folder), \
                 patch.object(main, "settings_from_args", return_value=({}, settings, 60, "grower@example.com", "http://localhost")), \
                 patch.object(main, "WeatherClient"), patch.object(main.httpx, "Client"), \
                 patch.object(main, "collect_report", return_value=(snapshot(), {"status": "unavailable"})), \
                 patch.object(main, "send_report", side_effect=RuntimeError("send failed")):
                with self.assertRaises(RuntimeError):
                    main.run(args)
                self.assertFalse((folder / "runtime/last_sent.json").exists())


if __name__ == "__main__":
    unittest.main()
