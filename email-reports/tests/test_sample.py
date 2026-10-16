import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample"))
import send_sample


class SampleTests(unittest.TestCase):
    def test_preview_does_not_send_and_labels_all_data_as_simulated(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(send_sample, "HERE", folder), \
                 patch.object(send_sample, "send_report") as send, \
                 patch.object(send_sample, "dotenv_values", return_value={}), \
                 patch.dict(send_sample.os.environ, {}, clear=True), patch("builtins.print"):
                self.assertEqual(send_sample.run(argparse.Namespace(send=False, scenario="anomaly")), 0)
            send.assert_not_called()
            self.assertFalse((folder / "delivery.json").exists())
            self.assertIn("SAMPLE REPORT", (folder / "report.html").read_text())
            self.assertIn("Anomaly detected: Yes", (folder / "report.txt").read_text())
            self.assertIn("weather are simulated", (folder / "report.txt").read_text())

    def test_send_uses_configured_recipient_once_with_both_email_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(send_sample, "HERE", folder), \
                 patch.object(send_sample, "send_report", return_value="sample-gmail-id") as send, \
                 patch.object(send_sample, "dotenv_values", return_value={"EMAIL_TO": "grower@example.com"}), \
                 patch.dict(send_sample.os.environ, {}, clear=True), patch("builtins.print"):
                self.assertEqual(send_sample.run(argparse.Namespace(send=True, scenario="normal")), 0)
            send.assert_called_once()
            self.assertEqual(send.call_args.args[1], "grower@example.com")
            self.assertTrue(send.call_args.args[2].startswith("[SAMPLE]"))
            self.assertIn("Anomaly detected: No", send.call_args.args[3])
            self.assertIn("SAMPLE REPORT", send.call_args.kwargs["html_body"])
            self.assertEqual(json.loads((folder / "delivery.json").read_text())["gmail_message_id"], "sample-gmail-id")


if __name__ == "__main__":
    unittest.main()
