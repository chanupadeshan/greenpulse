import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from greenhouse_agents.api import Monitor, create_app
from greenhouse_agents.config import Settings
from greenhouse_agents.decision import decision_report
from greenhouse_agents.dynamodb import DynamoSensorClient
from greenhouse_agents.ml import MLClient
from greenhouse_agents.sensors import SensorClient


def item(epoch=None, incomplete=False):
    return {"timestamp": Decimal(str(epoch or datetime.now(timezone.utc).timestamp())),
            "device_id": "device-1", "air_temperature_c": Decimal("25.6"),
            "humidity_percent": Decimal("91"), "moisture_l1_percent": Decimal("100"),
            "moisture_l2_percent": Decimal("45"), "soil_temperature_l1_c": None if incomplete else Decimal("26"),
            "soil_temperature_l2_c": None if incomplete else Decimal("27")}


class DynamoTests(unittest.TestCase):
    def test_query_is_device_scoped_newest_first_and_bounded(self):
        class Table:
            def query(inner, **kwargs):
                self.assertFalse(kwargs['ScanIndexForward'])
                self.assertEqual(kwargs['Limit'], 120)
                self.assertTrue(kwargs['ConsistentRead'])
                expr = kwargs['KeyConditionExpression'].get_expression()
                self.assertEqual(expr['values'][1], 'device-1')
                return {'Items': [item(1791768274), item(1791768254)]}
        settings = Settings(sensor_source='dynamodb', dynamodb_device_id='device-1')
        report = DynamoSensorClient(settings, Table()).fetch()
        self.assertEqual(report['device_id'], 'device-1')
        self.assertEqual(report['latest']['air_temp_c'], 25.6)
        self.assertLess(report['readings'][0]['timestamp'], report['latest']['timestamp'])
        self.assertEqual(report['latest']['timestamp'], datetime.fromtimestamp(1791768274, timezone.utc).isoformat())
        json.dumps(report, allow_nan=False)

    def test_missing_temperatures_are_visible_and_block_real_model(self):
        settings = Settings()
        report = DynamoSensorClient(settings).normalize([item(incomplete=True)])
        self.assertEqual(report['missing_fields'], ['soil_temp_l1_c', 'soil_temp_l2_c'])
        self.assertIsNone(report['latest']['soil_temp_l1_c'])
        with patch('greenhouse_agents.ml.predict', side_effect=AssertionError('Inference must not run')):
            result = MLClient(settings).analyze(report['readings'])
        self.assertEqual(result['status'], 'blocked')
        self.assertNotIn('score', result)
        self.assertTrue(any('calibration' in w for w in report['warnings']))

    def test_invalid_values_and_missing_device_fail_closed(self):
        bad = item(); bad['humidity_percent'] = Decimal('NaN')
        with self.assertRaises(ValueError):
            DynamoSensorClient(Settings()).normalize([bad])
        with self.assertRaises(ValueError):
            DynamoSensorClient(Settings()).fetch()
        with self.assertRaises(ValueError):
            DynamoSensorClient(Settings()).normalize([])

    def test_millisecond_epoch_is_explicit(self):
        report = DynamoSensorClient(Settings(dynamodb_timestamp_unit='ms')).normalize([item(1791768274000)])
        self.assertEqual(report['latest']['timestamp'], datetime.fromtimestamp(1791768274, timezone.utc).isoformat())

    def test_sensor_transport_dispatch(self):
        with patch.object(DynamoSensorClient, 'fetch', return_value={'status': 'ok'}) as fetch:
            self.assertEqual(SensorClient(Settings(sensor_source='dynamodb')).fetch(), {'status': 'ok'})
            fetch.assert_called_once()

    def test_environment_mapping_configuration(self):
        import os
        with patch.dict(os.environ, {'SENSOR_SOURCE':'dynamodb', 'DYNAMODB_DEVICE_ID':'device-1',
                                    'SENSOR_FIELD_MAP': '{"air_temp_c":"temperature"}'}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'.env';p.write_text('')
                s=Settings.from_env(p)
        self.assertEqual(s.sensor_source, 'dynamodb')
        self.assertEqual(s.sensor_field_map['air_temp_c'], 'temperature')


class MonitorTests(unittest.TestCase):
    def make_monitor(self, tmp, sensors=None):
        class Sensors:
            def fetch(inner):
                return DynamoSensorClient(Settings()).normalize([item(incomplete=True)])
        return Monitor(Settings(groq_api_key='never-disclose-key'), sensors=sensors or Sensors(), result_dir=tmp)

    def test_api_reports_blocked_prediction_and_persists_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self.make_monitor(tmp)
            app = create_app(monitor.settings, monitor, start_monitor=False)
            with TestClient(app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                accepted = client.post('/api/run', json={'mode':'prediction'})
                self.assertEqual(accepted.status_code, 202)
                monitor.executor.shutdown(wait=True)
                payload = client.get('/api/status').json()
                self.assertEqual(payload['prediction']['ml_report']['status'], 'blocked')
                self.assertEqual(payload['prediction']['decision_report']['status'], 'needs_measurements')
                self.assertIsNone(payload['running'])
                self.assertNotIn('never-disclose-key', json.dumps(payload))
                self.assertEqual(client.post('/api/run', json={'mode':'delete'}).status_code, 422)
            saved = (Path(tmp)/'latest.json').read_text()
            self.assertNotIn('never-disclose-key', saved)

    def test_concurrent_run_is_rejected(self):
        started = threading.Event(); release = threading.Event()
        class Sensors:
            def fetch(inner):
                started.set(); release.wait(5)
                return DynamoSensorClient(Settings()).normalize([item(incomplete=True)])
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self.make_monitor(tmp, Sensors())
            with TestClient(create_app(monitor.settings, monitor, start_monitor=False)) as client:
                self.assertEqual(client.post('/api/run',json={'mode':'prediction'}).status_code,202)
                self.assertTrue(started.wait(2))
                self.assertEqual(client.post('/api/run',json={'mode':'assessment'}).status_code,409)
                release.set()

    def test_sensor_failure_does_not_leak_exception_or_old_reading(self):
        class Sensors:
            def fetch(inner):
                raise ValueError('secret-in-url')
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self.make_monitor(tmp, Sensors())
            monitor.execute('prediction', 'failure-test')
            payload = monitor.snapshot()
            self.assertEqual(payload['prediction']['sensor_report']['status'], 'error')
            self.assertEqual(payload['prediction']['ml_report']['status'], 'skipped')
            self.assertNotIn('secret-in-url', json.dumps(payload))
            monitor.close()

    def test_decision_cannot_dispatch_hardware(self):
        report = decision_report({'sensor_report':{'status':'ok','missing_fields':['soil_temp_l1_c']},
                                  'ml_report':{'status':'blocked'}})
        self.assertFalse(report['automated_actuation'])
        self.assertEqual(report['execution'],'grower_review')
        self.assertIn('soil_temp_l1_c', report['actions'][0]['action'])


if __name__ == '__main__':
    unittest.main()
