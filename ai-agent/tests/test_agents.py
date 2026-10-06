import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from pydantic import ValidationError

from greenhouse_agents.config import PROJECT_ROOT, Settings
from greenhouse_agents.demo import DemoSensorClient, DemoWeatherClient
from greenhouse_agents.master import MasterClient, build_chain, build_context
from greenhouse_agents.ml import MLClient
from greenhouse_agents.sensors import SensorClient
from greenhouse_agents.weather import WeatherClient
from greenhouse_agents.workflow import build_workflow, initial_state


def reading(timestamp=None):
    return {"timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "air_temp_c": 30.2, "humidity": 78.5, "soil_moisture_l1": 35.4,
            "soil_temp_l1_c": 26.8, "soil_moisture_l2": 46.5, "soil_temp_l2_c": 26.7}


def weather_payload():
    return {"main": {"temp": 29.5, "humidity": 80}, "wind": {"speed": 2.5},
            "weather": [{"description": "light rain"}], "dt": int(datetime.now(timezone.utc).timestamp())}


class FakeML:
    def analyze(self, readings):
        assert readings
        return {"status": "ok", "score": 0.2, "threshold": 0.1, "is_anomaly": True}


class CaptureMaster:
    def __init__(self):
        self.calls = []

    def advise(self, state):
        self.calls.append(state.copy())
        return {"status": "ok", "recommendations": []}


class AgentTests(unittest.TestCase):
    def test_environment_parsing_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.env'
            path.write_text('GROQ_API_KEY=test-key\nML_MODEL=autoencoder\nSENSOR_CSV_PATH=readings.csv\n')
            with patch.dict(os.environ, {'ML_MODEL': 'isolation_forest'}, clear=True):
                settings = Settings.from_env(path)
            self.assertEqual(settings.ml_model, 'isolation_forest')
            self.assertEqual(settings.sensor_csv_path, PROJECT_ROOT / 'readings.csv')
            self.assertNotIn('test-key', repr(settings))
            self.assertIsNone(settings.latitude)

    def test_sensor_timezone_and_invalid_readings(self):
        client = SensorClient(Settings())
        report = client.normalize([reading('2026-10-06 12:00:00')], 'csv')
        self.assertEqual(report['latest']['timestamp'], '2026-10-06T06:30:00Z')
        for rows in [[], [reading(), reading('bad-date')], [reading(), reading()]]:
            if len(rows) == 2 and rows[1]['timestamp'] != 'bad-date':
                rows[1]['timestamp'] = rows[0]['timestamp']
            with self.assertRaises(ValueError):
                client.normalize(rows, 'csv')
        invalid = reading()
        invalid['humidity'] = float('nan')
        with self.assertRaises(ValidationError):
            client.normalize([invalid], 'csv')

    def test_stale_sensor_data_is_explicit(self):
        report = SensorClient(Settings()).normalize(
            [reading((datetime.now(timezone.utc) - timedelta(days=2)).isoformat())], 'csv')
        self.assertTrue(any('stale' in warning for warning in report['warnings']))

    def test_http_sensor_contract_and_secret_redaction(self):
        def handler(request):
            self.assertEqual(request.headers['Authorization'], 'Bearer secret-token')
            return httpx.Response(200, json={'readings': [reading()]})
        settings = Settings(sensor_source='http', sensor_api_url='https://sensors.test/latest',
                            sensor_api_key='secret-token')
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            report = SensorClient(settings, client).fetch()
        self.assertEqual(report['row_count'], 1)
        self.assertNotIn('secret-token', json.dumps(report))

    def test_openweather_units_coordinates_and_missing_config(self):
        def handler(request):
            self.assertEqual(request.url.params['units'], 'metric')
            self.assertEqual(request.url.params['lat'], '7.0')
            self.assertEqual(request.url.params['appid'], 'weather-key')
            return httpx.Response(200, json=weather_payload())
        settings = Settings(weather_api_key='weather-key', latitude=7, longitude=80)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            report = WeatherClient(settings, client).fetch()
        self.assertEqual(report['conditions']['rain_last_hour_mm'], 0)
        self.assertNotIn('weather-key', json.dumps(report))
        with self.assertRaises(ValueError):
            WeatherClient(Settings()).fetch()
        with self.assertRaises(ValueError):
            WeatherClient(Settings(weather_api_key='key')).fetch()

    def test_graph_joins_both_branches_and_calls_master_once(self):
        settings = Settings()
        capture = CaptureMaster()
        class SlowWeather:
            def fetch(self):
                time.sleep(0.05)
                return WeatherClient.normalize(weather_payload())
        graph = build_workflow(settings, sensor_client=DemoSensorClient(settings),
                               weather_client=SlowWeather(), ml_client=FakeML(), master_client=capture)
        result = graph.invoke(initial_state(settings))
        self.assertEqual(len(capture.calls), 1)
        self.assertEqual(capture.calls[0]['ml_report']['status'], 'ok')
        self.assertEqual(capture.calls[0]['weather_report']['status'], 'ok')
        self.assertCountEqual(result['history'], ['sensor_agent', 'weather_agent', 'ml_agent', 'master_agent'])

    def test_sensor_failure_skips_ml_and_keeps_weather(self):
        class BrokenSensor:
            def fetch(self):
                raise httpx.RequestError('private-api-key-in-url')
        settings = Settings()
        graph = build_workflow(settings, sensor_client=BrokenSensor(), weather_client=DemoWeatherClient(),
                               ml_client=FakeML(), dry_run=True)
        result = graph.invoke(initial_state(settings))
        self.assertEqual(result['ml_report']['status'], 'skipped')
        self.assertEqual(result['weather_report']['status'], 'ok')
        self.assertEqual(result['master_report']['status'], 'dry_run')
        self.assertNotIn('private-api-key-in-url', json.dumps(result))

    def test_master_structured_prompt_uses_all_sources(self):
        class Chain:
            def invoke(inner, values):
                context = json.loads(values['context'])
                self.assertEqual(context['crop_context']['country'], 'Sri Lanka')
                self.assertIn('ml', context)
                self.assertIn('weather', context)
                self.assertNotIn('readings', context['sensors'])
                return {'summary': 'Check the sensors', 'recommendations': [{
                    'priority': 'medium', 'action': 'Verify moisture sensor calibration',
                    'reason': 'Model reports an unusual pattern', 'evidence_sources': ['ml', 'sensors']}],
                    'missing_information': ['crop'], 'cautions': ['No labeled anomaly data']}
        settings = Settings()
        state = initial_state(settings)
        state.update(sensor_report=DemoSensorClient(settings).fetch(), ml_report=FakeML().analyze([reading()]),
                     weather_report=DemoWeatherClient().fetch())
        report = MasterClient(settings, chain=Chain()).advise(state)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(len(report['recommendations']), 1)
        self.assertEqual(MasterClient(settings).advise(state)['status'], 'unavailable')

    def test_groq_chain_parses_tool_output_with_mocked_http(self):
        from langchain_groq import ChatGroq
        advice = {"summary": "Collect crop context", "recommendations": [],
                  "missing_information": ["crop name"], "cautions": ["Example measurements"]}
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["model"], "openai/gpt-oss-120b")
            self.assertEqual(body["tools"][0]["function"]["name"], "PlantAdvice")
            self.assertIn("sensor_report", json.loads(body["messages"][1]["content"].split("\n", 1)[1]))
            return httpx.Response(200, json={
                "id": "chat-test", "object": "chat.completion", "created": 1,
                "model": "openai/gpt-oss-120b", "choices": [{"index": 0,
                "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call-test", "type": "function", "function": {
                    "name": "PlantAdvice", "arguments": json.dumps(advice)}}]}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            model = ChatGroq(model="openai/gpt-oss-120b", api_key="test-groq-key", http_client=client)
            output = build_chain(Settings(), model).invoke({"context": json.dumps({"sensor_report": {"status": "ok"}})})
        self.assertEqual(output.summary, advice["summary"])
        self.assertEqual(output.missing_information, ["crop name"])

    def test_misaligned_observations_warn_master(self):
        settings = Settings()
        state = initial_state(settings)
        state.update(sensor_report=SensorClient(settings).normalize([reading('2020-01-01T00:00:00Z')], 'csv'),
                     weather_report=DemoWeatherClient().fetch())
        context = build_context(state)
        self.assertTrue(any('two hours apart' in warning for warning in context['data_quality_warnings']))

    @unittest.skipUnless((PROJECT_ROOT / "ml/models/autoencoder/model.pt").exists(), "Restore or train dense model artifacts first")
    def test_real_dense_autoencoder_integration(self):
        settings = Settings()
        sensors = DemoSensorClient(settings).fetch()
        result = MLClient(settings).analyze(sensors['readings'])
        self.assertEqual(result['model'], 'autoencoder')
        self.assertGreaterEqual(result['score'], 0)
        self.assertEqual(result['is_anomaly'], result['score'] > result['threshold'])


if __name__ == '__main__':
    unittest.main()
