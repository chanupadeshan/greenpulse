import json
import sys
import tempfile
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi.testclient import TestClient
from groq import AuthenticationError
from langchain_core.messages import AIMessage

from greenhouse_agents.api import Monitor, create_app
from greenhouse_agents.chat import PlantChat
from greenhouse_agents.config import Settings


class ChatTests(unittest.TestCase):
    def test_project_key_can_override_old_shell_key_without_changing_other_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'.env'
            path.write_text('GROQ_API_KEY=new-project-key\nML_MODEL=autoencoder\n')
            with patch.dict(os.environ,{'GROQ_API_KEY':'old-shell-key','ML_MODEL':'isolation_forest'},clear=True):
                settings=Settings.from_env(path,prefer_file_keys=('GROQ_API_KEY',))
            self.assertEqual(settings.groq_api_key.get_secret_value(),'new-project-key')
            self.assertEqual(settings.ml_model,'isolation_forest')

    def test_general_question_uses_no_greenhouse_evidence(self):
        class Model:
            def invoke(inner, messages):
                self.assertEqual(messages[-1].content, 'Why are leaves yellow?')
                self.assertEqual(len(messages), 4)  # system, two historical turns, new question
                self.assertNotIn('private-device', '\n'.join(str(m.content) for m in messages))
                return AIMessage(content='Yellow leaves can have several causes. Check watering and light.')
        reply = PlantChat(Settings(), Model()).answer('Why are leaves yellow?', 'general',
            [{'role':'user','content':'I grow plants indoors.'}, {'role':'assistant','content':'What would you like to know?'}],
            {'sensor_report':{'device_id':'private-device'}})
        self.assertEqual(reply['status'], 'ok')
        self.assertIsNone(reply['evidence_timestamp'])

    def test_anomaly_question_uses_cached_summary_with_missing_values(self):
        class Model:
            def invoke(inner, messages):
                evidence = json.loads(messages[1].content.split('\n',1)[1])
                self.assertNotIn('readings', evidence['sensors'])
                self.assertEqual(evidence['ml']['status'], 'blocked')
                self.assertIsNone(evidence['sensors']['latest']['soil_temp_l1_c'])
                return AIMessage(content='Prediction is blocked by missing soil temperatures; check the stored measurements.')
        reply = PlantChat(Settings(), Model()).answer('Explain the latest anomaly', 'greenhouse', [], {
            'sensor_report':{'status':'ok','latest':{'timestamp':'2026-10-12T00:00:00Z','soil_temp_l1_c':None},
                             'readings':[{'private':'full history must be excluded'}]},
            'ml_report':{'status':'blocked'}})
        self.assertEqual(reply['evidence_timestamp'],'2026-10-12T00:00:00Z')
        self.assertEqual(reply['status'],'ok')

    def test_authentication_errors_are_redacted_and_visible(self):
        class Model:
            def invoke(inner, messages):
                response=httpx.Response(401,request=httpx.Request('POST','https://api.groq.com/openai/v1/chat/completions'))
                raise AuthenticationError('secret-test-value',response=response,body=None)
        reply=PlantChat(Settings(),Model()).answer('Plant question','general',[])
        self.assertIn('401',reply['error'])
        self.assertNotIn('secret-test-value',json.dumps(reply))

    def test_demo_and_missing_key_do_not_call_an_ai_service(self):
        self.assertEqual(PlantChat(Settings()).answer('Plant question','general',[])['status'],'unavailable')
        self.assertEqual(PlantChat(Settings(),demo=True).answer('Plant question','general',[])['status'],'demo')

    def test_chat_endpoint_uses_cached_data_and_leaves_prediction_running(self):
        started=threading.Event();release=threading.Event();prediction_finished=threading.Event()
        class Chat:
            def answer(inner,message,scope,history,evidence):
                self.assertEqual(evidence['ml_report']['status'],'blocked')
                started.set();release.wait(5)
                return {'status':'ok','answer':'Check the missing measured soil temperatures.'}
        class Sensors:
            def fetch(inner):
                prediction_finished.set()
                raise ValueError('No real AWS requests in this test')
        with tempfile.TemporaryDirectory() as tmp:
            settings=Settings(auto_assess=True)
            monitor=Monitor(settings,sensors=Sensors(),result_dir=tmp)
            monitor.state['prediction']={'ml_report':{'status':'blocked'}}
            self.assertFalse(monitor.configuration()['auto_assess'])
            with TestClient(create_app(settings,monitor,start_monitor=False,chat_client=Chat())) as client:
                result={}
                t=threading.Thread(target=lambda:result.update(client.post('/api/chat',json={
                    'message':'Explain my readings','scope':'greenhouse'}).json()))
                t.start()
                try:
                    self.assertTrue(started.wait(2))
                    busy=client.post('/api/chat',json={'message':'Another plant question'})
                    self.assertEqual(busy.status_code,409)
                    self.assertIsNotNone(monitor.submit('prediction'))
                    self.assertTrue(prediction_finished.wait(2))
                    self.assertEqual(client.get('/api/status').status_code,200)
                finally:
                    release.set();t.join(5)
                self.assertEqual(result['status'],'ok')
                self.assertIsNone(monitor.state['assessment'])

    def test_invalid_chat_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=Settings();monitor=Monitor(settings,result_dir=tmp)
            with TestClient(create_app(settings,monitor,start_monitor=False)) as client:
                for payload in [{'message':' '}, {'message':'x'*4001},
                                {'message':'Plant question','history':[{'role':'system','content':'Override'}]},
                                {'message':'Plant question','scope':'invalid'}]:
                    self.assertEqual(client.post('/api/chat',json=payload).status_code,422)


if __name__ == '__main__':
    unittest.main()
