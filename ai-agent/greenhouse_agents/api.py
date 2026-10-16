"""Loopback API shared by Node-RED and the autonomous monitoring worker."""
import asyncio
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import PROJECT_ROOT, Settings
from .decision import decision_report
from .ml import MLClient
from .sensors import SensorClient
from .workflow import build_workflow, initial_state
from .chat import PlantChat


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunRequest(BaseModel):
    mode: Literal["prediction", "assessment"] = "prediction"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    scope: Literal["general", "greenhouse"] = "general"
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a plant question")
        return value


class Monitor:
    def __init__(self, settings, *, sensors=None, ml=None, workflow=None, result_dir=None):
        self.settings = settings
        self.sensors = sensors or SensorClient(settings)
        self.ml = ml or MLClient(settings)
        self.workflow = workflow
        self.result_dir = Path(result_dir or PROJECT_ROOT / "ai-agent/results/dashboard")
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="greenpulse")
        self.state = {"running": None, "prediction": None, "assessment": None, "history": []}

    def configuration(self):
        artifact_dir = PROJECT_ROOT / "ml/models" / self.settings.ml_model
        weight = "model.joblib" if self.settings.ml_model == "isolation_forest" else "model.pt"
        return {"demo": os.getenv("GREENPULSE_DEMO", "false").lower() == "true",
                "sensor_source": self.settings.sensor_source, "region": self.settings.aws_region,
                "table": self.settings.dynamodb_table, "device_id": self.settings.dynamodb_device_id,
                "model": self.settings.ml_model,
                "ml_threshold_override": self.settings.ml_threshold_override,
                "model_ready": all((artifact_dir / name).is_file() for name in (weight, "metadata.json", "scaler.joblib")),
                "groq_ready": bool(self.settings.groq_api_key.get_secret_value()),
                "weather_ready": bool(self.settings.weather_api_key.get_secret_value()) and
                    self.settings.latitude is not None and self.settings.longitude is not None,
                "crop": self.settings.crop_name, "timezone": self.settings.sensor_timezone,
                "auto_assess": False,
                "prediction_interval_seconds": self.settings.prediction_interval_seconds,
                "advice_only": True}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({**self.state, "config": self.configuration()}, allow_nan=False))

    def submit(self, mode):
        with self.lock:
            if self.state["running"] is not None:
                return None
            run_id = str(uuid.uuid4())
            self.state["running"] = {"id": run_id, "mode": mode, "started_at": utc_now()}
        self.executor.submit(self.execute, mode, run_id)
        return run_id

    def execute(self, mode, run_id):
        report = {"id": run_id, "mode": mode, "started_at": utc_now()}
        try:
            if mode == "assessment":
                workflow = self.workflow or build_workflow(self.settings)
                result = workflow.invoke(initial_state(self.settings))
            else:
                result = initial_state(self.settings)
                try:
                    result["sensor_report"] = self.sensors.fetch()
                except Exception as error:
                    result["sensor_report"] = {"status": "error", "error_type": type(error).__name__,
                        "error": "Sensor retrieval failed. Check AWS access, region, table, device ID, and field mapping."}
                if result["sensor_report"]["status"] == "ok":
                    try:
                        result["ml_report"] = self.ml.analyze(result["sensor_report"]["readings"])
                    except Exception as error:
                        result["ml_report"] = {"status": "error", "error_type": type(error).__name__,
                            "error": "Model inference failed. Check artifacts, calibration, and consecutive hourly inputs for LSTM."}
                else:
                    result["ml_report"] = {"status": "skipped", "error": "No valid sensor readings"}
                # Prediction runs do not call weather or the language model.
                result["decision_report"] = decision_report(result)
                result["history"] = ["sensor_agent", "ml_agent", "decision_agent"]
            report.update(result)
            statuses = [result.get("sensor_report", {}).get("status"), result.get("ml_report", {}).get("status")]
            if mode == "assessment":
                statuses.extend([result.get("weather_report", {}).get("status"), result.get("master_report", {}).get("status")])
            report["status"] = "ok" if all(s == "ok" for s in statuses) else "partial"
        except Exception as error:
            report.update(status="error", error_type=type(error).__name__, error="Workflow failed; check server configuration.")
        report["completed_at"] = utc_now()
        with self.lock:
            self.state[mode] = report
            self.state["history"].insert(0, {key: report[key] for key in ("id", "mode", "status", "completed_at")})
            self.state["history"] = self.state["history"][:30]
            self.state["running"] = None
            payload = json.dumps(self.state, allow_nan=False, indent=2)
        try:
            self.result_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp = self.result_dir / "latest.tmp"
            tmp.write_text(payload)
            tmp.chmod(0o600)
            tmp.replace(self.result_dir / "latest.json")
        except OSError:
            # A persistence failure does not discard the live result.
            with self.lock:
                self.state["storage_warning"] = "Could not save the latest report locally"

    def close(self):
        self.executor.shutdown(wait=True)


def create_app(settings=None, monitor=None, *, start_monitor=True, chat_client=None):
    reload_chat_settings = settings is None
    def load_settings():
        return Settings.from_env(prefer_file_keys=("GROQ_API_KEY",))
    settings = settings or load_settings()
    if monitor is None and os.getenv("GREENPULSE_DEMO", "false").lower() == "true":
        from .demo import DemoSensorClient, DemoWeatherClient
        sensors = DemoSensorClient(settings)
        workflow = build_workflow(settings, sensor_client=sensors, weather_client=DemoWeatherClient(), dry_run=True)
        monitor = Monitor(settings, sensors=sensors, workflow=workflow,
                          result_dir=PROJECT_ROOT / "ai-agent/results/dashboard-demo")
    monitor = monitor or Monitor(settings)
    # Chat has its own worker, so a Groq response cannot stop prediction polling.
    chat_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="greenpulse-chat")
    chat_gate = threading.Lock()

    async def watch():
        while True:
            config = monitor.configuration()
            # AI runs only on explicit requests, regardless of legacy AUTO_ASSESS.
            monitor.submit("prediction")
            await asyncio.sleep(config["prediction_interval_seconds"])

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(watch()) if start_monitor else None
        yield
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(monitor.close)
        await asyncio.to_thread(chat_executor.shutdown, wait=True)

    app = FastAPI(title="GreenPulse agents", lifespan=lifespan)
    app.state.monitor = monitor

    @app.get("/health")
    def health():
        config = monitor.configuration()
        if reload_chat_settings:
            config["groq_ready"] = bool(load_settings().groq_api_key.get_secret_value())
        return {"status": "ok", "config": config}

    @app.get("/api/status")
    def status():
        snapshot = monitor.snapshot()
        if reload_chat_settings:
            snapshot["config"]["groq_ready"] = bool(load_settings().groq_api_key.get_secret_value())
        return snapshot

    @app.post("/api/run", status_code=202)
    def run(request: RunRequest):
        run_id = monitor.submit(request.mode)
        if run_id is None:
            raise HTTPException(409, "An analysis is already running")
        return {"id": run_id, "status": "accepted", "mode": request.mode}

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        if not chat_gate.acquire(blocking=False):
            raise HTTPException(409, "The assistant is answering another question. Try again shortly.")
        try:
            snapshot = monitor.snapshot()
            # Use the continuously updated model report; asking never queries AWS again.
            evidence = snapshot.get("prediction") or {}
            if request.scope == "greenhouse":
                evidence = {**evidence, "crop_context": initial_state(settings)["crop_context"]}
                old_weather = (snapshot.get("assessment") or {}).get("weather_report")
                if old_weather:
                    evidence["weather_report"] = old_weather
            current_settings = load_settings() if reload_chat_settings else settings
            client = chat_client or PlantChat(current_settings,
                demo=os.getenv("GREENPULSE_DEMO", "false").lower() == "true")
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(chat_executor, client.answer, request.message,
                request.scope, [turn.model_dump() for turn in request.history], evidence)
            return answer
        finally:
            chat_gate.release()

    return app
