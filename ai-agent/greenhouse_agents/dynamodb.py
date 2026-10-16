"""Read one device's latest records without scanning or substituting missing inputs."""
import math
from datetime import datetime, timezone
from decimal import Decimal

from .schemas import SensorReading

SENSOR_FIELDS = tuple(name for name in SensorReading.model_fields if name != "timestamp")


class DynamoSensorClient:
    def __init__(self, settings, table=None):
        self.settings = settings
        self.table = table

    def fetch(self):
        if not self.settings.dynamodb_device_id:
            raise ValueError("Set DYNAMODB_DEVICE_ID; readings from different devices must not be mixed")
        if self.table is None:
            import boto3
            from botocore.config import Config
            resource = boto3.resource("dynamodb", region_name=self.settings.aws_region,
                config=Config(connect_timeout=10, read_timeout=self.settings.request_timeout_seconds,
                              retries={"max_attempts": 2, "mode": "standard"}))
            self.table = resource.Table(self.settings.dynamodb_table)
        from boto3.dynamodb.conditions import Key
        # Descending Query returns the newest records for this device. Limit is bounded.
        response = self.table.query(
            KeyConditionExpression=Key("device_id").eq(self.settings.dynamodb_device_id),
            ScanIndexForward=False, Limit=self.settings.dynamodb_read_limit, ConsistentRead=True,
        )
        return self.normalize(response.get("Items", []))

    def normalize(self, items):
        if not items:
            raise ValueError("No sensor records found for the configured device")
        rows = []
        for item in items:
            epoch = float(item["timestamp"])
            if self.settings.dynamodb_timestamp_unit == "ms":
                epoch /= 1000
            row = {"timestamp": datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()}
            for name in SENSOR_FIELDS:
                value = item.get(self.settings.sensor_field_map.get(name, name))
                if value is None:
                    row[name] = None
                elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    row[name] = float(value)
                else:
                    raise ValueError(f"Invalid numeric sensor field: {name}")
            rows.append(row)
        rows.sort(key=lambda row: row["timestamp"])
        if len({row["timestamp"] for row in rows}) != len(rows):
            raise ValueError("Duplicate sensor timestamps")
        latest = rows[-1]
        missing = [name for name in SENSOR_FIELDS if latest[name] is None]
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(latest["timestamp"])).total_seconds() / 60
        warnings = []
        if missing:
            warnings.append("Model inputs are missing: " + ", ".join(missing) + "; no values will be substituted")
        if age > self.settings.max_sensor_age_minutes:
            warnings.append("Sensor readings are stale; collect fresh readings before acting")
        if age < -5:
            warnings.append("Sensor timestamp is in the future; verify device clock")
        for field in ("humidity", "soil_moisture_l1", "soil_moisture_l2"):
            value = latest[field]
            if value is not None and not 0 <= value <= 100:
                warnings.append(f"{field} is outside 0–100%; verify calibration")
            elif field.startswith("soil_moisture") and value == 100:
                warnings.append(f"{field} reports 100%; confirm wet/dry calibration before using it for irrigation advice")
        return {"status": "ok", "source": "dynamodb", "table": self.settings.dynamodb_table,
                "device_id": self.settings.dynamodb_device_id, "readings": rows, "latest": latest,
                "row_count": len(rows), "age_minutes": round(age, 2), "missing_fields": missing,
                "warnings": warnings,
                "units": {"temperature": "°C", "humidity": "%", "soil_moisture": "%"}}
