"""Build an email from the cached prediction and current outdoor weather."""

from datetime import datetime, timezone
from html import escape
from zoneinfo import ZoneInfo


SENSOR_FIELDS = (
    ("Air temperature", "air_temp_c", " °C"), ("Humidity", "humidity", " %"),
    ("Soil moisture L1", "soil_moisture_l1", " %"), ("Soil moisture L2", "soil_moisture_l2", " %"),
    ("Soil temperature L1", "soil_temp_l1_c", " °C"), ("Soil temperature L2", "soil_temp_l2_c", " °C"),
)


def display_value(mapping, key, unit=""):
    result = mapping.get(key)
    if result is None:
        return "unavailable"
    if isinstance(result, float):
        result = f"{result:.4f}" if key in ("score", "threshold") else f"{result:.2f}"
    return f"{result}{unit}"


def situation_summary(snapshot, weather):
    # Summarize only supplied measurements; no LLM call or invented care targets.
    prediction = snapshot["prediction"]
    latest = prediction.get("sensor_report", {}).get("latest", {})
    model = prediction.get("ml_report", {})
    if model.get("status") != "ok":
        headline = "Prediction unavailable"
        summary = "The model could not determine whether the latest readings contain an anomaly. Check the measurement warnings below."
    elif model.get("is_anomaly"):
        headline = "Unusual sensor pattern detected"
        summary = "The anomaly score is above the active alert limit. Inspect the readings and sensor calibration before changing growing conditions."
    else:
        headline = "No anomaly detected by the model"
        summary = "The latest anomaly score is within the active alert limit. This describes the sensor pattern and does not confirm plant health."
    paragraphs = [summary,
        f"Air temperature is {display_value(latest, 'air_temp_c', ' °C')} and humidity is {display_value(latest, 'humidity', ' %')}. "
        f"Soil moisture is {display_value(latest, 'soil_moisture_l1', ' %')} at L1 and {display_value(latest, 'soil_moisture_l2', ' %')} at L2."]
    if weather.get("status") == "ok":
        conditions = weather.get("conditions", {})
        paragraphs.append(f"Outdoor weather: {display_value(conditions, 'description')}, "
                          f"{display_value(conditions, 'air_temp_c', ' °C')}, with "
                          f"{display_value(conditions, 'rain_last_hour_mm', ' mm')} rain reported in the last hour.")
    else:
        paragraphs.append("Current outdoor weather is unavailable for this report.")
    return headline, paragraphs


def measurement_warnings(sensors, weather):
    warnings = list(sensors.get("warnings", [])) + list(weather.get("warnings", []))
    if weather.get("error"):
        warnings.append(weather["error"])
    missing = [label for label, key, _ in SENSOR_FIELDS if sensors.get("latest", {}).get(key) is None]
    if missing:
        warnings.append("Missing measurements: " + ", ".join(missing) + ".")
    age = None
    try:
        timestamp = datetime.fromisoformat(sensors["latest"]["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Timestamp needs a timezone")
        age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
        if age > 2:
            warnings.append("Sensor reading is older than two minutes; check whether fresh measurements are arriving.")
        if age < -5:
            warnings.append("Sensor reading is future-dated; check the device clock.")
    except (KeyError, TypeError, ValueError):
        warnings.append("Sensor reading time is unavailable or invalid.")
    return age, list(dict.fromkeys(warnings))


def report_body(snapshot, weather, latitude, longitude):
    prediction = snapshot["prediction"]
    sensors = prediction.get("sensor_report", {})
    latest = sensors.get("latest", {})
    model = prediction.get("ml_report", {})
    conditions = weather.get("conditions", {})
    headline, summary = situation_summary(snapshot, weather)

    def value(mapping, key, unit=""):
        result = mapping.get(key)
        return "unavailable" if result is None else f"{result}{unit}"

    lines = [
        "GreenPulse - latest monitoring report",
        f"Report time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Prediction run: {prediction['id']}",
        f"Prediction completed: {prediction.get('completed_at', 'unavailable')}",
        f"Sensor reading time: {latest.get('timestamp', 'unavailable')}",
        f"Location: latitude {latitude}, longitude {longitude}",
    ]
    if snapshot.get("config", {}).get("demo"):
        lines += ["", "SAMPLE: all sensor readings, model results, and weather are simulated."
                  if weather.get("source") == "sample" else "DEMO: greenhouse readings are simulated; weather is requested separately."]
    lines += ["", "SITUATION SUMMARY", headline, *summary]
    lines += ["", "CURRENT OUTDOOR WEATHER", f"Status: {weather.get('status', 'unavailable')}"]
    for label, key, unit in (
        ("Temperature", "air_temp_c", " °C"), ("Humidity", "humidity", " %"),
        ("Wind speed", "wind_speed_m_s", " m/s"), ("Rain in last hour", "rain_last_hour_mm", " mm"),
        ("Conditions", "description", ""), ("Observation time", "observed_at", ""),
    ):
        lines.append(f"{label}: {value(conditions, key, unit)}")
    lines += ["", "GREENHOUSE READINGS", f"Status: {sensors.get('status', 'unavailable')}"]
    for label, key, unit in SENSOR_FIELDS:
        lines.append(f"{label}: {value(latest, key, unit)}")
    lines += ["", "MODEL RESULT", f"Status: {model.get('status', 'unavailable')}"]
    if model.get("status") == "ok":
        for label, key in (("Model", "model"), ("Score", "score"), ("Alert limit", "threshold"), ("Severity", "severity")):
            lines.append(f"{label}: {value(model, key)}")
        lines.append("Anomaly detected: " + ("Yes" if model.get("is_anomaly") else "No"))
    else:
        lines.append("Anomaly detected: unavailable")
        lines.append(model.get("error", "Prediction unavailable; check measured inputs."))
    age, warnings = measurement_warnings(sensors, weather)
    if age is not None:
        lines.append(f"\nSensor reading age: {age:.1f} minutes")
    lines += ["", "WARNINGS"] + (warnings or ["No additional warnings reported."])
    lines += ["", "This is the latest cached result, not an average over the reporting interval.",
              "Outdoor current weather is not a forecast or an indoor measurement.",
              "An anomaly score is a pattern alert, not a disease diagnosis or probability."]
    return "\n".join(lines)


def report_html(snapshot, weather, latitude, longitude):
    # Tables and inline styles work in email clients without scripts or remote images.
    prediction = snapshot["prediction"]
    sensors = prediction.get("sensor_report", {})
    latest = sensors.get("latest", {})
    model = prediction.get("ml_report", {})
    conditions = weather.get("conditions", {})
    headline, summary = situation_summary(snapshot, weather)
    age, warnings = measurement_warnings(sensors, weather)
    e = lambda value: escape(str(value), quote=True)
    zone_name = snapshot.get("config", {}).get("timezone", "Asia/Colombo")
    try:
        zone = ZoneInfo(zone_name)
    except (KeyError, ValueError, TypeError):
        zone_name, zone = "UTC", timezone.utc

    def timestamp(value):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return "unavailable"
            return parsed.astimezone(zone).strftime("%d %b %Y, %H:%M:%S")
        except (AttributeError, TypeError, ValueError):
            return "unavailable"

    good = model.get("status") == "ok" and not model.get("is_anomaly")
    color = "#21664c" if good else "#9b4b14"
    tint = "#eef7f1" if good else "#fff4e5"
    badge = "Not available" if model.get("status") != "ok" else "Yes" if model.get("is_anomaly") else "No"
    generated = datetime.now(zone).strftime("%d %b %Y · %H:%M")
    cards = []
    for index in range(0, len(SENSOR_FIELDS), 2):
        cells = ""
        for label, key, unit in SENSOR_FIELDS[index:index + 2]:
            cells += f'''<td width="50%" valign="top" style="padding:7px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7f5;border:1px solid #e3ebe6;border-radius:10px;"><tr><td style="padding:16px;"><div style="font-size:12px;color:#5e7168;">{e(label)}</div><div style="font-size:25px;font-weight:bold;color:#183f30;padding-top:7px;">{e(display_value(latest, key, unit))}</div></td></tr></table></td>'''
        cards.append(f"<tr>{cells}</tr>")

    def rows(items):
        return "".join(f'<tr><td style="padding:10px 0;border-bottom:1px solid #e6ede8;color:#617269;font-size:13px;">{e(label)}</td><td align="right" style="padding:10px 0 10px 12px;border-bottom:1px solid #e6ede8;font-size:13px;font-weight:bold;">{e(value)}</td></tr>' for label, value in items)

    weather_rows = rows([
        ("Conditions", display_value(conditions, "description")),
        ("Temperature", display_value(conditions, "air_temp_c", " °C")),
        ("Humidity", display_value(conditions, "humidity", " %")),
        ("Wind", display_value(conditions, "wind_speed_m_s", " m/s")),
        ("Rain · last hour", display_value(conditions, "rain_last_hour_mm", " mm")),
        ("Observed", timestamp(conditions.get("observed_at"))),
    ])
    model_rows = rows([
        ("Anomaly detected", badge), ("Model", display_value(model, "model")),
        ("Score", display_value(model, "score") if model.get("status") == "ok" else "unavailable"),
        ("Active alert limit", display_value(model, "threshold") if model.get("status") == "ok" else "unavailable"),
        ("Severity", display_value(model, "severity") if model.get("status") == "ok" else "unavailable"),
    ])
    warning_html = "".join(f'<li style="margin:7px 0;">{e(warning)}</li>' for warning in warnings)
    checks = (f'<div style="padding:18px;background:#fff7e9;border:1px solid #f1dfbb;border-radius:10px;"><strong style="color:#82551b;">Measurement &amp; weather checks</strong><ul style="padding-left:19px;margin-bottom:0;font-size:13px;line-height:1.6;">{warning_html}</ul></div>'
              if warnings else '<p style="font-size:13px;color:#617269;">No additional measurement warnings reported.</p>')
    demo = '<p style="background:#fff1d6;padding:12px;font-size:13px;"><strong>DEMO REPORT</strong> · Greenhouse readings are simulated.</p>' if snapshot.get("config", {}).get("demo") else ""
    if snapshot.get("config", {}).get("demo") and weather.get("source") == "sample":
        demo = '<p style="background:#fff1d6;padding:12px;font-size:13px;line-height:1.6;"><strong>SAMPLE REPORT</strong> · All sensor readings, model results, and weather are simulated.</p>'
    paragraphs = "".join(f'<p style="margin:10px 0 0;font-size:14px;line-height:1.7;">{e(paragraph)}</p>' for paragraph in summary)
    error = f'<p style="font-size:13px;color:#82551b;">{e(model.get("error", ""))}</p>' if model.get("status") != "ok" else ""
    age_text = f"{age:.1f} minutes" if age is not None else "unavailable"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GreenPulse situation report</title></head>
<body style="margin:0;padding:0;background:#edf2ee;color:#243f32;font-family:Arial,Helvetica,sans-serif;">
<div style="display:none;max-height:0;overflow:hidden;">{e(headline)}. Latest greenhouse readings and outdoor weather.</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:28px 12px;">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="width:100%;max-width:640px;background:#fff;border:1px solid #dce6df;border-radius:16px;">
<tr><td style="padding:30px 26px;background:#173f30;border-radius:16px 16px 0 0;color:#fff;">
<div style="font-size:11px;letter-spacing:3px;color:#b8d4c2;">GREENPULSE AI</div>
<h1 style="font-size:28px;line-height:1.2;margin:12px 0;">Your greenhouse, at a glance.</h1>
<div style="font-size:12px;color:#d4e4d9;">{e(generated)} · {e(zone_name)}</div></td></tr>
<tr><td style="padding:24px 26px;">{demo}
<div style="background:{tint};border-left:4px solid {color};padding:20px;border-radius:8px;">
<div style="font-size:11px;letter-spacing:1px;color:{color};font-weight:bold;">SITUATION SUMMARY</div>
<h2 style="font-size:20px;line-height:1.4;margin:9px 0;color:{color};">{e(headline)}</h2>{paragraphs}</div>
<h2 style="font-size:18px;margin:27px 0 6px;">Greenhouse measurements</h2>
<p style="font-size:12px;color:#617269;line-height:1.7;margin-top:0;">Measured: {e(timestamp(latest.get('timestamp')))} · {e(zone_name)}<br>Reading age: {e(age_text)}</p>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{''.join(cards)}</table>
<h2 style="font-size:18px;margin:26px 0 8px;">Anomaly assessment</h2>
<table width="100%" cellpadding="0" cellspacing="0">{model_rows}</table>{error}
<h2 style="font-size:18px;margin:26px 0 4px;">Current outdoor weather</h2>
<p style="font-size:12px;color:#617269;line-height:1.6;">Location: {e(latitude)}, {e(longitude)} · Status: {e(weather.get('status', 'unavailable'))}</p>
<table width="100%" cellpadding="0" cellspacing="0">{weather_rows}</table>
<div style="margin-top:24px;">{checks}</div>
<p style="font-size:11px;color:#73847b;line-height:1.7;margin:24px 0 0;">Latest cached situation, not an average over the reporting interval. Outdoor weather is not a forecast or an indoor measurement. An anomaly is a sensor-pattern alert, not a disease diagnosis.</p>
</td></tr><tr><td style="padding:18px 26px;border-top:1px solid #e3ebe6;font-size:11px;color:#73847b;line-height:1.7;">GREENPULSE · Plant monitoring<br>Prediction completed: {e(timestamp(prediction.get('completed_at')))} · {e(zone_name)}<br>Run: {e(prediction.get('id', 'unavailable'))}</td></tr>
</table></td></tr></table></body></html>'''
