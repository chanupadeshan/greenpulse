"""Evidence-driven follow-up plan. Advice never dispatches actuator commands."""
def decision_report(state):
    sensor = state.get("sensor_report", {})
    ml = state.get("ml_report", {})
    master = state.get("master_report", {})
    actions = []
    if sensor.get("status") != "ok":
        actions.append({"priority": "high", "action": "Restore sensor access and confirm the configured device ID."})
    if sensor.get("missing_fields"):
        actions.append({"priority": "high", "action": "Restore measured values for: " + ", ".join(sensor["missing_fields"])})
    if sensor.get("age_minutes", 0) > 120 or sensor.get("age_minutes", 0) < -5:
        actions.append({"priority": "high", "action": "Collect fresh readings and check the device clock."})
    if ml.get("status") == "ok" and ml.get("is_anomaly"):
        actions.append({"priority": "high", "action": "Inspect the greenhouse and verify sensor calibration before changing growing conditions."})
    if "weather_report" in state and state["weather_report"].get("status") != "ok":
        actions.append({"priority": "low", "action": "Configure greenhouse coordinates and weather access for outdoor context."})
    if master.get("status") == "ok":
        actions.extend({"priority": r["priority"], "action": r["action"]} for r in master.get("recommendations", []))
    elif "master_report" in state and master.get("status") != "dry_run":
        actions.append({"priority": "low", "action": "Configure GROQ_API_KEY to enable AI recommendations."})
    blocked = sensor.get("status") != "ok" or ml.get("status") != "ok" or bool(sensor.get("warnings"))
    return {"status": "needs_measurements" if blocked else "ready_for_review", "actions": actions,
            "execution": "grower_review", "automated_actuation": False}
