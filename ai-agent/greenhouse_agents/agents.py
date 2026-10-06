from .state import GreenhouseState


def build_nodes(sensor_client, ml_client, weather_client, master_client):
    def sensor_agent(state: GreenhouseState):
        try:
            report = sensor_client.fetch()
        except Exception as error:
            # Exception messages can contain request URLs, query keys or raw payloads.
            report = {"status": "error", "error_type": type(error).__name__,
                      "error": "Sensor retrieval failed; check source configuration, timestamp and six sensor fields"}
        return {"sensor_report": report, "history": ["sensor_agent"]}

    def ml_agent(state: GreenhouseState):
        if state["sensor_report"]["status"] != "ok":
            report = {"status": "skipped", "error": "No valid sensor readings available for ML"}
        else:
            try:
                report = ml_client.analyze(state["sensor_report"]["readings"])
            except Exception as error:
                report = {"status": "error", "error_type": type(error).__name__,
                          "error": "ML scoring failed; check trained artifacts and required consecutive readings"}
        return {"ml_report": report, "history": ["ml_agent"]}

    def weather_agent(state: GreenhouseState):
        try:
            report = weather_client.fetch()
        except Exception as error:
            report = {"status": "error", "error_type": type(error).__name__,
                      "error": "Weather retrieval failed; check OpenWeather key, coordinates and connectivity"}
        return {"weather_report": report, "history": ["weather_agent"]}

    def master_agent(state: GreenhouseState):
        try:
            report = master_client.advise(state)
        except Exception as error:
            report = {"status": "error", "error_type": type(error).__name__,
                      "error": "Plant recommendation failed; check Groq configuration, availability and output schema"}
        return {"master_report": report, "history": ["master_agent"]}

    return {"sensor_agent": sensor_agent, "ml_agent": ml_agent,
            "weather_agent": weather_agent, "master_agent": master_agent}
