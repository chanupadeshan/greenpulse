from .state import GreenhouseState


def master_error_report(error):
    error_type = type(error).__name__
    messages = {
        "AuthenticationError": "Groq rejected the API key (401). Replace GROQ_API_KEY in .env with a valid Groq key, then restart the backend.",
        "PermissionDeniedError": "Groq denied access (403). Check that the API key and account can use the configured model.",
        "RateLimitError": "Groq's request or token limit was reached (429). Wait before retrying and check your account limits.",
        "APITimeoutError": "Groq timed out. Retry the assessment and check connectivity.",
        "APIConnectionError": "Could not connect to Groq. Check network connectivity and retry.",
        "NotFoundError": "Groq could not find the configured model. Check GROQ_MODEL in .env and restart the backend.",
    }
    return {"status": "error", "error_type": error_type,
            "error": messages.get(error_type, "Plant recommendation failed; check Groq configuration, availability and output schema")}


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
            report = master_error_report(error)
        return {"master_report": report, "history": ["master_agent"]}

    return {"sensor_agent": sensor_agent, "ml_agent": ml_agent,
            "weather_agent": weather_agent, "master_agent": master_agent}
