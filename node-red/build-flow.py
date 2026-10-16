"""Generate the importable flow from the reviewable Vue dashboard source."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLOW = "greenpulse-flow"


def node(id, type, name, x, y, wires, **props):
    return {"id": id, "type": type, "z": FLOW, "name": name, "x": x, "y": y, "wires": wires, **props}


nodes = [
    {"id": FLOW, "type": "tab", "label": "GreenPulse live agents", "disabled": False,
     "info": "Prediction refreshes continuously. The separate AI assistant answers submitted questions only. Secrets stay in Python."},
    {"id": "gp-base", "type": "ui-base", "name": "GreenPulse", "path": "/dashboard", "appIcon": "mdi-sprout",
     "includeClientData": True, "acceptsClientConfig": ["ui-notification", "ui-control", "ui-template"],
     "showPathInSidebar": False, "headerContent": "page", "navigationStyle": "none", "titleBarStyle": "hidden"},
    {"id": "gp-theme", "type": "ui-theme", "name": "GreenPulse light", "colors": {
        "surface": "#f6f8f5", "primary": "#217454", "bgPage": "#f6f8f5", "groupBg": "#f6f8f5", "groupOutline": "#f6f8f5"},
     "sizes": {"density": "default", "pagePadding": "0px", "groupGap": "0px", "groupBorderRadius": "0px", "widgetGap": "0px"}},
    {"id": "gp-page", "type": "ui-page", "name": "Greenhouse", "ui": "gp-base", "path": "/greenhouse",
     "icon": "mdi-sprout", "layout": "grid", "theme": "gp-theme", "order": 1, "visible": True, "disabled": False,
     "breakpoints": [{"name": "Default", "px": "0", "cols": "12"}, {"name": "Tablet", "px": "576", "cols": "12"},
                     {"name": "Small Desktop", "px": "768", "cols": "12"}, {"name": "Desktop", "px": "1024", "cols": "12"}]},
    {"id": "gp-group", "type": "ui-group", "name": "Greenhouse intelligence", "page": "gp-page", "width": "12",
     "height": "1", "order": 1, "showTitle": False, "visible": True, "disabled": False},
    {"id": "gp-ai-page", "type": "ui-page", "name": "AI assistant", "ui": "gp-base", "path": "/assistant",
     "icon": "mdi-chat", "layout": "grid", "theme": "gp-theme", "order": 2, "visible": True, "disabled": False,
     "breakpoints": [{"name": "Default", "px": "0", "cols": "12"}, {"name": "Tablet", "px": "576", "cols": "12"},
                     {"name": "Small Desktop", "px": "768", "cols": "12"}, {"name": "Desktop", "px": "1024", "cols": "12"}]},
    {"id": "gp-ai-group", "type": "ui-group", "name": "Plant conversation", "page": "gp-ai-page", "width": "12",
     "height": "1", "order": 1, "showTitle": False, "visible": True, "disabled": False},
    node("gp-poll", "inject", "Poll status every 5 sec", 170, 100, [["gp-status-url"]],
         props=[{"p": "payload"}, {"p": "topic", "vt": "str"}], repeat="5", crontab="", once=True,
         onceDelay="1", topic="", payload="", payloadType="date"),
    node("gp-status-url", "function", "GET backend status", 410, 100, [["gp-get"]], outputs=1, func=
         'msg.url = (env.get("GREENPULSE_API_URL") || "http://127.0.0.1:8010") + "/api/status";\n'
         'msg.method = "GET"; delete msg.payload; return msg;'),
    node("gp-get", "http request", "Read live state", 640, 100, [["gp-status-format"]], method="use", ret="obj",
         paytoqs="ignore", url="", tls="", persist=False, proxy="", insecureHTTPParser=False,
         senderr=False, headers=[], requestTimeout="15000"),
    node("gp-status-format", "function", "Status / offline", 870, 100, [["gp-dashboard", "gp-assistant"]], outputs=1, func=
         'if(msg.statusCode === 200 && msg.payload && msg.payload.config) { msg.topic="status"; delete msg._client; return msg; }\n'
         'msg.topic="request-error"; msg.payload={error:"Backend unavailable. Check the API process.",disconnected:true}; return msg;'),
    node("gp-dashboard", "ui-template", "GreenPulse dashboard", 1100, 200, [["gp-run-url"]], group="gp-group",
         page="", ui="", order=1, width="12", height="0", head="", format=ROOT.joinpath("dashboard.vue").read_text(),
         storeOutMessages=True, passthru=False, resendOnRefresh=True, templateScope="local", className="gp-widget"),
    node("gp-assistant", "ui-template", "Plant assistant chat", 1100, 480, [["gp-chat-url"]], group="gp-ai-group",
         page="", ui="", order=1, width="12", height="0", head="", format=ROOT.joinpath("assistant.vue").read_text(),
         storeOutMessages=False, passthru=False, resendOnRefresh=False, templateScope="local", className="gp-chat-widget"),
    node("gp-run-url", "function", "Request prediction / AI", 400, 260, [["gp-post"]], outputs=1, func=
         'if (!["prediction", "assessment"].includes(msg.topic)) return null;\n'
         'msg.url = (env.get("GREENPULSE_API_URL") || "http://127.0.0.1:8010") + "/api/run";\n'
         'msg.method="POST"; msg.headers={"Content-Type":"application/json"}; msg.payload={mode:msg.topic}; return msg;'),
    node("gp-post", "http request", "Start background run", 640, 260, [["gp-run-format"]], method="use", ret="obj",
         paytoqs="ignore", url="", tls="", persist=False, proxy="", insecureHTTPParser=False,
         senderr=False, headers=[], requestTimeout="15000"),
    node("gp-run-format", "function", "Run accepted / error", 870, 260, [["gp-dashboard"], ["gp-status-url"]], outputs=2, func=
         'if(msg.statusCode === 202) { msg.topic="accepted"; return [msg, {payload:null}]; }\n'
         'msg.topic="request-error"; msg.payload={error:msg.statusCode===409 ? "An analysis is already running." : "Could not start analysis. Check the API."}; return [msg,null];'),
    node("gp-catch", "catch", "Backend connection errors", 630, 380, [["gp-error"]], scope=["gp-get", "gp-post"], uncaught=False),
    node("gp-error", "function", "Show disconnected state", 870, 380, [["gp-dashboard", "gp-assistant"]], outputs=1,
         func='msg.topic="request-error"; msg.payload={error:"Backend connection failed. Start scripts/start-dashboard.sh.",disconnected:true}; return msg;'),
    node("gp-chat-url", "function", "Ask plant assistant", 400, 480, [["gp-chat-post"]], outputs=1, func=
         'if (msg.topic !== "chat" || typeof msg.payload?.message !== "string") return null;\n'
         'msg.url = (env.get("GREENPULSE_API_URL") || "http://127.0.0.1:8010") + "/api/chat";\n'
         'msg.method="POST"; msg.headers={"Content-Type":"application/json"}; msg.requestTimeout=120000; return msg;'),
    node("gp-chat-post", "http request", "Receive AI answer", 640, 480, [["gp-chat-format"]], method="use", ret="obj",
         paytoqs="ignore", url="", tls="", persist=False, proxy="", insecureHTTPParser=False,
         senderr=False, headers=[], requestTimeout="120000"),
    node("gp-chat-format", "function", "Answer / visible error", 870, 480, [["gp-assistant"]], outputs=1, func=
         'msg.topic="chat-reply";\n'
         'if(msg.statusCode !== 200) msg.payload={status:"error",error:msg.statusCode===409 ? "The assistant is busy. Try again shortly." : "Could not answer. Check the API connection and question length."};\n'
         'return msg;'),
    node("gp-chat-catch", "catch", "Chat connection errors", 650, 600, [["gp-chat-error"]], scope=["gp-chat-post"], uncaught=False),
    node("gp-chat-error", "function", "Chat request failed", 870, 600, [["gp-assistant"]], outputs=1,
         func='msg.topic="chat-reply"; msg.payload={status:"error",error:"The assistant request failed or timed out. Please retry."}; return msg;'),
]
ROOT.joinpath("flows.json").write_text(json.dumps(nodes, indent=2) + "\n")
print("Built flows.json")
