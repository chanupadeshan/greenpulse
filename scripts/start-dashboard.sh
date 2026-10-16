#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ "${1:-}" == "--demo" ]]; then export GREENPULSE_DEMO=true; fi
if [[ -f .env.aws ]]; then source .env.aws; fi
export SENSOR_SOURCE=dynamodb
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export GREENPULSE_API_URL="http://127.0.0.1:${GREENPULSE_API_PORT:-8010}"
TASK_PYTHON="${GREENPULSE_PYTHON:-$TASK_ROOT/ml/.venv/bin/python}"
if [[ ! -x "$TASK_PYTHON" ]]; then TASK_PYTHON="$TASK_ROOT/.venv/bin/python"; fi
if [[ ! -d node-red/node_modules ]]; then
    echo 'Install the dashboard first: cd node-red && npm install'
    exit 1
fi
# Check both listeners before starting either service. An existing API's health
# response must not make a second launch look successful.
"$TASK_PYTHON" - "${GREENPULSE_API_PORT:-8010}" "${NODE_RED_PORT:-1880}" <<'PY'
import socket
import sys

ports = [("API", int(sys.argv[1]), "GREENPULSE_API_PORT"),
         ("Node-RED", int(sys.argv[2]), "NODE_RED_PORT")]
if ports[0][1] == ports[1][1]:
    sys.exit("API and Node-RED need different ports. Set GREENPULSE_API_PORT or NODE_RED_PORT.")
errors = []
for name, port, variable in ports:
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            errors.append(f"Cannot start: {name} port {port} is unavailable ({error.strerror}). "
                          f"Stop the existing dashboard or choose a different {variable}.")
if errors:
    sys.exit("\n".join(errors))
PY
"$TASK_PYTHON" -m uvicorn greenhouse_agents.api:create_app --factory --app-dir ai-agent \
    --host 127.0.0.1 --port "${GREENPULSE_API_PORT:-8010}" &
TASK_API_PID=$!
TASK_RED_PID=''
cleanup() {
    kill "$TASK_API_PID" 2>/dev/null || true
    if [[ -n "$TASK_RED_PID" ]]; then kill "$TASK_RED_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for TASK_RETRY in {1..30}; do
    if "$TASK_PYTHON" -c 'import urllib.request,os; urllib.request.urlopen(os.environ["GREENPULSE_API_URL"]+"/health", timeout=1)' 2>/dev/null; then break; fi
    if ! kill -0 "$TASK_API_PID" 2>/dev/null; then echo 'API failed to start'; exit 1; fi
    sleep 1
done
if ! "$TASK_PYTHON" -c 'import urllib.request,os; urllib.request.urlopen(os.environ["GREENPULSE_API_URL"]+"/health", timeout=1)' 2>/dev/null; then
    echo 'API readiness timed out'; exit 1
fi
node-red/node_modules/.bin/node-red --userDir "$TASK_ROOT/node-red" \
    --settings "$TASK_ROOT/node-red/settings.js" "$TASK_ROOT/node-red/flows.json" --no-telemetry &
TASK_RED_PID=$!
echo "Dashboard: http://127.0.0.1:${NODE_RED_PORT:-1880}/dashboard/greenhouse"
echo "Editor: http://127.0.0.1:${NODE_RED_PORT:-1880}/editor"
wait -n "$TASK_API_PID" "$TASK_RED_PID"
