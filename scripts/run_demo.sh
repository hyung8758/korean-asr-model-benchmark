#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${DEMO_BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${DEMO_BACKEND_PORT:-16000}"
FRONTEND_HOST="${DEMO_FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${DEMO_FRONTEND_PORT:-16010}"
GUNICORN_WORKERS="${DEMO_GUNICORN_WORKERS:-1}"
CONDA_ENV="${DEMO_CONDA_ENV:-korean-asr-benchmark}"
START_WHISPER_CPP="${DEMO_START_WHISPER_CPP:-1}"
WHISPER_CPP_HOST="${DEMO_WHISPER_CPP_HOST:-127.0.0.1}"
WHISPER_CPP_PORT="${DEMO_WHISPER_CPP_PORT:-8100}"
WHISPER_CPP_DEVICE_INDEX="${DEMO_WHISPER_CPP_DEVICE_INDEX:-2}"
WHISPER_CPP_THREADS="${DEMO_WHISPER_CPP_THREADS:-4}"
WHISPER_CPP_PROCESSORS="${DEMO_WHISPER_CPP_PROCESSORS:-1}"
WHISPER_CPP_FLASH_ATTN="${DEMO_WHISPER_CPP_FLASH_ATTN:-0}"
WHISPER_CPP_BINARY="${DEMO_WHISPER_CPP_BINARY:-$PROJECT_ROOT/third_party/whisper.cpp/build/bin/whisper-server}"
WHISPER_CPP_MODEL_PATH="${DEMO_WHISPER_CPP_MODEL_PATH:-$PROJECT_ROOT/third_party/whisper.cpp/models/ggml-large-v3-q5_0.bin}"
RUNTIME_DIR="${DEMO_RUNTIME_DIR:-$PROJECT_ROOT/demo/.runtime}"
LOG_DIR="${DEMO_LOG_DIR:-$RUNTIME_DIR/logs}"
SAVE_DIR="${DEMO_SAVE_DIR:-$RUNTIME_DIR/saved_audio}"
BACKEND_PID="$RUNTIME_DIR/backend.pid"
FRONTEND_PID="$RUNTIME_DIR/frontend.pid"
WHISPER_CPP_PID="$RUNTIME_DIR/whisper_cpp_server.pid"

export DEMO_RUNTIME_DIR="$RUNTIME_DIR"
export DEMO_LOG_DIR="$LOG_DIR"
export DEMO_SAVE_DIR="$SAVE_DIR"
export DEMO_BACKEND_PORT="$BACKEND_PORT"
export DEMO_FRONTEND_PORT="$FRONTEND_PORT"
export DEMO_WHISPER_CPP_SERVER_URL="http://$WHISPER_CPP_HOST:$WHISPER_CPP_PORT/inference"

usage() {
  cat <<EOF
Usage: scripts/run_demo.sh [console|start|stop|restart|status]

Default action is console.

Environment variables:
  DEMO_CONDA_ENV          Default: korean-asr-benchmark
  DEMO_BACKEND_HOST       Default: 0.0.0.0
  DEMO_BACKEND_PORT       Default: 16000
  DEMO_FRONTEND_HOST      Default: 0.0.0.0
  DEMO_FRONTEND_PORT      Default: 16010
  DEMO_GUNICORN_WORKERS   Default: 1
  DEMO_LOG_DIR            Default: demo/.runtime/logs
  DEMO_SAVE_DIR           Default: demo/.runtime/saved_audio
  DEMO_START_WHISPER_CPP  Default: 1
  DEMO_WHISPER_CPP_PORT   Default: 8100
  DEMO_WHISPER_CPP_FLASH_ATTN Default: 0
  DEMO_WHISPER_CPP_MODEL_PATH
EOF
}

ensure_dirs() {
  mkdir -p "$LOG_DIR" "$SAVE_DIR"
}

check_command() {
  local command_name="$1"
  local install_hint="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name command not found." >&2
    echo "Install hint: $install_hint" >&2
    exit 1
  fi
}

activate_conda() {
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$CONDA_ENV" ]]; then
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Activate $CONDA_ENV first or install conda." >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

check_dependencies() {
  check_command gunicorn "pip install -r demo/backend/requirements.txt"
  check_command npm "conda install -c conda-forge nodejs -y"
  if [[ ! -x "$PROJECT_ROOT/demo/frontend/node_modules/.bin/vite" ]]; then
    echo "vite command not found in demo/frontend/node_modules." >&2
    echo "Install hint: cd demo/frontend && npm install" >&2
    exit 1
  fi
}

whisper_cpp_library_path() {
  local build_dir
  build_dir="$(cd "$(dirname "$WHISPER_CPP_BINARY")/.." && pwd)"
  printf '%s:%s:%s' \
    "$build_dir/src" \
    "$build_dir/ggml/src" \
    "$build_dir/ggml/src/ggml-cuda"
}

port_is_open() {
  local host="$1"
  local port="$2"
  python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
try:
    sock.settimeout(0.5)
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

start_whisper_cpp_server() {
  if [[ "$START_WHISPER_CPP" != "1" ]]; then
    return
  fi
  if port_is_open "$WHISPER_CPP_HOST" "$WHISPER_CPP_PORT"; then
    echo "whisper.cpp server already listening at http://$WHISPER_CPP_HOST:$WHISPER_CPP_PORT"
    rm -f "$WHISPER_CPP_PID"
    return
  fi
  if [[ ! -x "$WHISPER_CPP_BINARY" ]]; then
    echo "whisper.cpp server binary not found or not executable: $WHISPER_CPP_BINARY" >&2
    echo "Demo will continue without whisper.cpp server." >&2
    return
  fi
  if [[ ! -f "$WHISPER_CPP_MODEL_PATH" ]]; then
    echo "whisper.cpp model not found: $WHISPER_CPP_MODEL_PATH" >&2
    echo "Demo will continue without whisper.cpp server." >&2
    return
  fi

  local library_path
  local flash_attn_arg
  library_path="$(whisper_cpp_library_path)"
  if [[ "$WHISPER_CPP_FLASH_ATTN" == "1" ]]; then
    flash_attn_arg="--flash-attn"
  else
    flash_attn_arg="--no-flash-attn"
  fi
  (
    cd "$PROJECT_ROOT"
    LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$WHISPER_CPP_BINARY" \
      --model "$WHISPER_CPP_MODEL_PATH" \
      --host "$WHISPER_CPP_HOST" \
      --port "$WHISPER_CPP_PORT" \
      --language ko \
      --beam-size 1 \
      --threads "$WHISPER_CPP_THREADS" \
      --processors "$WHISPER_CPP_PROCESSORS" \
      --device "$WHISPER_CPP_DEVICE_INDEX" \
      "$flash_attn_arg" \
      --no-language-probabilities
  ) >"$LOG_DIR/whisper_cpp_server.log" 2>&1 &
  echo "$!" >"$WHISPER_CPP_PID"
  echo "whisper.cpp server: http://$WHISPER_CPP_HOST:$WHISPER_CPP_PORT"
}

backend_command() {
  cd "$PROJECT_ROOT"
  exec gunicorn app.main:app \
    --chdir "$PROJECT_ROOT/demo/backend" \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "$GUNICORN_WORKERS" \
    --bind "$BACKEND_HOST:$BACKEND_PORT" \
    --timeout 0 \
    --access-logfile - \
    --error-logfile -
}

frontend_command() {
  cd "$PROJECT_ROOT/demo/frontend"
  exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1
}

stop_pid() {
  local name="$1"
  local pid_file="$2"
  if ! is_running "$pid_file"; then
    rm -f "$pid_file"
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  echo "Stopping $name pid=$pid"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$pid_file"
      return
    fi
    sleep 0.5
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pid_file"
}

start_background() {
  ensure_dirs
  activate_conda
  check_dependencies

  if is_running "$BACKEND_PID" || is_running "$FRONTEND_PID"; then
    echo "Demo is already running. Use restart or stop first."
    status
    exit 1
  fi

  start_whisper_cpp_server

  (
    backend_command
  ) >"$LOG_DIR/backend.log" 2>&1 &
  echo "$!" >"$BACKEND_PID"

  (
    frontend_command
  ) >"$LOG_DIR/frontend.log" 2>&1 &
  echo "$!" >"$FRONTEND_PID"

  echo "Backend:  http://127.0.0.1:$BACKEND_PORT"
  echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
  echo "whisper.cpp: http://$WHISPER_CPP_HOST:$WHISPER_CPP_PORT"
  echo "Logs:     $LOG_DIR"
  echo "Audio:    $SAVE_DIR"
}

stop_all() {
  stop_pid "frontend" "$FRONTEND_PID"
  stop_pid "backend" "$BACKEND_PID"
  stop_pid "whisper.cpp server" "$WHISPER_CPP_PID"
}

status() {
  if is_running "$BACKEND_PID"; then
    echo "backend running pid=$(cat "$BACKEND_PID")"
  else
    echo "backend stopped"
  fi

  if is_running "$FRONTEND_PID"; then
    echo "frontend running pid=$(cat "$FRONTEND_PID")"
  else
    echo "frontend stopped"
  fi

  if is_running "$WHISPER_CPP_PID"; then
    echo "whisper.cpp server running pid=$(cat "$WHISPER_CPP_PID")"
  elif port_is_open "$WHISPER_CPP_HOST" "$WHISPER_CPP_PORT"; then
    echo "whisper.cpp server running on port $WHISPER_CPP_PORT"
  else
    echo "whisper.cpp server stopped"
  fi
}

console() {
  ensure_dirs
  activate_conda
  check_dependencies

  local backend_pid=""
  local frontend_pid=""
  local whisper_cpp_pid=""

  cleanup() {
    if [[ -n "$frontend_pid" ]]; then
      kill "$frontend_pid" >/dev/null 2>&1 || true
    fi
    if [[ -n "$backend_pid" ]]; then
      kill "$backend_pid" >/dev/null 2>&1 || true
    fi
    if [[ -n "$whisper_cpp_pid" ]]; then
      kill "$whisper_cpp_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT INT TERM

  start_whisper_cpp_server
  if is_running "$WHISPER_CPP_PID"; then
    whisper_cpp_pid="$(cat "$WHISPER_CPP_PID")"
  fi

  (
    backend_command
  ) &
  backend_pid="$!"

  (
    frontend_command
  ) &
  frontend_pid="$!"

  echo "Backend:  http://127.0.0.1:$BACKEND_PORT"
  echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
  echo "whisper.cpp: http://$WHISPER_CPP_HOST:$WHISPER_CPP_PORT"
  echo "Logs:     $LOG_DIR"
  echo "Audio:    $SAVE_DIR"
  echo "Press Ctrl-C to stop."

  wait -n "$backend_pid" "$frontend_pid"
}

action="${1:-console}"
case "$action" in
  console)
    console
    ;;
  start)
    start_background
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    start_background
    ;;
  status)
    status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
