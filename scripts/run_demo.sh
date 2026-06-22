#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${DEMO_CONFIG_PATH:-$PROJECT_ROOT/demo/config.yaml}"
CONDA_ENV="${DEMO_CONDA_ENV:-korean-asr-benchmark}"

demo_config_value() {
  local key="$1"
  local default_value="$2"
  python "$PROJECT_ROOT/demo/tools/read_config_value.py" "$CONFIG_PATH" "$key" "$default_value"
}

resolve_project_path() {
  local path_value="$1"
  if [[ "$path_value" = /* ]]; then
    printf '%s\n' "$path_value"
  else
    printf '%s\n' "$PROJECT_ROOT/$path_value"
  fi
}

BACKEND_HOST="${DEMO_BACKEND_HOST:-$(demo_config_value server.backend_host 0.0.0.0)}"
BACKEND_PORT="${DEMO_BACKEND_PORT:-$(demo_config_value server.backend_port 16000)}"
FRONTEND_HOST="${DEMO_FRONTEND_HOST:-$(demo_config_value server.frontend_host 0.0.0.0)}"
FRONTEND_PORT="${DEMO_FRONTEND_PORT:-$(demo_config_value server.frontend_port 16010)}"
GUNICORN_WORKERS="${DEMO_GUNICORN_WORKERS:-$(demo_config_value server.gunicorn_workers 1)}"
SSL_ENABLED="${DEMO_SSL_ENABLED:-$(demo_config_value server.ssl.enabled 0)}"
SSL_AUTO_GENERATE="${DEMO_SSL_AUTO_GENERATE:-$(demo_config_value server.ssl.auto_generate 1)}"
SSL_CERT_FILE_VALUE="${DEMO_SSL_CERT_FILE:-$(demo_config_value server.ssl.cert_file certs/demo.crt)}"
SSL_KEY_FILE_VALUE="${DEMO_SSL_KEY_FILE:-$(demo_config_value server.ssl.key_file certs/demo.key)}"
SSL_HOSTS="${DEMO_SSL_HOSTS:-$(demo_config_value server.ssl.hosts auto)}"
SSL_CERT_FILE="$(resolve_project_path "$SSL_CERT_FILE_VALUE")"
SSL_KEY_FILE="$(resolve_project_path "$SSL_KEY_FILE_VALUE")"
if [[ "$SSL_ENABLED" == "1" ]]; then
  BACKEND_PROXY_SCHEME="https"
else
  BACKEND_PROXY_SCHEME="http"
fi
BACKEND_PROXY_TARGET="${DEMO_BACKEND_PROXY_TARGET:-$BACKEND_PROXY_SCHEME://127.0.0.1:$BACKEND_PORT}"
RUN_ID="${DEMO_RUN_ID:-$(date +%Y%m%d_%H%M%S)_log}"
RUN_DIR="${DEMO_RUN_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
LOG_DIR="${DEMO_LOG_DIR:-$RUN_DIR}"
SAVE_DIR="${DEMO_SAVE_DIR:-$RUN_DIR/saved_audio}"
PID_DIR="${DEMO_PID_DIR:-$PROJECT_ROOT/logs/.pid}"
CURRENT_LOG_LINK="${DEMO_CURRENT_LOG_LINK:-$PROJECT_ROOT/logs/current_log}"
BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"

export DEMO_RUN_DIR="$RUN_DIR"
export DEMO_LOG_DIR="$LOG_DIR"
export DEMO_SAVE_DIR="$SAVE_DIR"
export DEMO_CONFIG_PATH="$CONFIG_PATH"
export DEMO_BACKEND_PORT="$BACKEND_PORT"
export DEMO_FRONTEND_PORT="$FRONTEND_PORT"
export DEMO_SSL_ENABLED="$SSL_ENABLED"
export DEMO_SSL_CERT_FILE="$SSL_CERT_FILE"
export DEMO_SSL_KEY_FILE="$SSL_KEY_FILE"
export VITE_BACKEND_TARGET="$BACKEND_PROXY_TARGET"

usage() {
  cat <<EOF
Usage: scripts/run_demo.sh [console|start|stop|restart|status]

Default action is console.

Environment variables:
  DEMO_CONDA_ENV          Default: korean-asr-benchmark
  DEMO_CONFIG_PATH        Default: demo/config.yaml
  DEMO_BACKEND_HOST       Default: demo config server.backend_host
  DEMO_BACKEND_PORT       Default: demo config server.backend_port
  DEMO_FRONTEND_HOST      Default: demo config server.frontend_host
  DEMO_FRONTEND_PORT      Default: demo config server.frontend_port
  DEMO_GUNICORN_WORKERS   Default: demo config server.gunicorn_workers
  DEMO_SSL_ENABLED        Default: demo config server.ssl.enabled
  DEMO_SSL_AUTO_GENERATE  Default: demo config server.ssl.auto_generate
  DEMO_SSL_CERT_FILE      Default: demo config server.ssl.cert_file
  DEMO_SSL_KEY_FILE       Default: demo config server.ssl.key_file
  DEMO_SSL_HOSTS          Default: demo config server.ssl.hosts
  DEMO_BACKEND_PROXY_TARGET Default: http(s)://127.0.0.1:backend_port
  DEMO_FRONTEND_API_BASE  Default: empty, use frontend origin and Vite proxy
  DEMO_RUN_ID             Default: YYYYMMDD_HHMMSS_log
  DEMO_RUN_DIR            Default: logs/YYYYMMDD_HHMMSS_log
  DEMO_LOG_DIR            Default: DEMO_RUN_DIR
  DEMO_SAVE_DIR           Default: DEMO_RUN_DIR/saved_audio
  DEMO_PID_DIR            Default: logs/.pid
  DEMO_CURRENT_LOG_LINK   Default: logs/current_log
EOF
}

ensure_dirs() {
  mkdir -p "$LOG_DIR" "$SAVE_DIR"
  ensure_pid_dir
}

ensure_pid_dir() {
  mkdir -p "$PID_DIR"
}

refresh_current_log_link() {
  mkdir -p "$(dirname "$CURRENT_LOG_LINK")"
  if [[ -L "$CURRENT_LOG_LINK" || ! -e "$CURRENT_LOG_LINK" ]]; then
    ln -sfn "$LOG_DIR" "$CURRENT_LOG_LINK"
    return
  fi

  local backup_path
  backup_path="${CURRENT_LOG_LINK}.backup.$(date +%Y%m%d_%H%M%S)"
  mv "$CURRENT_LOG_LINK" "$backup_path"
  ln -sfn "$LOG_DIR" "$CURRENT_LOG_LINK"
  echo "Moved existing current_log directory to $backup_path"
}

public_scheme() {
  if [[ "$SSL_ENABLED" == "1" ]]; then
    printf '%s\n' "https"
  else
    printf '%s\n' "http"
  fi
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
  if [[ "$SSL_ENABLED" == "1" && "$SSL_AUTO_GENERATE" == "1" ]]; then
    check_command openssl "conda install -c conda-forge openssl -y"
  fi
  if [[ ! -x "$PROJECT_ROOT/demo/frontend/node_modules/.bin/vite" ]]; then
    echo "vite command not found in demo/frontend/node_modules." >&2
    echo "Install hint: cd demo/frontend && npm install" >&2
    exit 1
  fi
}

detect_ssl_hosts() {
  if [[ "$SSL_HOSTS" != "auto" ]]; then
    printf '%s\n' "$SSL_HOSTS"
    return
  fi

  local hosts
  hosts="localhost 127.0.0.1"
  if command -v hostname >/dev/null 2>&1; then
    hosts="$hosts $(hostname -f 2>/dev/null || true) $(hostname -I 2>/dev/null || true)"
  fi
  printf '%s\n' "$hosts" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' '
}

write_ssl_openssl_config() {
  local config_file="$1"
  local hosts="$2"
  local dns_index=1
  local ip_index=1

  {
    cat <<'EOF'
[req]
default_bits = 2048
prompt = no
default_md = sha256
x509_extensions = v3_req
distinguished_name = dn

[dn]
CN = korean-asr-demo

[v3_req]
subjectAltName = @alt_names

[alt_names]
EOF
    for host in $hosts; do
      if [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf 'IP.%d = %s\n' "$ip_index" "$host"
        ip_index=$((ip_index + 1))
      else
        printf 'DNS.%d = %s\n' "$dns_index" "$host"
        dns_index=$((dns_index + 1))
      fi
    done
  } >"$config_file"
}

ensure_ssl_cert() {
  if [[ "$SSL_ENABLED" != "1" ]]; then
    return
  fi
  if [[ -f "$SSL_CERT_FILE" && -f "$SSL_KEY_FILE" ]]; then
    return
  fi
  if [[ "$SSL_AUTO_GENERATE" != "1" ]]; then
    echo "SSL cert/key not found: $SSL_CERT_FILE / $SSL_KEY_FILE" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$SSL_CERT_FILE")" "$(dirname "$SSL_KEY_FILE")"
  local config_file
  local hosts
  config_file="$(mktemp)"
  hosts="$(detect_ssl_hosts)"
  write_ssl_openssl_config "$config_file" "$hosts"
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SSL_KEY_FILE" \
    -out "$SSL_CERT_FILE" \
    -config "$config_file" >/dev/null 2>&1
  rm -f "$config_file"
  echo "Generated self-signed SSL certificate: $SSL_CERT_FILE"
  echo "Certificate hosts: $hosts"
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

ensure_port_free() {
  local name="$1"
  local port="$2"
  if port_is_open 127.0.0.1 "$port"; then
    echo "$name port is already in use: $port" >&2
    echo "Stop the existing process first, or change demo/config.yaml." >&2
    exit 1
  fi
}

ensure_single_backend_worker() {
  if [[ "$GUNICORN_WORKERS" != "1" ]]; then
    echo "DEMO_GUNICORN_WORKERS/server.gunicorn_workers must be 1 for the demo." >&2
    echo "Model workers already provide engine-level process isolation; multiple backend workers would reuse the same internal worker ports." >&2
    exit 1
  fi
}

backend_command() {
  cd "$PROJECT_ROOT"
  local ssl_args=()
  if [[ "$SSL_ENABLED" == "1" ]]; then
    ssl_args=(--certfile "$SSL_CERT_FILE" --keyfile "$SSL_KEY_FILE")
  fi
  exec gunicorn app.main:app \
    --chdir "$PROJECT_ROOT/demo/backend" \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "$GUNICORN_WORKERS" \
    --bind "$BACKEND_HOST:$BACKEND_PORT" \
    --timeout 0 \
    --access-logfile - \
    --error-logfile - \
    "${ssl_args[@]}"
}

frontend_command() {
  cd "$PROJECT_ROOT/demo/frontend"
  VITE_API_BASE="${DEMO_FRONTEND_API_BASE:-}" \
  VITE_BACKEND_TARGET="$BACKEND_PROXY_TARGET" \
  exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
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

stop_matching_processes() {
  local name="$1"
  local pattern="$2"
  if ! pgrep -f "$pattern" >/dev/null 2>&1; then
    return
  fi

  echo "Stopping orphan $name process."
  pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      return
    fi
    sleep 0.3
  done
  pkill -KILL -f "$pattern" >/dev/null 2>&1 || true
}

stop_orphan_demo_processes() {
  stop_matching_processes "frontend vite" "korean-asr-model-benchmark/demo/frontend/node_modules/.bin/vite"
  stop_matching_processes "frontend npm" "npm run dev -- --host .* --port"
  stop_matching_processes "backend gunicorn" "gunicorn app.main:app --chdir .*/korean-asr-model-benchmark/demo/backend"
  stop_matching_processes "demo model worker" "app.worker_process --engine-id"
  stop_matching_processes "whisper.cpp server" "korean-asr-model-benchmark/third_party/whisper.cpp/build/bin/whisper-server"
}

start_background() {
  ensure_pid_dir
  if is_running "$BACKEND_PID" || is_running "$FRONTEND_PID"; then
    echo "Demo is already running. Use restart or stop first."
    status
    exit 1
  fi

  ensure_dirs
  activate_conda
  check_dependencies
  ensure_single_backend_worker
  ensure_ssl_cert
  ensure_port_free "backend" "$BACKEND_PORT"
  ensure_port_free "frontend" "$FRONTEND_PORT"

  refresh_current_log_link

  (
    backend_command
  ) >"$LOG_DIR/backend.log" 2>&1 &
  echo "$!" >"$BACKEND_PID"

  (
    frontend_command
  ) >"$LOG_DIR/frontend.log" 2>&1 &
  echo "$!" >"$FRONTEND_PID"

  echo "Backend:  $(public_scheme)://127.0.0.1:$BACKEND_PORT"
  echo "Frontend: $(public_scheme)://127.0.0.1:$FRONTEND_PORT"
  echo "API proxy: $BACKEND_PROXY_TARGET"
  echo "whisper.cpp: managed by backend engine activation"
  echo "Logs:     $LOG_DIR"
  echo "Audio:    $SAVE_DIR"
}

stop_all() {
  ensure_pid_dir
  stop_pid "frontend" "$FRONTEND_PID"
  stop_pid "backend" "$BACKEND_PID"
  stop_orphan_demo_processes
}

status() {
  ensure_pid_dir
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

  echo "whisper.cpp server is managed by backend engine activation."
}

console() {
  ensure_pid_dir
  if is_running "$BACKEND_PID" || is_running "$FRONTEND_PID"; then
    echo "Demo is already running. Use restart or stop first."
    status
    exit 1
  fi

  ensure_dirs
  activate_conda
  check_dependencies
  ensure_single_backend_worker
  ensure_ssl_cert
  ensure_port_free "backend" "$BACKEND_PORT"
  ensure_port_free "frontend" "$FRONTEND_PORT"

  local backend_pid=""
  local frontend_pid=""

  cleanup() {
    if [[ -n "$frontend_pid" ]]; then
      kill "$frontend_pid" >/dev/null 2>&1 || true
    fi
    if [[ -n "$backend_pid" ]]; then
      kill "$backend_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT INT TERM

  refresh_current_log_link

  (
    backend_command
  ) > >(tee -a "$LOG_DIR/backend.log") 2>&1 &
  backend_pid="$!"

  (
    frontend_command
  ) > >(tee -a "$LOG_DIR/frontend.log") 2>&1 &
  frontend_pid="$!"

  echo "Backend:  $(public_scheme)://127.0.0.1:$BACKEND_PORT"
  echo "Frontend: $(public_scheme)://127.0.0.1:$FRONTEND_PORT"
  echo "API proxy: $BACKEND_PROXY_TARGET"
  echo "whisper.cpp: managed by backend engine activation"
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
