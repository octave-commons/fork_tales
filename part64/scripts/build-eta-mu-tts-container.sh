#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${PID_FILE:-/tmp/eta-mu-tts-build.pid}"
LOG_FILE="${LOG_FILE:-/tmp/eta-mu-tts-build.log}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[eta-mu-tts-build] already running with pid $pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$ROOT"
nohup docker compose -f docker-compose.yml build eta-mu-tts >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[eta-mu-tts-build] started pid $(cat "$PID_FILE") log $LOG_FILE"
