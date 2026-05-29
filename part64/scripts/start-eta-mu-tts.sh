#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${PID_FILE:-/tmp/eta-mu-tts.pid}"
LOG_FILE="${LOG_FILE:-/tmp/eta-mu-tts.log}"
export ETA_MU_TTS_HOST="${ETA_MU_TTS_HOST:-0.0.0.0}"
export ETA_MU_TTS_PORT="${ETA_MU_TTS_PORT:-8790}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[eta-mu-tts] already running with pid $pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$ROOT"
nohup python3 code/tts_service.py >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[eta-mu-tts] started pid $(cat "$PID_FILE") on ${ETA_MU_TTS_HOST}:${ETA_MU_TTS_PORT}"
