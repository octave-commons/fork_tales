#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-/tmp/eta-mu-tts.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[eta-mu-tts] not running"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid" || true
fi

rm -f "$PID_FILE"
echo "[eta-mu-tts] stopped"
