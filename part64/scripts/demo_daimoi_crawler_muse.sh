#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART64_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8787}"

pushd "${PART64_DIR}" >/dev/null
trap 'popd >/dev/null' EXIT

echo "[demo] starting runtime stack (eta-mu-system, eta-mu-gateway)"
docker compose up -d eta-mu-system eta-mu-gateway >/dev/null

echo "[demo] waiting for runtime health"
for _ in $(seq 1 40); do
  if curl -fsS "${BASE_URL}/api/catalog" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

ROOT_CODE="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/")"
CATALOG_CODE="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/catalog")"
SIM_CODE="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/simulation?perspective=hybrid&payload=trimmed")"
echo "[demo] health codes: root=${ROOT_CODE} catalog=${CATALOG_CODE} simulation=${SIM_CODE}"

echo "[demo] muse query #1: crawler learning"
RESP1="$(curl -fsS -X POST "${BASE_URL}/api/muse/message" \
  --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{
    "muse_id": "witness_thread",
    "mode": "deterministic",
    "token_budget": 1024,
    "text": "/graph web_resource_summary https://example.org/a"
  }')"

echo "[demo] muse query #2: daimoi win/loss explanation"
RESP2="$(curl -fsS -X POST "${BASE_URL}/api/muse/message" \
  --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{
    "muse_id": "witness_thread",
    "mode": "deterministic",
    "token_budget": 1024,
    "text": "/graph explain_daimoi field:witness_thread:001"
  }')"

python - <<'PY' "$RESP1" "$RESP2"
import json
import sys

def render(label: str, raw: str) -> None:
    try:
        payload = json.loads(raw)
    except Exception:
        print(f"[{label}] invalid json response")
        print(raw)
        return
    print(f"[{label}] ok={payload.get('ok')} turn_id={payload.get('turn_id', '')}")
    print(f"[{label}] reply={payload.get('reply', '')}")
    receipts = payload.get('grounded_receipts', {})
    if isinstance(receipts, dict) and receipts:
        print(f"[{label}] receipts={json.dumps(receipts, ensure_ascii=False)}")

render("muse-1", sys.argv[1])
render("muse-2", sys.argv[2])
PY

echo "[demo] done"
