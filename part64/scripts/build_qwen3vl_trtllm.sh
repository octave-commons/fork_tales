#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_WRAPPER="$ROOT_DIR/scripts/trtllm_env.sh"

if [[ ! -x "$ENV_WRAPPER" ]]; then
  echo "missing trtllm env wrapper: $ENV_WRAPPER" >&2
  exit 1
fi

CHECKPOINT_DIR="${1:-${QWEN3VL_CHECKPOINT_DIR:-}}"
OUTPUT_DIR="${2:-${QWEN3VL_ENGINE_DIR:-$ROOT_DIR/artifacts/trtllm/qwen3-vl}}"
MODEL_REF="${QWEN3VL_MODEL_REF:-Qwen/Qwen3-VL-2B-Instruct}"
WORKSPACE_DIR="${TRTLLM_WORKSPACE_DIR:-$ROOT_DIR/.cache-trtllm-build}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "usage: $0 <checkpoint_dir> [output_dir]" >&2
  echo "or set QWEN3VL_CHECKPOINT_DIR and optional QWEN3VL_ENGINE_DIR" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORKSPACE_DIR"

MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${TRTLLM_MAX_SEQ_LEN:-2048}"
MAX_NUM_TOKENS="${TRTLLM_MAX_NUM_TOKENS:-4096}"
TP_SIZE="${TRTLLM_TP_SIZE:-1}"
PP_SIZE="${TRTLLM_PP_SIZE:-1}"
TRUST_REMOTE_CODE="${TRTLLM_TRUST_REMOTE_CODE:-true}"
NO_WEIGHTS_LOADING="${TRTLLM_NO_WEIGHTS_LOADING:-false}"

if [[ "$TRUST_REMOTE_CODE" != "true" && "$TRUST_REMOTE_CODE" != "false" ]]; then
  echo "TRTLLM_TRUST_REMOTE_CODE must be 'true' or 'false'" >&2
  exit 1
fi
if [[ "$NO_WEIGHTS_LOADING" != "true" && "$NO_WEIGHTS_LOADING" != "false" ]]; then
  echo "TRTLLM_NO_WEIGHTS_LOADING must be 'true' or 'false'" >&2
  exit 1
fi

"$ENV_WRAPPER" trtllm-bench \
  -m "$MODEL_REF" \
  --model_path "$CHECKPOINT_DIR" \
  -w "$WORKSPACE_DIR" \
  build \
  --tp_size "$TP_SIZE" \
  --pp_size "$PP_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --trust_remote_code "$TRUST_REMOTE_CODE" \
  --no_weights_loading "$NO_WEIGHTS_LOADING"

ENGINE_DIR="$WORKSPACE_DIR/$MODEL_REF/tp_${TP_SIZE}_pp_${PP_SIZE}"
if [[ ! -d "$ENGINE_DIR" ]]; then
  echo "expected engine directory not found: $ENGINE_DIR" >&2
  exit 1
fi

if compgen -G "$ENGINE_DIR/*.engine" > /dev/null; then
  cp -a "$ENGINE_DIR/." "$OUTPUT_DIR/"
else
  echo "no *.engine artifacts found under $ENGINE_DIR" >&2
  exit 1
fi

echo "built TensorRT-LLM engine: $OUTPUT_DIR"
