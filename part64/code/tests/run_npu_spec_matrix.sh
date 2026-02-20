#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/err/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5/snapshots/e5cf08aadaa33385f5990def41f7a23405aec398/onnx/model.onnx}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/err/.cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/72ae6878a1ab06eac891dc58577ed1652379afb5/onnx/model_int8.onnx}"
LIB_DIR="${LIB_DIR:-./onnxruntime-linux-x64-gpu-1.20.1/lib}"

export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"

echo "Building bench_embed_native..."
g++ -std=c++17 -O3 -o bench_embed_native \
  code/tests/bench_embed_native.cpp \
  -I onnxruntime-linux-x64-gpu-1.20.1/include \
  -L "$LIB_DIR" \
  -lonnxruntime \
  -Wl,-rpath,"$LIB_DIR"

echo "Running CPU baseline (model_only/full)..."
./bench_embed_native --model "$MODEL_PATH" --device CPU --verify-device strict --timing model_only --dim 0 --n 80 --warmup 10 --out runs/npu_spec_cpu_model_full --tag cpu_model_full

echo "Running CPU baseline (boundary/full)..."
./bench_embed_native --model "$MODEL_PATH" --device CPU --verify-device strict --timing boundary --dim 0 --n 80 --warmup 10 --out runs/npu_spec_cpu_boundary_full --tag cpu_boundary_full

echo "Running CPU baseline (model_only/128)..."
./bench_embed_native --model "$MODEL_PATH" --device CPU --verify-device strict --timing model_only --dim 128 --n 80 --warmup 10 --out runs/npu_spec_cpu_model_128 --tag cpu_model_128

echo "Running CUDA baseline (model_only/full)..."
./bench_embed_native --model "$MODEL_PATH" --device CUDA --verify-device strict --timing model_only --dim 0 --n 80 --warmup 10 --out runs/npu_spec_cuda_model_full --tag cuda_model_full

echo "Running CUDA baseline (boundary/full)..."
./bench_embed_native --model "$MODEL_PATH" --device CUDA --verify-device strict --timing boundary --dim 0 --n 80 --warmup 10 --out runs/npu_spec_cuda_boundary_full --tag cuda_boundary_full

echo "Running CUDA baseline (model_only/128)..."
./bench_embed_native --model "$MODEL_PATH" --device CUDA --verify-device strict --timing model_only --dim 128 --n 80 --warmup 10 --out runs/npu_spec_cuda_model_128 --tag cuda_model_128

if [ -f "$QWEN_MODEL_PATH" ]; then
  echo "Running Qwen3 int8 (CPU/model_only/128)..."
  ./bench_embed_native --model "$QWEN_MODEL_PATH" --device CPU --verify-device strict --timing model_only --dim 128 --n 40 --warmup 5 --out runs/npu_spec_qwen_int8_cpu_model_128 --tag qwen_int8_cpu_model_128

  echo "Running Qwen3 int8 (CUDA/model_only/128)..."
  ./bench_embed_native --model "$QWEN_MODEL_PATH" --device CUDA --verify-device strict --timing model_only --dim 128 --n 30 --warmup 5 --out runs/npu_spec_qwen_int8_cuda_model_128 --tag qwen_int8_cuda_model_128
else
  echo "Qwen model not found at $QWEN_MODEL_PATH; skipping Qwen rows."
fi

echo "Running NPU strict verification (expected to fail if OpenVINO EP missing)..."
if ./bench_embed_native --model "$MODEL_PATH" --device NPU --verify-device strict --timing model_only --dim 0 --n 10 --warmup 2 --out runs/npu_spec_npu_model_full --tag npu_model_full; then
  echo "NPU strict run succeeded."
else
  echo "NPU strict run failed as expected when OpenVINO EP or NPU runtime is unavailable."
fi

python -m code.tests.compare_bench_runs

echo "Done. See runs/compare.json"
