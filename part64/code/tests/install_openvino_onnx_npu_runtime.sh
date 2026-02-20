#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ORT_VERSION="${ORT_VERSION:-1.22.0}"
NPU_DRIVER_REPO="${NPU_DRIVER_REPO:-intel/linux-npu-driver}"
NPU_DRIVER_PATTERN="${NPU_DRIVER_PATTERN:-*ubuntu2404.tar.gz}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache-npu}"
LIBZE_LOCAL_ROOT="${LIBZE_LOCAL_ROOT:-$CACHE_DIR/libze}"
LIBZE_LOCAL_LIB="$LIBZE_LOCAL_ROOT/extracted/usr/lib/x86_64-linux-gnu"

mkdir -p "$CACHE_DIR"

echo "[1/7] Installing ONNX Runtime OpenVINO Python runtime..."
if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import openvino  # noqa: F401
PY
then
  echo "OpenVINO Python package already present; leaving current version unchanged."
else
  "$PYTHON_BIN" -m pip install --user "openvino>=2025.4.0"
fi
"$PYTHON_BIN" -m pip install --user --upgrade "onnxruntime-openvino==${ORT_VERSION}"

echo "[2/7] Resolving ONNX Runtime C API directory..."
ORT_CAPI_DIR="$($PYTHON_BIN - <<'PY'
import os
import onnxruntime
print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))
PY
)"
echo "ORT_CAPI_DIR=$ORT_CAPI_DIR"

echo "[3/7] Ensuring soname symlinks exist..."
if [ -f "$ORT_CAPI_DIR/libonnxruntime.so.${ORT_VERSION}" ]; then
  ln -sf "$ORT_CAPI_DIR/libonnxruntime.so.${ORT_VERSION}" "$ORT_CAPI_DIR/libonnxruntime.so.1"
  ln -sf "$ORT_CAPI_DIR/libonnxruntime.so.${ORT_VERSION}" "$ORT_CAPI_DIR/libonnxruntime.so"
else
  if [ -f "$ORT_CAPI_DIR/libonnxruntime.so.1" ]; then
    ln -sf "$ORT_CAPI_DIR/libonnxruntime.so.1" "$ORT_CAPI_DIR/libonnxruntime.so"
  fi
fi

echo "[3b/7] Ensuring Level Zero loader availability (non-root fallback)..."
if ldconfig -p | rg -q libze_loader; then
  echo "System Level Zero loader detected via ldconfig."
else
  mkdir -p "$LIBZE_LOCAL_ROOT"
  if [ ! -f "$LIBZE_LOCAL_ROOT/libze1_1.16.1-1build1_amd64.deb" ]; then
    wget -q "http://us.archive.ubuntu.com/ubuntu/pool/universe/o/oneapi-level-zero/libze1_1.16.1-1build1_amd64.deb" -O "$LIBZE_LOCAL_ROOT/libze1_1.16.1-1build1_amd64.deb"
  fi
  rm -rf "$LIBZE_LOCAL_ROOT/extracted"
  mkdir -p "$LIBZE_LOCAL_ROOT/extracted"
  dpkg-deb -x "$LIBZE_LOCAL_ROOT/libze1_1.16.1-1build1_amd64.deb" "$LIBZE_LOCAL_ROOT/extracted"
  echo "Extracted libze loader to: $LIBZE_LOCAL_LIB"
fi

echo "[4/7] Downloading ONNX Runtime headers for C++ harness..."
ORT_HEADERS_TGZ="$CACHE_DIR/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_HEADERS_DIR="$ROOT_DIR/onnxruntime-linux-x64-${ORT_VERSION}"
if [ ! -d "$ORT_HEADERS_DIR" ]; then
  wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -O "$ORT_HEADERS_TGZ"
  tar -zxf "$ORT_HEADERS_TGZ" -C "$ROOT_DIR"
fi

echo "[5/7] Downloading Intel Linux NPU driver package bundle..."
DRIVER_TAR=""
if ls "$CACHE_DIR"/linux-npu-driver-*ubuntu2404.tar.gz >/dev/null 2>&1; then
  DRIVER_TAR="$(ls -1 "$CACHE_DIR"/linux-npu-driver-*ubuntu2404.tar.gz | head -n 1)"
else
  gh release download -R "$NPU_DRIVER_REPO" --pattern "$NPU_DRIVER_PATTERN" --dir "$CACHE_DIR"
  DRIVER_TAR="$(ls -1 "$CACHE_DIR"/linux-npu-driver-*ubuntu2404.tar.gz | head -n 1)"
fi

DRIVER_EXTRACT_DIR="$CACHE_DIR/linux-npu-driver"
rm -rf "$DRIVER_EXTRACT_DIR"
mkdir -p "$DRIVER_EXTRACT_DIR"
tar -zxf "$DRIVER_TAR" -C "$DRIVER_EXTRACT_DIR"

echo "[6/7] Installing Intel NPU driver .deb packages (requires root)..."
DEB_LIST=(
  "$DRIVER_EXTRACT_DIR/intel-fw-npu_"*.deb
  "$DRIVER_EXTRACT_DIR/intel-driver-compiler-npu_"*.deb
  "$DRIVER_EXTRACT_DIR/intel-level-zero-npu_"*.deb
)

APT_LEVEL_ZERO_DEPS=(libze1)

if [ "${SKIP_DRIVER_INSTALL:-0}" = "1" ]; then
  echo "SKIP_DRIVER_INSTALL=1 set; skipping root driver installation."
elif [ "$(id -u)" = "0" ]; then
  apt-get update
  apt-get install -y "${APT_LEVEL_ZERO_DEPS[@]}"
  dpkg -i "${DEB_LIST[@]}" || true
  apt-get install -f -y
elif command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y "${APT_LEVEL_ZERO_DEPS[@]}"
    sudo dpkg -i "${DEB_LIST[@]}" || true
    sudo apt-get install -f -y
  else
    echo "sudo requires password. Run this command manually to finish NPU driver installation:"
    echo "  sudo apt-get update && sudo apt-get install -y ${APT_LEVEL_ZERO_DEPS[*]} && sudo dpkg -i ${DRIVER_EXTRACT_DIR}/intel-fw-npu_*.deb ${DRIVER_EXTRACT_DIR}/intel-driver-compiler-npu_*.deb ${DRIVER_EXTRACT_DIR}/intel-level-zero-npu_*.deb && sudo apt-get install -f -y"
  fi
else
  echo "No root privileges detected. Run this command manually to finish NPU driver installation:"
  echo "  sudo apt-get update && sudo apt-get install -y ${APT_LEVEL_ZERO_DEPS[*]} && sudo dpkg -i ${DRIVER_EXTRACT_DIR}/intel-fw-npu_*.deb ${DRIVER_EXTRACT_DIR}/intel-driver-compiler-npu_*.deb ${DRIVER_EXTRACT_DIR}/intel-level-zero-npu_*.deb && sudo apt-get install -f -y"
fi

echo "[7/7] Verifying runtime visibility..."
if [ -d "$LIBZE_LOCAL_LIB" ]; then
  export LD_LIBRARY_PATH="$LIBZE_LOCAL_LIB:$ORT_CAPI_DIR:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$ORT_CAPI_DIR:${LD_LIBRARY_PATH:-}"
fi

if [ -f "/usr/lib/x86_64-linux-gnu/libze_intel_npu.so" ]; then
  export ZE_ENABLE_ALT_DRIVERS="/usr/lib/x86_64-linux-gnu/libze_intel_npu.so"
fi

"$PYTHON_BIN" - <<'PY'
import os
import onnxruntime as ort
import openvino as ov

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
print("ZE_ENABLE_ALT_DRIVERS:", os.environ.get("ZE_ENABLE_ALT_DRIVERS", ""))
print("onnxruntime providers:", ort.get_available_providers())
core = ov.Core()
print("openvino devices:", core.available_devices)

if "NPU" in core.available_devices:
    print("NPU device is visible to OpenVINO.")
else:
    print("NPU device NOT visible yet. Driver install/reboot may still be required.")

try:
    model = core.read_model('/home/err/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5/snapshots/e5cf08aadaa33385f5990def41f7a23405aec398/onnx/model.onnx')
    model.reshape({'input_ids': [1, 128], 'token_type_ids': [1, 128], 'attention_mask': [1, 128]})
    core.compile_model(model, 'NPU')
    print('OpenVINO compile_model(..., "NPU") succeeded.')
except Exception as exc:
    print('OpenVINO NPU compile test failed:')
    print(exc)
PY

cat <<EOF

Install script completed.

If NPU is still not visible, reboot and verify:
  LD_LIBRARY_PATH="$LIBZE_LOCAL_LIB:\$LD_LIBRARY_PATH" ZE_ENABLE_ALT_DRIVERS="/usr/lib/x86_64-linux-gnu/libze_intel_npu.so" python3 -c "import openvino as ov; print(ov.Core().available_devices)"
and ensure NPU appears.

To compile the C++ harness against this OpenVINO-enabled ORT runtime:
  ORT_CAPI_DIR="$ORT_CAPI_DIR"
  g++ -std=c++17 -O3 -o bench_embed_native_ov code/tests/bench_embed_native.cpp \
    -I "${ORT_HEADERS_DIR}/include" \
    -L "${ORT_CAPI_DIR}" \
    -l:libonnxruntime.so.${ORT_VERSION} \
    -Wl,-rpath,"${ORT_CAPI_DIR}"

Then run strict NPU verification:
  LD_LIBRARY_PATH="$LIBZE_LOCAL_LIB:${ORT_CAPI_DIR}:\$LD_LIBRARY_PATH" ZE_ENABLE_ALT_DRIVERS="/usr/lib/x86_64-linux-gnu/libze_intel_npu.so" ./bench_embed_native_ov \
    --model /path/to/model.onnx --device NPU --verify-device strict --timing model_only --dim 128 --n 40 --warmup 5 --out runs/npu_spec_ov_npu_model_128 --tag ov_npu_model_128

EOF
