#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES_DEFAULT="$HOME/.local/lib/python3.12/site-packages"
SITE_PACKAGES="${SITE_PACKAGES:-$SITE_PACKAGES_DEFAULT}"
MPI_ROOT="$ROOT_DIR/.cache-mpi/root/usr"
MPI_LIB="$MPI_ROOT/lib/x86_64-linux-gnu"

if [[ ! -d "$SITE_PACKAGES" ]]; then
  echo "missing site-packages: $SITE_PACKAGES" >&2
  exit 1
fi

if [[ ! -d "$MPI_LIB" ]]; then
  echo "missing local MPI runtime: $MPI_LIB" >&2
  echo "hint: extract OpenMPI deb packages into $ROOT_DIR/.cache-mpi/root" >&2
  exit 1
fi

LD_PATH="$MPI_LIB:$SITE_PACKAGES/tensorrt_libs:$SITE_PACKAGES/tensorrt_llm/libs"
if [[ -d "$SITE_PACKAGES/nvidia" ]]; then
  while IFS= read -r libdir; do
    LD_PATH="$LD_PATH:$libdir"
  done < <(find "$SITE_PACKAGES/nvidia" -type d -name lib | sort)
fi

export OPAL_PREFIX="$MPI_ROOT"
export PMIX_INSTALL_PREFIX="$MPI_ROOT"
export PATH="$MPI_ROOT/bin:$PATH"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$LD_PATH:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$LD_PATH"
fi

if [[ -z "${OMPI_MCA_pml:-}" ]]; then
  export OMPI_MCA_pml="ob1"
fi
if [[ -z "${OMPI_MCA_btl:-}" ]]; then
  export OMPI_MCA_btl="self,vader,tcp"
fi
if [[ -z "${OMPI_MCA_mtl:-}" ]]; then
  export OMPI_MCA_mtl="^ofi,psm,psm2"
fi

if [[ "$#" -eq 0 ]]; then
  exec trtllm-build --help
fi

exec "$@"
