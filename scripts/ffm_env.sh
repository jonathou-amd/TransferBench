#!/bin/bash
# Source inside customized-competes-gfx1260-*.sif before running TransferBench/roccap.
#
#   cd TransferBench-singularity
#   source scripts/ffm_env.sh

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [[ -d /workspace/rocplaycap-Linux/opt/rocm/bin ]]; then
  export PATH="/workspace/rocplaycap-Linux/opt/rocm/bin:${ROCM_PATH}/bin:${PATH}"
  export LD_LIBRARY_PATH="/workspace/rocplaycap-Linux/opt/rocm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH}"
else
  export PATH="${ROCM_PATH}/bin:${PATH}"
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH}"
fi

# Do NOT bind-mount host libhsa into the container.
