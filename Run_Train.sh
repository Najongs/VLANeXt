#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Default: final Qwen v11 fine-align recipe. Override with:
#   CONFIG=config/your_config.yaml GPUS=1 bash Run_Train.sh
CONFIG="${CONFIG:-config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml}"
GPUS="${GPUS:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS=${#GPU_LIST[@]}

echo "=== VLANeXt Train ==="
echo "Config: ${CONFIG}"
echo "GPUs:   ${GPUS}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29505}" \
        -m scripts.train --config "${CONFIG}" "$@"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m scripts.train --config "${CONFIG}" "$@"
fi
