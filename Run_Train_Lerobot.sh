#!/bin/bash
# Run_Train_Lerobot.sh
# Train ACT / Diffusion Policy / VQ-BeT on the converted fine-align lerobot dataset.
#
# Prerequisites
# -------------
# 1) lerobot available at ./lerobot.
#    Install (separate env recommended):
#       cd /home/najo/NAS/VLANeXt/lerobot && uv sync --extra all
#    or `uv pip install -e .` for a minimal install.
#
# 2) Convert the HDF5 dataset once (run from VLANeXt repo root):
#       python -m dataset.convert_to_lerobot \
#           --src /home/najo/NAS/VLANeXt/dataset/approach/approach_00 \
#           --repo-id vlanext/sim_align_baseline \
#           --root /home/najo/NAS/VLANeXt/dataset/lerobot \
#           --fps 15
#
# Usage
# -----
#   bash Run_Train_Lerobot.sh act
#   bash Run_Train_Lerobot.sh dp        # Diffusion Policy (DDPM, horizon=8 + n_obs=1 to match ACT)
#   bash Run_Train_Lerobot.sh vqbet
#   bash Run_Train_Lerobot.sh act-resume <prior_run_dir>   # resume from latest checkpoint
#
# Env overrides:
#   GPUS=2 STEPS=40000 BATCH=32 bash Run_Train_Lerobot.sh act
#   DATASET_ROOT=/path/to/lerobot bash Run_Train_Lerobot.sh dp
#
# Output
# ------
# Lerobot writes checkpoints under outputs/train/<run_name>/checkpoints/.
# Use the resulting `pretrained_model` dir with the eval bridge:
#   bash Run_Eval_Parallel.sh lerobot_act \
#       outputs/train/<run>/checkpoints/last/pretrained_model \
#       --max-steps 250 --eval-seed 2026 \
#       --perturb-mode grid --xy-steps 3 --z-steps 2 --angle-steps 3 --repeats 1
#
# CLI flags follow lerobot v0.5 (draccus). Input/output feature shapes are
# auto-inferred from the dataset metadata; we only need to pick the policy and
# basic hyperparameters. Override anything else by appending `--<dotted.path>=<value>`.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

POLICY=${1:-act}
DATASET_REPO_ID=${DATASET_REPO_ID:-vlanext/sim_align_baseline}
DATASET_ROOT=${DATASET_ROOT:-${REPO_ROOT}/dataset/lerobot_sim}
GPUS=${GPUS:-1}
NUM_GPUS=$(echo "$GPUS" | awk -F, '{print NF}')
# Workaround: GPU 0 in error state poisons CUDA enumeration → force PCI bus ordering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Reduce fragmentation OOM at large feature maps (480x640 → 300 spatial tokens).
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN_NAME="lerobot_${POLICY}_align_$(date +%Y%m%d_%H%M)"

# lerobot >=0.5: launch via accelerate for multi-GPU (DDP). Single-GPU also works.
LEROBOT_TRAIN="${LEROBOT_TRAIN:-$(command -v lerobot-train || true)}"
if [ -z "${LEROBOT_TRAIN}" ]; then
    echo "ERROR: lerobot-train not found. Activate the lerobot env or set LEROBOT_TRAIN=/path/to/lerobot-train."
    exit 1
fi
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=no ${LEROBOT_TRAIN}"
else
    TRAIN_CMD="${LEROBOT_TRAIN}"
fi

COMMON_ARGS=(
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--output_dir=outputs/train/${RUN_NAME}"
    "--job_name=${RUN_NAME}"
    "--seed=2026"
    "--save_freq=5000"
    "--log_freq=100"
    "--policy.device=cuda"
    "--policy.push_to_hub=false"
)

case "$POLICY" in
    act)
        EXTRA=(
            "--policy.type=act"
            "--policy.chunk_size=8"
            "--policy.n_action_steps=8"
            "--batch_size=${BATCH:-48}"
            "--steps=${STEPS:-80000}"
        )
        ;;
    act-resume|resume)
        # Usage: bash Run_Train_Lerobot.sh act-resume outputs/train/<prior_run_dir>
        PRIOR_DIR="${2:?need prior run output_dir as 2nd arg}"
        if [ ! -d "${PRIOR_DIR}" ]; then
            echo "ERROR: prior run dir not found: ${PRIOR_DIR}"; exit 1
        fi
        echo "Resuming from: ${PRIOR_DIR}"
        # When resuming, lerobot reads the prior config from output_dir/checkpoints/last/.
        CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD} \
            --config_path="${PRIOR_DIR}/checkpoints/last/pretrained_model/train_config.json" \
            --output_dir="${PRIOR_DIR}" \
            --dataset.root="${DATASET_ROOT}" \
            --resume=true "${@:3}"
        exit $?
        ;;
    dp|diffusion)
        # horizon=8 + n_obs_steps=1 → matches ACT chunk/obs structure for fair comparison.
        EXTRA=(
            "--policy.type=diffusion"
            "--policy.horizon=8"
            "--policy.n_action_steps=8"
            "--policy.n_obs_steps=1"
            "--batch_size=${BATCH:-64}"
            "--steps=${STEPS:-80000}"
        )
        ;;
    vqbet)
        EXTRA=(
            "--policy.type=vqbet"
            "--policy.action_chunk_size=8"
            "--batch_size=64"
            "--steps=120000"
        )
        ;;
    *)
        echo "Unknown policy: $POLICY (expected: act | dp | vqbet)"; exit 1 ;;
esac

echo "=== Lerobot Train: ${POLICY} ==="
echo "Run:     ${RUN_NAME}"
echo "Dataset: ${DATASET_REPO_ID} (root=${DATASET_ROOT})"
echo "GPUs:    ${GPUS} (num=${NUM_GPUS})"
echo

CUDA_VISIBLE_DEVICES=${GPUS} ${TRAIN_CMD} "${COMMON_ARGS[@]}" "${EXTRA[@]}" "${@:2}"
