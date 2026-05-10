#!/bin/bash
# Run_Train_Lerobot.sh
# Train ACT / Diffusion Policy / VQ-BeT on the converted fine-align lerobot dataset.
#
# Prerequisites
# -------------
# 1) lerobot cloned at /data/public/NAS/VLANeXt/lerobot.
#    Install (separate env recommended):
#       cd /data/public/NAS/VLANeXt/lerobot && uv sync --extra all
#    or `uv pip install -e .` for a minimal install.
#
# 2) Convert the HDF5 dataset once (run from VLANeXt repo root):
#       python -m dataset.convert_to_lerobot \
#           --src /data/public/NAS/VLANeXt/dataset/approach/approach_00 \
#           --repo-id vlanext/sim_align_baseline \
#           --root /data/public/NAS/VLANeXt/dataset/lerobot \
#           --fps 15
#
# Usage
# -----
#   bash Run_Train_Lerobot.sh act
#   bash Run_Train_Lerobot.sh dp        # Diffusion Policy
#   bash Run_Train_Lerobot.sh vqbet
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

POLICY=${1:-act}
DATASET_REPO_ID=${DATASET_REPO_ID:-vlanext/sim_align_baseline}
DATASET_ROOT=${DATASET_ROOT:-/data/public/NAS/VLANeXt/dataset/lerobot}
GPUS=${GPUS:-1}
NUM_GPUS=$(echo "$GPUS" | awk -F, '{print NF}')
# Workaround: GPU 0 in error state poisons CUDA enumeration → force PCI bus ordering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Reduce fragmentation OOM at large feature maps (480x640 → 300 spatial tokens).
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN_NAME="lerobot_${POLICY}_align_$(date +%Y%m%d_%H%M)"

# lerobot >=0.5: launch via accelerate for multi-GPU (DDP). Single-GPU also works.
LEROBOT_TRAIN=$(command -v lerobot-train || echo "lerobot-train")
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
            "--batch_size=48"
            "--steps=80000"
        )
        ;;
    dp|diffusion)
        EXTRA=(
            "--policy.type=diffusion"
            "--policy.horizon=16"
            "--policy.n_action_steps=8"
            "--policy.n_obs_steps=2"
            "--batch_size=64"
            "--steps=200000"
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
