#!/bin/bash
# Parallel eval across GPUs
# Usage:
#   bash Run_Eval_Parallel.sh [mode] [checkpoint_path] [extra_flags...]
#
# Modes:
#   align     - Fine-alignment eval (기본)
#   approach  - Approach eval (먼 거리 → 트로카 접근)
#
# Examples:
#   bash Run_Eval_Parallel.sh align /path/to/checkpoint
#   bash Run_Eval_Parallel.sh align /path/to/checkpoint --randomize-phantom
#   bash Run_Eval_Parallel.sh approach /path/to/checkpoint

MODE=${1:-align}
CHECKPOINT=${2:-/data/public/NAS/VLANeXt/output_dir_align_0410}
NUM_SHARDS=2

# Mode-specific config
if [ "$MODE" = "approach" ]; then
    CONFIG=config/sim_eval_approach_config.yaml
    TRAIN_CONFIG=config/sim_train_align_config.yaml
    EVAL_SCRIPT=scripts.sim_eval_approach_only
    MERGE_PREFIX="approach"
else
    CONFIG=config/sim_eval_align_config.yaml
    TRAIN_CONFIG=config/sim_train_align_config.yaml
    EVAL_SCRIPT=scripts.sim_eval_align_only
    MERGE_PREFIX="align"
fi

# Parse extra flags (--randomize-phantom, etc.)
EXTRA_FLAGS=""
for arg in "${@:3}"; do
    EXTRA_FLAGS="${EXTRA_FLAGS} ${arg}"
done

echo "=== Parallel Eval: ${NUM_SHARDS} GPUs ==="
echo "Mode: ${MODE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Script: ${EVAL_SCRIPT}"
if [[ "${EXTRA_FLAGS}" == *"--randomize-phantom"* ]]; then
    echo "Phantom: RANDOMIZED"
else
    echo "Phantom: FIXED"
fi

# Launch shards in parallel
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID=${SHARD}
    echo "Starting shard ${SHARD} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m ${EVAL_SCRIPT} \
        --config ${CONFIG} \
        --checkpoint ${CHECKPOINT} \
        --train-config ${TRAIN_CONFIG} \
        --shard-id ${SHARD} \
        --num-shards ${NUM_SHARDS} ${EXTRA_FLAGS} &
done

echo "All shards launched. Waiting..."
wait
echo "All shards complete."

# Merge results
python scripts/merge_eval_shards.py ${CHECKPOINT} --num-shards ${NUM_SHARDS} --prefix ${MERGE_PREFIX}
