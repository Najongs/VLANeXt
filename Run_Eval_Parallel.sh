#!/bin/bash
# Parallel eval across 3 GPUs
# Usage: bash Run_Eval_Parallel.sh [checkpoint_path]

CHECKPOINT=${1:-/data/public/NAS/VLANeXt/output_dir_align_new2}
CONFIG=config/sim_eval_align_config.yaml
TRAIN_CONFIG=config/sim_train_align_config.yaml
NUM_SHARDS=3

echo "=== Parallel Eval: ${NUM_SHARDS} GPUs ==="
echo "Checkpoint: ${CHECKPOINT}"

# Launch 3 shards in parallel
for SHARD in 0 1 2; do
    GPU_ID=${SHARD}
    echo "Starting shard ${SHARD} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m scripts.sim_eval_align_only \
        --config ${CONFIG} \
        --checkpoint ${CHECKPOINT} \
        --train-config ${TRAIN_CONFIG} \
        --shard-id ${SHARD} \
        --num-shards ${NUM_SHARDS} &
done

echo "All shards launched. Waiting..."
wait
echo "All shards complete."

# Merge results
python scripts/merge_eval_shards.py ${CHECKPOINT} --num-shards ${NUM_SHARDS}
