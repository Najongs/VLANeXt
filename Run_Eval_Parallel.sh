#!/bin/bash
# Parallel eval across 3 GPUs
# Usage:
#   bash Run_Eval_Parallel.sh [checkpoint_path] [--randomize-phantom]
#
# Examples:
#   bash Run_Eval_Parallel.sh                                    # 기본 (고정 phantom)
#   bash Run_Eval_Parallel.sh /data/public/NAS/VLANeXt/output_dir_align_0408                   # 특정 체크포인트
#   bash Run_Eval_Parallel.sh /data/public/NAS/VLANeXt/output_dir_align_0408 --randomize-phantom # 랜덤 phantom


CHECKPOINT=${1:-/data/public/NAS/VLANeXt/output_dir_align_0409}
CONFIG=config/sim_eval_align_config.yaml
TRAIN_CONFIG=config/sim_train_align_config.yaml
NUM_SHARDS=2

# Parse extra flags (--randomize-phantom)
EXTRA_FLAGS=""
for arg in "${@:2}"; do
    EXTRA_FLAGS="${EXTRA_FLAGS} ${arg}"
done

echo "=== Parallel Eval: ${NUM_SHARDS} GPUs ==="
echo "Checkpoint: ${CHECKPOINT}"
if [[ "${EXTRA_FLAGS}" == *"--randomize-phantom"* ]]; then
    echo "Phantom: RANDOMIZED"
else
    echo "Phantom: FIXED"
fi

# Launch 3 shards in parallel
for SHARD in 0 1; do
    GPU_ID=${SHARD}
    echo "Starting shard ${SHARD} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m scripts.sim_eval_align_only \
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
python scripts/merge_eval_shards.py ${CHECKPOINT} --num-shards ${NUM_SHARDS}
