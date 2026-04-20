#!/bin/bash
# Parallel eval across GPUs
# Usage:
#   bash Run_Eval_Parallel.sh [mode] [checkpoint_path] [extra_flags...]
#   bash Run_Eval_Parallel.sh [checkpoint_path] [extra_flags...]    (mode defaults to align)
#
# Modes:
#   align     - Fine-alignment eval (기본)
#   approach  - Approach eval (먼 거리 → 트로카 접근)
#   insertion - Insertion eval (정렬 후 삽입)
#
# NOTE: Perturbation Z 범위
#   align 데이터 수집/eval에서 Z perturbation은 [0, +Z_MAX]mm만 사용.
#   Z < 0이면 tool_camera에서 바늘 팁이 팬텀(눈 모형)에 가려짐.
#   (occlusion grid 테스트 결과: dataset/occlusion_grid/, dataset/occlusion_grid_angle/)
#   관련 파일: Save_dataset_align_only.py, sim_eval_align_only.py, run_parallel.py
#
# Examples:
#   bash Run_Eval_Parallel.sh align /data/public/NAS/VLANeXt/output_dir_align_0417
#   bash Run_Eval_Parallel.sh /path/to/checkpoint --sensor-success
#   bash Run_Eval_Parallel.sh align /path/to/checkpoint --randomize-phantom --sensor-success
#   bash Run_Eval_Parallel.sh approach /data/public/NAS/VLANeXt/output_dir_approach_0414
#   bash Run_Eval_Parallel.sh insertion /data/public/NAS/VLANeXt/output_dir_insertion_0415

# Auto-detect: if first arg starts with / or . it's a checkpoint path, not a mode
if [[ "$1" == /* ]] || [[ "$1" == .* ]]; then
    MODE="align"
    CHECKPOINT="$1"
    EXTRA_ARGS=("${@:2}")
elif [ "$1" = "align" ] || [ "$1" = "approach" ] || [ "$1" = "insertion" ]; then
    MODE="$1"
    CHECKPOINT="${2:-/data/public/NAS/VLANeXt/output_dir_align_0410}"
    EXTRA_ARGS=("${@:3}")
else
    MODE="align"
    CHECKPOINT="${1:-/data/public/NAS/VLANeXt/output_dir_align_0410}"
    EXTRA_ARGS=("${@:2}")
fi

NUM_SHARDS=2

# Mode-specific config
if [ "$MODE" = "approach" ]; then
    CONFIG=config/sim_eval_approach_config.yaml
    TRAIN_CONFIG=config/sim_train_approach_config.yaml
    EVAL_SCRIPT=scripts.sim_eval_approach_only
    MERGE_PREFIX="approach"
elif [ "$MODE" = "insertion" ]; then
    CONFIG=config/sim_eval_insertion_config.yaml
    TRAIN_CONFIG=config/sim_train_insertion_config.yaml
    EVAL_SCRIPT=scripts.sim_eval_insertion_only
    MERGE_PREFIX="insertion"
else
    CONFIG=config/sim_eval_align_config.yaml
    TRAIN_CONFIG=config/sim_train_align_config.yaml
    EVAL_SCRIPT=scripts.sim_eval_align_only
    MERGE_PREFIX="align"
fi

# Build extra flags string
EXTRA_FLAGS=""
for arg in "${EXTRA_ARGS[@]}"; do
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
if [[ "${EXTRA_FLAGS}" == *"--sensor-success"* ]]; then
    echo "Sensor Success: ON"
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
