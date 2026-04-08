#!/bin/bash
# Parallel eval across 3 GPUs
# Usage:
#   bash Run_Eval_Parallel.sh [checkpoint_path] [--randomize-phantom]
#
# Examples:
#   bash Run_Eval_Parallel.sh                                    # 기본 (고정 phantom)
#   bash Run_Eval_Parallel.sh /data/public/NAS/VLANeXt/output_dir_align_0408                   # 특정 체크포인트
#   bash Run_Eval_Parallel.sh /data/public/NAS/VLANeXt/output_dir_align_0408 --randomize-phantom # 랜덤 phantom

"""
Phase 1: Micro-Alignment (가까이에서 정렬)

고정 팬텀 (현재 완료)
bash Run_Collect.sh uniform 10 1000          # 10,000개, 팬텀 고정
- 스크립트: Save_dataset_align_only.py
- 가까운 perturbation(±30mm XY, ±20mm Z)에서 trocar 정렬

Phase 2: Spatial Generalization (랜덤 팬텀 정렬)

Y축 5위치에서 micro-alignment
bash Run_Collect.sh multi_phantom 10 2000    # 5위치 × 20,000 = 100,000개
- 스크립트: Save_dataset_align_only.py + --phantom-pos
- 팬텀 위치별: Y=0.0, -0.1, -0.2, -0.3, -0.4 (X=0 고정)
- 각 위치에서 가까운 perturbation → 정렬 녹화

Phase 3: Out-of-view Approach (먼 거리 접근)

Y축 5위치에서 approach+align (insertion 제외)
bash Run_Collect.sh approach 10 2000         # 5위치 × 20,000 = 100,000개
- 스크립트: Save_dataset.py + --phantom-pos + --no-insertion
- 먼 거리(home pose)에서 trocar까지 접근 + 정렬, 삽입 직전에 종료

Phase 4: Insertion (삽입)

Y축 5위치에서 full pipeline
bash Run_Collect.sh full 10 2000             # 5위치 × 20,000 = 100,000개
- 스크립트: Save_dataset.py + --phantom-pos
- approach + align + insert 전체 trajectory

"""


CHECKPOINT=${1:-/data/public/NAS/VLANeXt/output_dir_align_0408}
CONFIG=config/sim_eval_align_config.yaml
TRAIN_CONFIG=config/sim_train_align_config.yaml
NUM_SHARDS=3

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
for SHARD in 0 1 2; do
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
