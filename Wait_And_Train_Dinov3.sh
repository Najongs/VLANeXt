#!/bin/bash
# 현재 진행중인 학습이 끝나길 기다렸다가 DINOv3 학습 자동 시작.
# 백그라운드 실행 권장: `nohup bash Wait_And_Train_Dinov3.sh > logs/dinov3_watcher.log 2>&1 &`
#
# 감지 방식: 현재 돌고 있는 `scripts.train --config .../siglip2...` 프로세스 PID를 잡아서 wait.
# 없으면 즉시 시작.
#
# DINOv3 학습 GPU: 환경변수 GPU_UUID로 override. default는 살아있는 GPU1 (PCI 2F).

set -e
cd /data/public/NAS/VLANeXt

# conda env activate — 그 안 하면 lerobot env로 떨어져서 tensorflow 등 누락.
source /home/yohan/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

# 현재 돌고 있는 SigLIP2 train PID 탐색 — dataloader worker 빼고 부모(가장 작은) PID만.
PID=$(pgrep -f "scripts.train --config.*siglip2" | sort -n | head -1 || true)
if [ -n "$PID" ]; then
    echo "[$(date)] SigLIP2 학습 발견 (parent PID=$PID), 종료 대기..."
    tail --pid="$PID" -f /dev/null
    echo "[$(date)] SigLIP2 학습 종료 감지"
else
    echo "[$(date)] 진행중인 SigLIP2 학습 없음, 즉시 DINOv3 시작"
fi

# DINOv3가 SigLIP2와 같은 GPU 쓸 거니까, SigLIP2가 정말 GPU 해제했는지 짧게 대기
sleep 30

GPU_UUID="${GPU_UUID:-GPU-ab38c04c-0adf-17eb-fc9f-fab2e28559f5}"
echo "[$(date)] === DINOv3 학습 시작 (GPU=$GPU_UUID) ==="
mkdir -p logs
CUDA_VISIBLE_DEVICES="$GPU_UUID" python -m scripts.train \
    --config config/sim_train_align_dinov3_config.yaml \
    2>&1 | tee logs/train_dinov3_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] === DINOv3 학습 완료 ==="
