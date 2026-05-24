#!/bin/bash
# Baseline matrix eval (2026-05-22)
# 5 baselines @ retreat=2, exec=2 (VLA family), exec=1 (lerobot family).
#  - DINOv3 fresh (ours, encoder ablation)
#  - SigLIP2 fresh (ours, head-to-head with DINOv3)
#  - ConvNeXt unfreeze v5b ck1500 (already done — skip, just collect)
#  - ACT final (lerobot in-house) — retreat=2 rerun
#  - DP  final (lerobot in-house) — retreat=2 rerun
# Plus SigLIP2 minLat champion (lat_hold_v4_yneg_hold) for reference (already done).

set -e
cd /data/public/NAS/VLANeXt
LOG_DIR=logs/baseline_matrix
mkdir -p "$LOG_DIR"

run_eval() {
  local LABEL=$1
  local CFG=$2
  local CKPT=$3
  local EXEC=$4
  echo "=== Eval: $LABEL exec=$EXEC ==="
  if [ ! -e "$CKPT" ]; then
    echo "SKIP (not found): $CKPT"
    return 0
  fi
  TRAIN_CONFIG_OVERRIDE="$CFG" GPUS=0,1 \
    bash Run_Eval_Parallel.sh align "$CKPT" \
      --max-steps 250 --eval-seed 2026 --perturb-mode grid \
      --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
      --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
      --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
      --retreat-mm 2 --num-steps-execute "$EXEC" \
      2>&1 | tee -a "$LOG_DIR/eval_${LABEL}.log"
}

# === Ours: DINOv3 fresh ===
run_eval "dinov3_fresh" \
  config/sim_train_align_dinov3_baseline_v1_config.yaml \
  /data/public/NAS/VLANeXt/checkpoints/VLANeXt_DINOv3_baseline/v1/checkpoint_1500.pt 2

# === Ours: SigLIP2 fresh ===
run_eval "siglip2_fresh" \
  config/sim_train_align_siglip2_baseline_v1_config.yaml \
  /data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_baseline/v1/checkpoint_1500.pt 2

# === ACT (lerobot in-house) ===
run_eval "act_final" \
  config/sim_train_act_baseline_config.yaml \
  /data/public/NAS/VLANeXt/checkpoints/ACT_baseline_align/checkpoint_30000.pt 1

# === DP (lerobot in-house) ===
run_eval "dp_final" \
  config/sim_train_dp_baseline_config.yaml \
  /data/public/NAS/VLANeXt/checkpoints/DP_baseline_align/checkpoint_30000.pt 1

echo "=== Baseline matrix eval complete ==="
