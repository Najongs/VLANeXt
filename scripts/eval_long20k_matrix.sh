#!/bin/bash
# Long20k baseline matrix eval (DINOv3 + ConvNeXt, ck20000)
set -e
cd /home/najo/NAS/VLANeXt
LOG_DIR=logs/baseline_long20k
mkdir -p "$LOG_DIR"

run_eval() {
  local LABEL=$1; local CFG=$2; local CKPT=$3
  echo "=== Eval: $LABEL ==="
  if [ ! -e "$CKPT" ]; then echo "SKIP: $CKPT"; return 0; fi
  TRAIN_CONFIG_OVERRIDE="$CFG" GPUS=0,1 \
    bash Run_Eval_Parallel.sh align "$CKPT" \
      --max-steps 250 --eval-seed 2026 --perturb-mode grid \
      --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
      --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
      --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
      --retreat-mm 2 --num-steps-execute 2 \
      2>&1 | tee -a "$LOG_DIR/eval_${LABEL}.log"
}

run_eval "dinov3_long20k" config/sim_train_align_dinov3_long20k_v1_config.yaml \
  /home/najo/NAS/VLANeXt/checkpoints/VLANeXt_DINOv3_long20k/v1/checkpoint_20000.pt

run_eval "convnext_long20k" config/sim_train_align_convnext_long20k_v1_config.yaml \
  /home/najo/NAS/VLANeXt/checkpoints/VLANeXt_ConvNeXt_long20k/v1/checkpoint_20000.pt

echo "=== Long20k eval complete ==="
