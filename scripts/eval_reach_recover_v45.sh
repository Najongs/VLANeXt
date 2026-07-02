#!/bin/bash
set -e
cd /home/najo/NAS/VLANeXt
LOG_DIR=logs/reach_recover
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

# v4 longer (lr 1e-6, 5000 step) — eval 3000/4000/5000 (saturation curve)
for STEP in 3000 4000 5000; do
  run_eval "v4_ck${STEP}" config/sim_train_align_reach_recover_v4_longer_config.yaml \
    /home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL/reach_recover_v4_longer/checkpoint_${STEP}.pt
done

# v5 combo (lr 1e-6 + softhold) — eval 1500/2000/3000
for STEP in 1500 2000 3000; do
  run_eval "v5_ck${STEP}" config/sim_train_align_reach_recover_v5_combo_config.yaml \
    /home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_${STEP}.pt
done

echo "=== reach_recover v4/v5 eval complete ==="
