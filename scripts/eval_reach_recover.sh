#!/bin/bash
set -e
cd /data/public/NAS/VLANeXt
LOG_DIR=logs/reach_recover
mkdir -p "$LOG_DIR"

CFG=config/sim_train_align_reach_recover_v1_config.yaml
CKPT_DIR=/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL/reach_recover_v1

for STEP in 500 1000 1500; do
  CKPT="${CKPT_DIR}/checkpoint_${STEP}.pt"
  echo "=== Eval: reach_recover_v1 ck${STEP} exec=2 ==="
  if [ ! -e "$CKPT" ]; then echo "SKIP: $CKPT"; continue; fi
  TRAIN_CONFIG_OVERRIDE="$CFG" GPUS=0,1 \
    bash Run_Eval_Parallel.sh align "$CKPT" \
      --max-steps 250 --eval-seed 2026 --perturb-mode grid \
      --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
      --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
      --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
      --retreat-mm 2 --num-steps-execute 2 \
      2>&1 | tee -a "$LOG_DIR/eval_ck${STEP}.log"
done

echo "=== reach_recover eval complete ==="
