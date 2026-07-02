#!/bin/bash
# DCT ablation eval (2026-05-22)
# - 6 ckpts: dct_off_v1 / dct_on_v1 × {500, 1000, 1500}
# - 27-cell grid retreat=2, exec=2 (paper default)
# - GPUs 0,1 sharded (training already done)

set -e
cd /home/najo/NAS/VLANeXt

LOG_DIR=logs/dct_ablation
mkdir -p "$LOG_DIR"

CKPT_DIR_OFF=/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL/dct_off_v1
CKPT_DIR_ON=/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL/dct_on_v1

for STEP in 500 1000 1500; do
  for VARIANT in off on; do
    if [ "$VARIANT" = "off" ]; then
      CKPT_DIR=$CKPT_DIR_OFF
      CFG=config/sim_train_align_dct_off_v1_config.yaml
    else
      CKPT_DIR=$CKPT_DIR_ON
      CFG=config/sim_train_align_dct_on_v1_config.yaml
    fi
    CKPT="${CKPT_DIR}/checkpoint_${STEP}.pt"
    if [ ! -e "$CKPT" ]; then
      echo "SKIP (not found): $CKPT"
      continue
    fi
    echo "=== Eval: dct_${VARIANT}_v1 ck${STEP} exec=2 ==="
    TRAIN_CONFIG_OVERRIDE="$CFG" GPUS=0,1 \
      bash Run_Eval_Parallel.sh align "$CKPT" \
        --max-steps 250 --eval-seed 2026 --perturb-mode grid \
        --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
        --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
        --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
        --retreat-mm 2 --num-steps-execute 2 \
        2>&1 | tee -a "$LOG_DIR/eval_dct_${VARIANT}_ck${STEP}.log"
  done
done

echo "=== DCT ablation eval complete ==="
