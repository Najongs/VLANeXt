#!/bin/bash
# Eval a list of checkpoints on the 27-cell ang±5° paper grid.
# Usage:
#   bash scripts/eval_multi_ckpt.sh <train_config> <ckpt1> [ckpt2 ...]
# Example:
#   bash scripts/eval_multi_ckpt.sh \
#       config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v2_config.yaml \
#       checkpoints/VLANeXt_SigLIP2_repro_b24_ft10mm_aux_strong_v2/checkpoint_3000.pt \
#       checkpoints/VLANeXt_SigLIP2_repro_b24_ft10mm_aux_strong_v2/checkpoint_5000.pt
#
# 27-cell grid: xy={-10,0,+10}, y={-25,0,+25}, z=0, angle={-5,0,+5}, repeats=1
# Each ckpt: 27 episodes × ~5sec eval = ~2.5 min per ckpt on 2 GPUs.

set -e
cd "$(dirname "$0")/.."

TRAIN_CONFIG="$1"
shift
CKPTS=("$@")

if [ -z "$TRAIN_CONFIG" ] || [ ${#CKPTS[@]} -eq 0 ]; then
    echo "Usage: $0 <train_config> <ckpt1> [ckpt2 ...]"
    exit 1
fi

for CKPT in "${CKPTS[@]}"; do
    if [ ! -e "$CKPT" ]; then
        echo "SKIP (not found): $CKPT"
        continue
    fi
    echo "=== Eval: $CKPT ==="
    TRAIN_CONFIG_OVERRIDE="$TRAIN_CONFIG" \
        GPUS=0,1 \
        bash Run_Eval_Parallel.sh align "$CKPT" \
            --max-steps 250 --eval-seed 2026 --perturb-mode grid \
            --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
            --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
            --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0
done

echo "=== All eval done ==="
