#!/bin/bash
# Auto-eval watchdog: waits for training PIDs to finish, then runs eval on the
# final checkpoint, appends a summary line to docs/EXPERIMENTS_fine_align.md.
#
# Usage:
#   bash scripts/auto_eval_watchdog.sh <train_pid> <ckpt_dir> <train_config_yaml> <gpu_uuid> <run_label>

set -u
TRAIN_PID="$1"
CKPT_DIR="$2"
TRAIN_CFG="$3"
GPU_UUID="$4"
LABEL="$5"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY=$REPO/docs/EXPERIMENTS_fine_align.md
EVAL_LOG=$REPO/logs/auto_eval_${LABEL}_$(date +%H%M%S).log

echo "[$(date +%H:%M:%S)] watchdog: waiting on PID $TRAIN_PID ($LABEL)" >> $EVAL_LOG
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] watchdog: training $LABEL finished. waiting 30s for ckpt flush" >> $EVAL_LOG
sleep 30

CKPTS=()
for cand in checkpoint_10000.pt checkpoint_5000.pt; do
    if [ -f "$CKPT_DIR/$cand" ]; then
        CKPTS+=("$CKPT_DIR/$cand")
    fi
done
if [ "${#CKPTS[@]}" -eq 0 ] && [ -f "$CKPT_DIR/checkpoint_final.pt" ]; then
    CKPTS+=("$CKPT_DIR/checkpoint_final.pt")
fi

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] FATAL: no checkpoint in $CKPT_DIR" >> $EVAL_LOG
    echo "" >> $SUMMARY
    echo "## $LABEL eval FAILED (no ckpt) — $(date)" >> $SUMMARY
    exit 1
fi

if [ -n "${CONDA_SH:-}" ]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-VLANeXt}"
fi

for CKPT in "${CKPTS[@]}"; do
    echo "[$(date +%H:%M:%S)] eval $CKPT" >> $EVAL_LOG
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_UUID \
        MUJOCO_GL=egl \
        __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
        python -m scripts.sim_eval_align_only \
            --config $REPO/config/sim_eval_align_config.yaml \
            --train-config $REPO/$TRAIN_CFG \
            --checkpoint $CKPT \
            --shard-id 0 --num-shards 1 \
            --max-steps 250 --eval-seed 2026 \
            --perturb-mode grid --xy-steps 5 --z-steps 2 --angle-steps 1 --repeats 1 \
            >> $EVAL_LOG 2>&1
    EVAL_EXIT=$?
    {
        echo ""
        echo "### $LABEL eval $(basename $CKPT) ($(date +%Y-%m-%d\ %H:%M))"
        echo "- ckpt: \`$CKPT\`"
        echo "- exit: $EVAL_EXIT"
    } >> $SUMMARY
done
# legacy single-ckpt vars for downstream parse block
CKPT="${CKPTS[-1]}"

# Extract SR / dist / angle from the eval shard JSON if produced
SHARD_JSON=$(ls -t $CKPT_DIR/align_eval_*_shard0/eval_results*.json 2>/dev/null | head -1)
SR_LINE="(parse failed)"
if [ -n "$SHARD_JSON" ] && [ -f "$SHARD_JSON" ]; then
    SR_LINE=$(python3 -c "
import json,sys
d=json.load(open('$SHARD_JSON'))
sr=d.get('success_rate',d.get('sr','?'))
n=d.get('num_episodes',d.get('n','?'))
md=d.get('mean_dist_mm',d.get('mean_final_dist_mm','?'))
ma=d.get('mean_angle_deg',d.get('mean_final_angle_deg','?'))
print(f'SR={sr}  n={n}  mean_dist={md}mm  mean_angle={ma}deg')
" 2>/dev/null) || SR_LINE="(parse failed)"
fi

{
    echo ""
    echo "### $LABEL eval ($(date +%Y-%m-%d\ %H:%M))"
    echo "- ckpt: \`$CKPT\`"
    echo "- exit: $EVAL_EXIT"
    echo "- result: $SR_LINE"
    echo "- log: \`$EVAL_LOG\`"
    if [ -n "$SHARD_JSON" ]; then echo "- json: \`$SHARD_JSON\`"; fi
} >> $SUMMARY

echo "[$(date +%H:%M:%S)] watchdog: done, exit=$EVAL_EXIT" >> $EVAL_LOG
