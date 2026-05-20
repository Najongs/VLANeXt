#!/bin/bash
# Smart 2-stage overlay_v1 evaluation:
#
# Stage A (sparse ckpt sweep, realistic source):
#   - eval ckpt 1000/2000/3000 with overlay_source=predicted (real-world scenario)
#   - rank by composite precision metric → pick BEST_CKPT
#
# Stage B (ablation on winner):
#   - same BEST_CKPT with overlay_source ∈ {gt, off} (oracle + no-overlay control)
#   - plus v3 baseline (no overlay) for reference
#
# Output: ranked precision comparison saved to /tmp/overlay_smart_rank.md
#
# Usage: bash eval_overlay_smart.sh [overlay_v1_ckpt_dir]
set -uo pipefail
cd /data/public/NAS/VLANeXt

OVERLAY_DIR="${1:-/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_overlay/v1}"
V3_CKPT=/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_aux_strong_v3/checkpoint_1000.pt
UV_CKPT=/data/public/NAS/VLANeXt/checkpoints/keypoint_trocar/uv_only/head_best.pt
DIST_CKPT=/data/public/NAS/VLANeXt/checkpoints/keypoint_trocar/dist_only/head_best.pt
OVERLAY_TRAIN_CONFIG=config/sim_train_align_siglip2_overlay_v1_config.yaml
V3_TRAIN_CONFIG=config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v3_config.yaml
GPUS_OVERRIDE="${EVAL_GPUS:-0}"

# Eval grid: 27-cell @ retreat=2 (same as Stage 1 sweep)
COMMON="--max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
  --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
  --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
  --retreat-mm 2"

run_eval () {
    local label="$1" ckpt="$2" train_cfg="$3" overlay_src="$4" extra="$5"
    echo "[smart] $label  $(date +%H:%M:%S)"

    # Clean target dir to avoid stale-npz contamination
    PARENT=$(dirname $ckpt)
    STEP=$(basename $ckpt .pt | sed 's/checkpoint_//')
    rm -rf $PARENT/align_eval_step${STEP}_exec1_diff10 \
           $PARENT/align_eval_step${STEP}_exec1_diff10_shard0 \
           $PARENT/align_eval_step${STEP}_exec1_diff10_SR* \
           $PARENT/align_eval_step${STEP}_exec1_diff10_shard0_SR* 2>/dev/null

    TRAIN_CONFIG_OVERRIDE=$train_cfg GPUS=$GPUS_OVERRIDE \
        bash Run_Eval_Parallel.sh align $ckpt \
            --overlay-source $overlay_src \
            $COMMON $extra \
        > /tmp/smart_${label}.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[smart] EVAL FAILED rc=$rc: $label"
        tail -20 /tmp/smart_${label}.log
    fi
    return $rc
}

ckpt_dir_for () {
    local ckpt="$1"
    PARENT=$(dirname $ckpt); STEP=$(basename $ckpt .pt | sed 's/checkpoint_//')
    local d=$PARENT/align_eval_step${STEP}_exec1_diff10
    if [ ! -d "$d" ]; then d=${d}_shard0; fi
    if [ ! -d "$d" ]; then d=$(ls -d ${PARENT}/align_eval_step${STEP}_exec1_diff10* 2>/dev/null | head -1); fi
    echo "$d"
}

# --- Stage A: sparse ckpt sweep with predicted UV ---
declare -a STAGE_A_CKPTS=(
    "${OVERLAY_DIR}/checkpoint_1000.pt"
    "${OVERLAY_DIR}/checkpoint_2000.pt"
    "${OVERLAY_DIR}/checkpoint_3000.pt"
)
declare -a STAGE_A_DIRS=()
declare -a STAGE_A_LABELS=()

for ckpt in "${STAGE_A_CKPTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "[smart] skip missing $ckpt"
        continue
    fi
    step=$(basename $ckpt .pt | sed 's/checkpoint_//')
    label="ovlPred_step${step}"
    run_eval "$label" "$ckpt" "$OVERLAY_TRAIN_CONFIG" "predicted" \
        "--uv-ckpt $UV_CKPT --dist-ckpt $DIST_CKPT --no-kp-seed-handoff --no-kp-inline-trigger" \
        || continue
    d=$(ckpt_dir_for "$ckpt")
    STAGE_A_DIRS+=("$d")
    STAGE_A_LABELS+=("$label")
done

# Also add v3 baseline as reference
run_eval "v3_baseline" "$V3_CKPT" "$V3_TRAIN_CONFIG" "off" "" || true
V3_DIR=$(ckpt_dir_for "$V3_CKPT")

echo ""
echo "==========================================="
echo "Stage A — overlay_v1 ckpt sweep (predicted UV) + v3 baseline"
echo "==========================================="
python -m scripts.rank_models "$V3_DIR" "${STAGE_A_DIRS[@]}" \
    --labels v3_baseline "${STAGE_A_LABELS[@]}" \
    --out /tmp/overlay_smart_stageA.md || true

# --- Stage B: ablation on best ckpt (oracle GT + no-overlay) ---
if [ ${#STAGE_A_DIRS[@]} -gt 0 ]; then
    # Pick winner by rank (smallest rank_sum)
    BEST_LABEL=$(python -c "
import sys
sys.path.insert(0, '/data/public/NAS/VLANeXt/scripts')
from rank_models import load_eval_dir, rank_models
from pathlib import Path
dirs = ['$V3_DIR'] + '''${STAGE_A_DIRS[@]}'''.split()
labels = ['v3_baseline'] + '''${STAGE_A_LABELS[@]}'''.split()
sums = []
for d, l in zip(dirs, labels):
    s = load_eval_dir(Path(d))
    if s is None: continue
    s['_lbl'] = l
    sums.append(s)
rank_models(sums)
# Winner among overlay ckpts only (skip v3 baseline)
overlay_only = [s for s in sums if s['_lbl'].startswith('ovlPred_')]
if overlay_only:
    print(overlay_only[0]['_lbl'])
" 2>/dev/null)
    if [ -z "$BEST_LABEL" ]; then BEST_LABEL="ovlPred_step3000"; fi
    BEST_STEP="${BEST_LABEL#ovlPred_step}"
    BEST_CKPT=${OVERLAY_DIR}/checkpoint_${BEST_STEP}.pt
    echo ""
    echo "==========================================="
    echo "Stage A winner: $BEST_LABEL  → Stage B ablation"
    echo "==========================================="

    run_eval "ovlGT_step${BEST_STEP}"  "$BEST_CKPT" "$OVERLAY_TRAIN_CONFIG" "gt"  "" || true
    GT_DIR=$(ckpt_dir_for "$BEST_CKPT")
    # off-mode needs a different output namespace — but eval dir base name is identical
    # so we must run sequentially and snapshot
    # Workaround: rename gt result first, then run off
    if [ -d "$GT_DIR" ]; then mv "$GT_DIR" "${GT_DIR}_gt"; GT_DIR="${GT_DIR}_gt"; fi
    run_eval "ovlOff_step${BEST_STEP}" "$BEST_CKPT" "$OVERLAY_TRAIN_CONFIG" "off" "" || true
    OFF_DIR=$(ckpt_dir_for "$BEST_CKPT")
    if [ -d "$OFF_DIR" ]; then mv "$OFF_DIR" "${OFF_DIR}_off"; OFF_DIR="${OFF_DIR}_off"; fi

    BEST_DIR=$(ckpt_dir_for "$BEST_CKPT")
    # Stage A's predicted dir might no longer exist if renamed — find under STAGE_A_DIRS
    PRED_DIR=""
    for i in "${!STAGE_A_LABELS[@]}"; do
        if [ "${STAGE_A_LABELS[$i]}" = "$BEST_LABEL" ]; then
            PRED_DIR="${STAGE_A_DIRS[$i]}"; break
        fi
    done

    echo ""
    echo "==========================================="
    echo "Stage B — Final precision table"
    echo "==========================================="
    python -m scripts.rank_models \
        "$V3_DIR" "$PRED_DIR" "$GT_DIR" "$OFF_DIR" \
        --labels v3_baseline "$BEST_LABEL(predicted)" "$BEST_LABEL(gt)" "$BEST_LABEL(off)" \
        --out /tmp/overlay_smart_final.md || true
fi

echo ""
echo "All done at $(date)"
