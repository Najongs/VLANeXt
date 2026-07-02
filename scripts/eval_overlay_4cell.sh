#!/bin/bash
# 4-cell eval comparing v3/1000 baseline vs overlay_v1 (gt / predicted / off).
# Usage: bash eval_overlay_4cell.sh <overlay_v1_ckpt_path>
#
# Outputs: each cell creates a separate eval dir.
# Final comparison printed at end.
set -e
cd /home/najo/NAS/VLANeXt

OVERLAY_CKPT="${1:-/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_overlay/v1/checkpoint_3000.pt}"
V3_CKPT=/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_aux_strong_v3/checkpoint_1000.pt
UV_CKPT=/home/najo/NAS/VLANeXt/checkpoints/keypoint_trocar/uv_only/best.pt
DIST_CKPT=/home/najo/NAS/VLANeXt/checkpoints/keypoint_trocar/dist_only/best.pt
OVERLAY_TRAIN_CONFIG=config/sim_train_align_siglip2_overlay_v1_config.yaml
V3_TRAIN_CONFIG=config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v3_config.yaml

# Common eval args (27-cell @ retreat=2)
COMMON="--max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
  --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
  --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
  --retreat-mm 2"

run_cell () {
    local label="$1" ckpt="$2" train_cfg="$3" overlay_src="$4" extra="$5"
    echo ""
    echo "=========================================="
    echo "[overlay_eval] cell: $label  $(date +%H:%M:%S)"
    echo "=========================================="
    TRAIN_CONFIG_OVERRIDE=$train_cfg GPUS=${EVAL_GPUS:-0,1} \
        bash Run_Eval_Parallel.sh align $ckpt \
            --overlay-source $overlay_src \
            $COMMON $extra \
        2>&1 | tail -25 || echo "[overlay_eval] FAILED: $label"
}

# Cell 1: v3 baseline (no overlay, existing model) — sanity reference
run_cell "v3_baseline" "$V3_CKPT" "$V3_TRAIN_CONFIG" "off" ""

# Cell 2: overlay_v1 + GT UV (oracle ceiling)
run_cell "overlay_gt" "$OVERLAY_CKPT" "$OVERLAY_TRAIN_CONFIG" "gt" ""

# Cell 3: overlay_v1 + predicted UV (realistic deploy)
run_cell "overlay_predicted" "$OVERLAY_CKPT" "$OVERLAY_TRAIN_CONFIG" "predicted" \
    "--uv-ckpt $UV_CKPT --dist-ckpt $DIST_CKPT --no-kp-seed-handoff --no-kp-inline-trigger"

# Cell 4: overlay_v1 + no overlay (ablation: how dependent on overlay?)
run_cell "overlay_off" "$OVERLAY_CKPT" "$OVERLAY_TRAIN_CONFIG" "off" ""

echo ""
echo "==========================================="
echo "All 4 cells done. Summary:"
echo "==========================================="
python3 - << EOF
from pathlib import Path
import sys
sys.path.insert(0, '/home/najo/NAS/VLANeXt/scripts')
from analyze_trajectory import analyze_episode, summarize

V3_CKPT = "$V3_CKPT"
OVERLAY_CKPT = "$OVERLAY_CKPT"

# Map: label -> (parent_dir, dir_name_pattern)
candidates = {
    "v3_baseline": (Path(V3_CKPT).parent, "align_eval_step1000_exec1_diff10"),
    "overlay_gt": (Path(OVERLAY_CKPT).parent, "align_eval_step3000_exec1_diff10"),
    "overlay_predicted": (Path(OVERLAY_CKPT).parent, "align_eval_step3000_exec1_diff10"),
    "overlay_off": (Path(OVERLAY_CKPT).parent, "align_eval_step3000_exec1_diff10"),
}

print(f"{'cell':<22}  {'n':>3}  {'SR5':>5}  {'SR2':>5}  {'SR1':>5}  {'mean':>6}  {'med':>6}  {'p90':>6}")
print('-' * 75)
for label, (parent, dpat) in candidates.items():
    matches = sorted(parent.glob(f"{dpat}*"))
    if not matches:
        print(f"{label:<22}  MISSING ({parent}/{dpat})")
        continue
    d = matches[-1]
    rows = [r for r in (analyze_episode(f) for f in sorted(d.glob('traj_ep*.npz'))) if r is not None]
    if not rows:
        print(f"{label:<22}  NO EPISODES ({d.name})")
        continue
    s = summarize(rows, label=label)
    print(f"{label:<22}  {s['n_episodes']:>3}  {s['close_once_5mm_pct']:>4.1f}%  {s['close_once_2mm_pct']:>4.1f}%  {s['close_once_1mm_pct']:>4.1f}%  {s['min_dist_mean_mm']:>5.2f}  {s['min_dist_median_mm']:>5.2f}  {s['p90_dist_median_mm']:>5.2f}")
EOF
