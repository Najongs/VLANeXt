#!/bin/bash
# Stage 1 sweep: num_inference_timesteps × num_steps_execute on champion v3/1000.
# Goal: isolate diffusion-quantization ceiling (precision diagnosis #1).
# Eval grid: 27-cell (xy 3x3, y 3, z 1, angle 3) @ retreat=2mm, max-steps 250.
set -e
cd /data/public/NAS/VLANeXt

CKPT=/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_aux_strong_v3/checkpoint_1000.pt
TRAIN_CONFIG=config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v3_config.yaml
RESULTS_MD=/tmp/sweep_diff_exec_results.md
SUMMARY_JSON=/tmp/sweep_diff_exec_summary.json

# (diff_steps, exec_steps) cells
CELLS=(
  "10 1"
  "25 1"
  "50 1"
  "100 1"
  "25 2"
  "50 2"
  "25 4"
  "50 4"
)

COMMON_ARGS="--max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
  --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
  --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
  --retreat-mm 2"

# Header
{
  echo "# Stage 1 Sweep — diff_steps × exec_steps on v3/1000 @ retreat=2"
  echo ""
  echo "Started: $(date)"
  echo "Ckpt: $CKPT"
  echo "Grid: 27-cell, max-steps 250, eval-seed 2026, retreat=2mm"
  echo ""
  echo "| diff | exec | SR5mm | SR2mm | SR1mm | mean_min_dist | med_min_dist | med_p90 | time_near_2mm | wall |"
  echo "|------|------|-------|-------|-------|---------------|--------------|---------|---------------|------|"
} > $RESULTS_MD
echo "[]" > $SUMMARY_JSON

PARENT=$(dirname $CKPT)

for cell in "${CELLS[@]}"; do
    read diff exec <<< "$cell"
    echo ""
    echo "=========================================="
    echo "[sweep] cell: diff=$diff exec=$exec  $(date +%H:%M:%S)"
    echo "=========================================="
    t0=$(date +%s)
    TRAIN_CONFIG_OVERRIDE=$TRAIN_CONFIG GPUS=0,1 \
        bash Run_Eval_Parallel.sh align $CKPT \
            --num-inference-timesteps $diff \
            --num-steps-execute $exec \
            $COMMON_ARGS \
        2>&1 | tail -20 || { echo "[sweep] FAILED cell diff=$diff exec=$exec"; continue; }
    t1=$(date +%s)
    wall=$((t1 - t0))

    MERGED="$PARENT/align_eval_step1000_exec${exec}_diff${diff}"
    if [ ! -d "$MERGED" ]; then
        echo "[sweep] WARN merged dir missing $MERGED — falling back to shard0"
        MERGED="${MERGED}_shard0"
    fi

    # Parse metrics via python to avoid awk fragility
    python3 -c "
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/public/NAS/VLANeXt/scripts')
from analyze_trajectory import analyze_episode, summarize
d = Path('$MERGED')
rows = [r for r in (analyze_episode(f) for f in sorted(d.glob('traj_ep*.npz'))) if r is not None]
if not rows:
    print('NO_EPISODES')
    sys.exit(0)
s = summarize(rows, label='$diff/$exec')
# Append a row to results md and summary json
line = '| %s | %s | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.1f%% | %ds |' % (
    '$diff', '$exec',
    s['close_once_5mm_pct'], s['close_once_2mm_pct'], s['close_once_1mm_pct'],
    s['min_dist_mean_mm'], s['min_dist_median_mm'],
    s['p90_dist_median_mm'],
    s['time_near_2mm_median']*100, $wall,
)
with open('$RESULTS_MD', 'a') as f:
    f.write(line + '\n')
existing = json.load(open('$SUMMARY_JSON'))
existing.append({'diff': $diff, 'exec': $exec, 'wall_s': $wall, **{k: float(v) for k,v in s.items() if isinstance(v,(int,float))}})
json.dump(existing, open('$SUMMARY_JSON', 'w'), indent=2)
print(f'[sweep] diff=$diff exec=$exec n={s[\"n_episodes\"]} SR5={s[\"close_once_5mm_pct\"]:.1f}%% SR2={s[\"close_once_2mm_pct\"]:.1f}%% mean={s[\"min_dist_mean_mm\"]:.2f}mm wall=${wall}s')
" || echo "[sweep] parse failed"
done

echo "" >> $RESULTS_MD
echo "Finished: $(date)" >> $RESULTS_MD
echo ""
echo "==========================================="
echo "Sweep complete. Results:"
echo "==========================================="
cat $RESULTS_MD
