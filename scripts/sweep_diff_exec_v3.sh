#!/bin/bash
# Stage 1 sweep v2: single-shard on GPU 2 only (CUDA 1), full per-cell logs, dir cleanup.
# Fixes from v1:
#   - GPU 2 only (GPU 1 = overlay_v1 full train concurrent)
#   - No "| tail -20" truncation — save full log per cell
#   - Clean target dir before each cell to avoid stale-npz contamination
#   - Wall-time recorded BEFORE result parse
set -uo pipefail
cd /home/najo/NAS/VLANeXt

CKPT=/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_aux_strong_v3/checkpoint_1000.pt
TRAIN_CONFIG=config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v3_config.yaml
RESULTS_MD=/tmp/sweep_diff_exec_v3_results.md
SUMMARY_JSON=/tmp/sweep_diff_exec_v3_summary.json
LOGDIR=/tmp/sweep_v3_logs
mkdir -p $LOGDIR

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

{
  echo "# Stage 1 Sweep v3 — diff_steps × exec_steps on v3/1000 @ retreat=2"
  echo ""
  echo "Started: $(date)"
  echo "Ckpt: $CKPT"
  echo "Grid: 27-cell single-shard (GPU 2 / CUDA 1)"
  echo ""
  echo "| diff | exec | n  | SR5mm | SR2mm | SR1mm | mean_min | med_min | med_p90 | time_2mm | wall |"
  echo "|------|------|----|-------|-------|-------|----------|---------|---------|----------|------|"
} > $RESULTS_MD
echo "[]" > $SUMMARY_JSON

PARENT=$(dirname $CKPT)

for cell in "${CELLS[@]}"; do
    read diff exec <<< "$cell"
    cell_label="diff${diff}_exec${exec}"
    echo ""
    echo "=========================================="
    echo "[sweep_v3] $cell_label  $(date +%H:%M:%S)"
    echo "=========================================="

    # Clean target dirs to avoid stale-npz contamination
    rm -rf $PARENT/align_eval_step1000_exec${exec}_diff${diff} \
           $PARENT/align_eval_step1000_exec${exec}_diff${diff}_shard0 \
           $PARENT/align_eval_step1000_exec${exec}_diff${diff}_SR* \
           $PARENT/align_eval_step1000_exec${exec}_diff${diff}_shard0_SR* 2>/dev/null

    t0=$(date +%s)
    TRAIN_CONFIG_OVERRIDE=$TRAIN_CONFIG GPUS=1 \
        bash Run_Eval_Parallel.sh align $CKPT \
            --num-inference-timesteps $diff \
            --num-steps-execute $exec \
            $COMMON_ARGS \
        > $LOGDIR/${cell_label}.log 2>&1
    rc=$?
    t1=$(date +%s)
    wall=$((t1 - t0))

    if [ $rc -ne 0 ]; then
        echo "[sweep_v3] EVAL FAILED rc=$rc cell=$cell_label  (see $LOGDIR/${cell_label}.log)"
        tail -30 $LOGDIR/${cell_label}.log
        echo "| $diff | $exec | -  | FAIL  | FAIL  | FAIL  | -        | -       | -       | -        | ${wall}s |" >> $RESULTS_MD
        continue
    fi

    # GPUS=1 → single shard → eval dir = base (no _shard0 merge) OR _shard0
    # merge_eval_shards.py runs even with 1 shard so base dir exists.
    MERGED="$PARENT/align_eval_step1000_exec${exec}_diff${diff}"
    if [ ! -d "$MERGED" ]; then
        echo "[sweep_v3] WARN merged missing — try _shard0"
        MERGED="${MERGED}_shard0"
    fi
    # Final fallback: check _SR variants
    if [ ! -d "$MERGED" ]; then
        MERGED=$(ls -d ${PARENT}/align_eval_step1000_exec${exec}_diff${diff}* 2>/dev/null | head -1)
    fi
    if [ -z "$MERGED" ] || [ ! -d "$MERGED" ]; then
        echo "[sweep_v3] no result dir found"
        echo "| $diff | $exec | 0  | NORES | NORES | NORES | -        | -       | -       | -        | ${wall}s |" >> $RESULTS_MD
        continue
    fi

    python3 - <<EOF
import sys, json
from pathlib import Path
sys.path.insert(0, '/home/najo/NAS/VLANeXt/scripts')
from analyze_trajectory import analyze_episode, summarize
d = Path('$MERGED')
rows = [r for r in (analyze_episode(f) for f in sorted(d.glob('traj_ep*.npz'))) if r is not None]
if not rows:
    print('NO_EPISODES — empty merged dir')
    sys.exit(1)
s = summarize(rows, label='diff${diff}_exec${exec}')
line = '| ${diff} | ${exec} | %d | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.1f%% | ${wall}s |' % (
    s['n_episodes'],
    s['close_once_5mm_pct'], s['close_once_2mm_pct'], s['close_once_1mm_pct'],
    s['min_dist_mean_mm'], s['min_dist_median_mm'],
    s['p90_dist_median_mm'],
    s['time_near_2mm_median']*100,
)
with open('$RESULTS_MD', 'a') as f: f.write(line + '\n')
existing = json.load(open('$SUMMARY_JSON'))
existing.append({'diff': ${diff}, 'exec': ${exec}, 'wall_s': ${wall},
                 **{k: float(v) for k,v in s.items() if isinstance(v,(int,float))}})
json.dump(existing, open('$SUMMARY_JSON', 'w'), indent=2)
print(f'[sweep_v3] diff=${diff} exec=${exec} n={s["n_episodes"]} '
      f'SR5={s["close_once_5mm_pct"]:.1f}% SR2={s["close_once_2mm_pct"]:.1f}% '
      f'mean={s["min_dist_mean_mm"]:.2f}mm wall=${wall}s')
EOF
    if [ $? -ne 0 ]; then
        echo "[sweep_v3] parse failed for $cell_label"
        echo "| $diff | $exec | -  | PARSE | PARSE | PARSE | -        | -       | -       | -        | ${wall}s |" >> $RESULTS_MD
    fi
done

echo "" >> $RESULTS_MD
echo "Finished: $(date)" >> $RESULTS_MD
echo ""
echo "==========================================="
echo "Sweep v3 complete. Full results:"
echo "==========================================="
cat $RESULTS_MD
