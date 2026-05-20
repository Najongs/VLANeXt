#!/bin/bash
# NEARGOAL Dual Track (2026-05-20)
# Track A: pos+angle 5mm/5° eval-matched, 3000 ep (~4-5h)
# Track B: angle-only 15° specialized, 1000 ep (~1.5h)
# 두 데이터 모두 → champion v3 finetune cotrain mix

set -e
cd "$(dirname "$0")"

# MuJoCo rendering: Mesa EGL (GPU 0 dead, NVIDIA EGL hang 회피)
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

LOG_DIR=/data/public/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

# === Track A: NEARGOAL_eval_match_v2 ===
# phantom range = eval grid + 20% margin (x±12, y±29, angle±7°)
# perturb = default (5mm XY, ±5mm Z, 5° angle)
# hold_record_steps = 60 (v3 standard 30의 2배)
echo "Launching Track A (pos+angle, 3000 ep)..."
python run_parallel.py --script align --workers 10 --episodes 300 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_eval_match_v2 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 60 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --seed 2027 > "$LOG_DIR/datagen_neargoal_v2_trackA.log" 2>&1 &
PID_A=$!
echo "Track A: PID $PID_A, log: $LOG_DIR/datagen_neargoal_v2_trackA.log"

sleep 2

# === Track B: NEARGOAL_angle_only_v2 ===
# 동일 phantom range
# perturb XY=0, Z=0, angle=15° (angle 교정만 specialized)
echo "Launching Track B (angle-only 15°, 1000 ep)..."
python run_parallel.py --script align --workers 10 --episodes 100 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_angle_only_v2 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 60 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --perturb-xy-mm 0 --perturb-z-min-mm 0 --perturb-z-max-mm 0 --perturb-angle-deg 15 \
    --seed 2028 > "$LOG_DIR/datagen_neargoal_v2_trackB.log" 2>&1 &
PID_B=$!
echo "Track B: PID $PID_B, log: $LOG_DIR/datagen_neargoal_v2_trackB.log"

echo ""
echo "Both tracks launched. Monitoring..."
echo "  Track A: NEARGOAL_eval_match_v2 (pos+angle 5mm, 3000ep, hold 60)"
echo "  Track B: NEARGOAL_angle_only_v2 (angle-only 15°, 1000ep, hold 60)"
echo ""
echo "wait_for_both () to block until both complete."

wait_for_both() {
    wait $PID_A
    echo "[$(date +%H:%M:%S)] Track A complete"
    wait $PID_B
    echo "[$(date +%H:%M:%S)] Track B complete"
}

# If invoked directly (not sourced), wait for both. Otherwise return PIDs to caller.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    wait_for_both
fi
