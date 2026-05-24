#!/bin/bash
# Sub-mm sustained-hold dataset (2026-05-24, EXPERIMENTS Section 32)
# 목적: holdSR 52% ceiling 깨기 + min_lat <0.87mm sub-mm 천장 깨기.
#
# 진단: 기존 perfect_strict_v1 (1mm perturb + 150 hold) 도 800ep로 ceiling 못 깸.
#       Section 31 R1-R5에서 6 variants × 14 ckpts 모두 holdSR ~37-52% 정체.
#
# 가설: hold 학습 신호가 "충분히 길지 않다" + 데이터가 "충분히 sub-mm tight 하지 않다".
#       이 dataset = 1.5mm 극도 tight perturb + 250 step hold (기존 60-150의 1.7-4x) + 1800ep (perfect_strict 2x+).
#
# 사양:
# - script align_neargoal (NEARGOAL series 그대로)
# - phantom range: x ±10, y ±25, angle ±5° (eval grid EXACTLY 일치 — distribution match)
# - robot perturb: XY 1.5mm, angle 1.5° (perfect_strict 1mm보다 살짝 wider, perfect_hold 2mm보다 tight)
# - retreat 0 (start at goal — 모든 frame이 hold 대상)
# - hold-record-steps 250 (perfect_strict 150 → 1.67x, perfect_hold 120 → 2.08x)
# - 12 workers × 150 ep = 1800 ep target

set -e
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

LOG_DIR=/data/public/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

echo "Launching submm-hold datagen (1800 ep, hold 250)..."
python run_parallel.py --script align_neargoal --workers 12 --episodes 150 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_submm_hold_v1 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 250 \
    --retreat-mm 0 \
    --perturb-xy-mm 1.5 --perturb-angle-deg 1.5 \
    --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
    --phantom-x-mm -10 10 --phantom-y-mm -25 25 \
    --phantom-z-mm 0 0 --phantom-angle-deg -5 5 \
    --seed 2033 > "$LOG_DIR/datagen_submm_hold_v1.log" 2>&1 &
PID=$!
echo "submm_hold_v1: PID $PID, log: $LOG_DIR/datagen_submm_hold_v1.log"
disown
