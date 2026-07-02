#!/bin/bash
# y>0 dedicated dataset (2026-05-22)
# 목적: Reach champion (b100v4_ft_phase2_lowlr/ck1500/exec=2) y=+25 region 55.6% (5/9)
#       — minLat champion은 100%/100%이고 reach champion은 y=-25/y=0 100% but y=+25만 weak
# 가설: 8_yneg_dedicated.sh가 y<0 천장 깬 것과 mirror — y>0 specialized data로 reach 천장 깰 수 있음
#
# 사양 (8_yneg_dedicated.sh와 정확히 mirror, y 부호만 반전):
# - phantom y ∈ [+10, +29]  ← y>0 only (eval grid y=+25 + margin)
# - phantom x ±12, angle ±7°
# - robot perturb XY 5mm, angle 5° (default)
# - hold_record_steps 60
# - 10 workers × 150 ep = 1500 ep target

set -e
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

LOG_DIR=/home/najo/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

echo "Launching y>0 dedicated datagen (1500 ep)..."
python run_parallel.py --script align --workers 10 --episodes 150 \
    --base-dir /home/najo/NAS/VLANeXt/dataset/fine_align/NEARGOAL_ypos_v1 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 60 \
    --phantom-x-mm -12 12 --phantom-y-mm 10 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --seed 2031 > "$LOG_DIR/datagen_ypos_v1.log" 2>&1 &
PID=$!
echo "y>0 dedicated: PID $PID, log: $LOG_DIR/datagen_ypos_v1.log"
echo "Monitor: tail -f $LOG_DIR/datagen_ypos_v1.log"
disown
