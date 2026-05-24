#!/bin/bash
# y=-25 tight band datagen (2026-05-23)
# v5 ck2000 (SR 63%) 천장의 ACT 격차 = y=-25 region 2/9 (ACT 9/9).
# 기존 yneg_v1은 y ∈ [-29, -10] wide → y=-25 cell만 강화하려면 tight band.
#
# 사양: phantom y ∈ [-29, -21] (y=-25 ±4mm 좁은 범위), 1500ep
set -e
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

LOG_DIR=/data/public/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

echo "Launching y=-25 strict datagen (1500 ep)..."
python run_parallel.py --script align --workers 10 --episodes 150 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_yneg25_strict_v1 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 60 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 -21 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --seed 2032 > "$LOG_DIR/datagen_yneg25_strict.log" 2>&1 &
PID=$!
echo "yneg25 strict: PID $PID, log: $LOG_DIR/datagen_yneg25_strict.log"
disown
