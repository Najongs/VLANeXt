#!/bin/bash
# y<0 dedicated dataset (2026-05-20)
# 목적: Track A를 했지만 y=-25 region 여전히 fail (8.66mm, 2/9 SR)
# 원인: approach_00 cap 5000도 여전히 y>0 편향 → 새 데이터가 dilute됨
# 조치: phantom y ∈ [-29, -10] 전용 데이터 1500ep 추가, 후속 finetune mix
#
# 사양:
# - phantom y ∈ [-29, -10]  ← y<0 only (eval grid y=-25 + margin)
# - phantom x ±12, angle ±7°  ← Track A 동일
# - robot perturb XY 5mm, angle 5°  ← Track A 동일 (default)
# - hold_record_steps 60  ← Track A 동일
# - 10 workers × 150 ep = 1500 ep target

set -e
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

LOG_DIR=/home/najo/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

echo "Launching y<0 dedicated datagen (1500 ep)..."
python run_parallel.py --script align --workers 10 --episodes 150 \
    --base-dir /home/najo/NAS/VLANeXt/dataset/fine_align/NEARGOAL_yneg_v1 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 60 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 -10 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --seed 2030 > "$LOG_DIR/datagen_yneg_v1.log" 2>&1 &
PID=$!
echo "y<0 dedicated: PID $PID, log: $LOG_DIR/datagen_yneg_v1.log"
echo "Monitor: tail -f $LOG_DIR/datagen_yneg_v1.log"
disown
