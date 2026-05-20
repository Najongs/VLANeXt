#!/bin/bash
# 10K Datagen v3 (2026-05-20) — 다른 PC용 (NVIDIA EGL, GPU 가속 rendering)
#
# 지금까지 합의된 모든 인사이트 반영 (2026-05-20 사용자 수정 반영):
#   - Phantom range: eval grid (±10, ±25, ±5°) + 20% margin (±12, ±29, ±7°)
#   - Retreat: 0mm (전체 — trocar entry에 정확히 도달하는 trajectory 학습)
#   - Hold (approach + align 둘 다): 30 frames
#   - Track 2: 위치+각도 wide — XY±15mm, Z[-5,10]mm, angle 5° (default angle)
#   - Track 3: 작은 위치 + 큰 각도 — XY±5mm, Z[-5,5]mm, angle 20° (각도 회복 집중)
#     두 track 모두 랜덤 phantom pos 유지 → 다양한 phantom에서 회복 학습
#
# Distribution (총 10000 ep):
#   Track 1 — Approach (eval-matched, hold 30, retreat 0):       5000 ep (50%)
#   Track 2 — Align wide 15mm + angle 5° (hold 30, retreat 0):   3000 ep (30%)
#   Track 3 — Align small 5mm + angle 20° (hold 30, retreat 0):  2000 ep (20%)
#
# 예상 시간:
#   Approach (~14 sec/ep) → 500/10 × 14 ≈ 12분/worker  →  ~12min wall
#   Align pos+angle (~30 sec/ep, hold 60) → 300/10 × 30 ≈ 15분/worker → ~15min wall
#   Align angle-only (~10 sec/ep, hold 60) → 200/10 × 10 ≈ 3분/worker → ~3min wall
#   ※ 위는 30 worker 동시 launch 기준 — CPU 96 core 시스템 가정.
#     적으면 worker 수 또는 ALIGN_MULTIPLIER 조정.
#
# Champion v3 finetune mix에서 활용:
#   - Approach: 기존 approach_00 (~15K wide)에 5000ep eval-matched 추가
#   - Align pos+angle: NEARGOAL_eval_match 와 동일 사양 → 합산 가능
#   - Align angle-only: angle 교정 specialized signal

set -e
cd "$(dirname "$0")"

# ===== Rendering setup: NVIDIA EGL (다른 PC: GPU 정상 가정) =====
# Mesa 강제 안 함 — NVIDIA EGL vendor JSON 사용 (기본값)
# GPU rendering이 mesa software (54fps) 대비 훨씬 빠름 (200+fps).
export MUJOCO_GL=egl
unset __EGL_VENDOR_LIBRARY_FILENAMES  # mesa 강제 해제, NVIDIA default 사용

# Rendering 용 GPU 지정 (필요 시). 기본은 GPU 0.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=/data/public/NAS/VLANeXt/logs
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M)

echo "===================================================================="
echo "10K Datagen v3 (started $TS)"
echo "  Track 1: Approach 5000 ep                       → dataset/approach/approach_10k_v3"
echo "  Track 2: Align wide 15mm (angle 5°) 3000 ep     → dataset/fine_align/align_10k_v3"
echo "  Track 3: Align small 5mm + angle 20° 2000 ep    → dataset/fine_align/align_small_angle_10k_v3"
echo "  Total: 10000 ep"
echo "  Rendering: NVIDIA EGL (mesa unset)"
echo "===================================================================="

# ============================================================
# Track 1 — Approach (5000 ep, 10 worker × 500, hold 30)
#   학습 목표: "팬텀 어디 있든 그 위치로 끌고 와" (wide approach)
#   eval-matched phantom range, 기본 approach hold 30
# ============================================================
echo ""
echo "[1/3] Launching Approach (5000 ep)..."
python run_parallel.py --script approach --workers 10 --episodes 500 \
    --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_10k_v3 \
    --randomize-phantom-pos --no-side-camera --no-insertion \
    --hold-steps 30 --cameras tool_camera \
    --retreat-mm 0 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --seed 3027 > "$LOG_DIR/datagen_10k_approach_$TS.log" 2>&1 &
PID_AP=$!
echo "  Approach: PID $PID_AP, log: $LOG_DIR/datagen_10k_approach_$TS.log"

sleep 3

# ============================================================
# Track 2 — Align wide 15mm + angle 5° (3000 ep, 10 worker × 300)
#   학습 목표: "랜덤 phantom + 15mm 이내 위치/각도 회복" (wide recovery)
#   robot perturbation: XY ±15mm, Z [-5,10]mm (asymmetric), angle ±5° (default angle)
#   retreat 0mm, hold 30
# ============================================================
echo ""
echo "[2/3] Launching Align wide 15mm (3000 ep, hold 30, retreat 0)..."
python run_parallel.py --script align --workers 10 --episodes 300 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/align_10k_v3 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 30 \
    --retreat-mm 0 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --perturb-xy-mm 15 --perturb-z-min-mm -5 --perturb-z-max-mm 10 \
    --seed 3028 > "$LOG_DIR/datagen_10k_align_$TS.log" 2>&1 &
PID_AL=$!
echo "  Align: PID $PID_AL, log: $LOG_DIR/datagen_10k_align_$TS.log"

sleep 3

# ============================================================
# Track 3 — Align small 5mm + large angle 20° (2000 ep, 10 worker × 200)
#   학습 목표: "랜덤 phantom + 위치는 정렬 근처(±5mm) + 큰 각도(±20°) 회복"
#   robot perturbation: XY ±5mm, Z ±5mm, angle ±20° (큰 각도 specialized)
#   사용자 직관: 모델이 각도 잘 못잡음 → near-goal에서 angle 회복 강화
#   retreat 0mm, hold 30
# ============================================================
echo ""
echo "[3/3] Launching Align small 5mm + angle 20° (2000 ep, hold 30, retreat 0)..."
python run_parallel.py --script align --workers 10 --episodes 200 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/align_small_angle_10k_v3 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 30 \
    --retreat-mm 0 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -7 7 \
    --perturb-xy-mm 5 --perturb-z-min-mm -5 --perturb-z-max-mm 5 --perturb-angle-deg 20 \
    --seed 3029 > "$LOG_DIR/datagen_10k_small_angle_$TS.log" 2>&1 &
PID_AO=$!
echo "  Angle-only: PID $PID_AO, log: $LOG_DIR/datagen_10k_angle_only_$TS.log"

echo ""
echo "All 3 tracks launched. 30 workers total."
echo "Monitor: tail -f $LOG_DIR/datagen_10k_*_$TS.log"
echo ""
echo "Waiting for completion..."

wait $PID_AP
echo "[$(date +%H:%M:%S)] Track 1 (Approach) complete"

wait $PID_AL
echo "[$(date +%H:%M:%S)] Track 2 (Align pos+angle) complete"

wait $PID_AO
echo "[$(date +%H:%M:%S)] Track 3 (Align angle-only) complete"

echo ""
echo "===================================================================="
echo "10K Datagen v3 COMPLETE"
echo "  approach_10k_v3:            $(ls -1 /data/public/NAS/VLANeXt/dataset/approach/approach_10k_v3/worker_*/*.h5 2>/dev/null | wc -l) episodes"
echo "  align_10k_v3 (wide 10mm):   $(ls -1 /data/public/NAS/VLANeXt/dataset/fine_align/align_10k_v3/worker_*/*.h5 2>/dev/null | wc -l) episodes"
echo "  align_small_angle_10k_v3:   $(ls -1 /data/public/NAS/VLANeXt/dataset/fine_align/align_small_angle_10k_v3/worker_*/*.h5 2>/dev/null | wc -l) episodes"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "  1. Merge per-track: python run_parallel.py 출력 후 자동 merge됨 (collected_data_merged/)"
echo "  2. Champion v3 finetune config 추가:"
echo "     - approach_10k_v3/collected_data_merged"
echo "     - align_10k_v3/collected_data_merged"
echo "     - align_small_angle_10k_v3/collected_data_merged"
