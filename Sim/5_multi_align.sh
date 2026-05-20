#!/bin/bash
# Data generation for champion v2 finetune (2026-05-19)
# 목적: eval grid (x∈±10, y∈±25, z=0, angle∈±5/±10°)에 매칭된 분포 학습
#       (champion aux_strong/10000 실패 4셀이 모두 y=−25 row → 분포 가장자리 보강)
#
# Phantom range : x±12mm, y±29mm, z=0, angle±12° (eval grid + 15% margin)
# Hold steps    : 30 (eval 20-step hold 통과 + 10 margin, hold-bottleneck 보완)
# Sampling      : continuous uniform random per episode (per-cell 분할 아님)

set -e
cd "$(dirname "$0")"

# === MuJoCo rendering: Mesa EGL 강제 ===
# GPU 0 (dead) 때문에 NVIDIA EGL 드라이버가 nvkms_open_common에서 hang.
# Mesa 벤더 JSON만 노출해서 NVIDIA EGL 건너뜀 (libEGL warnings 무해, 렌더 OK).
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}

# ============================================================
# Track 1 — NEW approach (phantom 변동, 로봇 home 고정)
#   ~5000 ep, 15 workers × 334 ep, ~5-7h (GPU 1개 only)
#   학습 목표: "팬텀 어디 있든 그 위치로 끌고 와"
#   approach 스크립트는 occlusion check 없음 → 자동 통과
# ============================================================
python run_parallel.py --script approach --workers 15 --episodes 334 \
    --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_eval_range_v1 \
    --randomize-phantom-pos --no-side-camera --no-insertion \
    --hold-steps 30 --cameras tool_camera \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -12 12

# ============================================================
# Track 2 — NEW align (phantom 변동 + 로봇 ±5mm perturb recovery)
#   ~500 ep, 15 workers × 34 ep, ~30-60min
#   학습 목표: "phantom 위치마다 fine-align ±5mm/±5° 복귀 + hold 30"
#   --allow-occluded: 가린 ep도 keep (IK reach만 통과하면 OK)
# ============================================================
python run_parallel.py --script align --workers 15 --episodes 34 \
    --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/align_phantom_range_v1 \
    --randomize-phantom-pos --no-side-camera --cameras tool_camera --allow-occluded \
    --hold-record-steps 30 \
    --phantom-x-mm -12 12 --phantom-y-mm -29 29 \
    --phantom-z-mm 0 0 --phantom-angle-deg -12 12

# ============================================================
# === Archive (do not re-run, kept for reference) ===
# ============================================================
# # old approach_00 (wide phantom range, hold 10 — champion baseline data)
# python run_parallel.py --script approach --workers 25 --episodes 200 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/approach/approach_00 \
#     --randomize-phantom-pos --no-side-camera --no-insertion \
#     --hold-steps 10 --cameras tool_camera
#
# # tip2 (phantom 고정 origin, robot perturb ±10mm — champion baseline data)
# python run_parallel.py --script align --workers 20 --episodes 50 \
#     --base-dir /data/public/NAS/VLANeXt/dataset/fine_align/10mm_fine_align_00_tip2 \
#     --phantom-pos 0.0 0.0 --no-side-camera --cameras tool_camera --allow-occluded
