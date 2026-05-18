#!/bin/bash
# Real-robot eval (fine-alignment phase) — applies inferred actions to a Mecademic Meca500.
#
# 2026-05-18 NOTE:
#   - real_eval_align.py currently runs **VLA-only** (no handoff / no KP head).
#   - sim의 close_5mm 66.7% 수치는 handoff + KP head 포함 결과. real에서는 미구현이라
#     VLA 단독 성능만 확인됨. handoff 통합은 별도 작업 (sensor_handoff.py가 sim env
#     snapshot/restore 의존 → RealRobotEnv에 대응 메서드 필요).
#
# Usage:
#   bash Run_Real_Eval_Align.sh <checkpoint_path> [--train-config <path>] [extra_flags...]
#
# Recommended checkpoint (2026-05-18 honest WIDE eval, sensor_stop bubble 제거):
#   ★ BEST = HARD_cotrain_lr_low / checkpoint_3000.pt
#      ckpt:  checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_HARD_cotrain_lr_low/checkpoint_3000.pt
#      train: config/sim_train_align_siglip2_b24_ft10mm_HARD_cotrain_lr_low_config.yaml
#      sim honest 5mm SR = 20/54 = **37.0%** (WIDE grid, sensor_stop OFF, VLA-only)
#
# 폐기 후보 (참고만):
#   - HOLD_focus_v2/3000      : 25.9% (seed2026), 19% (seed2027) — seed lottery 아래쪽
#   - NEW_finetune/10000      : 16% honest — sensor_stop 거품 빼면 약함
#   - unfreeze_last4/1000     : seed lottery (4 seeds mean 34% < frozen) — 재현 불가
#
# Recommended workflow:
#   1. Position the phantom and manually pre-align the needle near the trocar
#      (e.g. via Robot_action.py teleop or the Mecademic web UI), then exit.
#   2. Run this script with --skip-home so it starts from your manually set pose.
#   3. 처음 1~2 episode는 --dry-run으로 동작 확인 → 정상이면 본 run.
#
# Examples:
#   # Dry-run smoke test (no motion):
#   bash Run_Real_Eval_Align.sh \
#     /data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_HARD_cotrain_lr_low/checkpoint_3000.pt \
#     --train-config config/sim_train_align_siglip2_b24_ft10mm_HARD_cotrain_lr_low_config.yaml \
#     --skip-home --dry-run --max-steps 20
#
#   # Live align run (manual pre-positioned start):
#   bash Run_Real_Eval_Align.sh \
#     /data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_HARD_cotrain_lr_low/checkpoint_3000.pt \
#     --train-config config/sim_train_align_siglip2_b24_ft10mm_HARD_cotrain_lr_low_config.yaml \
#     --skip-home --max-steps 100
#
#   # Swap camera roles if camera1/2 enumerated in unexpected order:
#   bash Run_Real_Eval_Align.sh <ckpt> --train-config <cfg> --skip-home --swap-cameras --max-steps 100
#
# Notes:
#   - **반드시 --train-config로 학습 시 사용한 config 명시할 것** (default는 stale).
#     ckpt와 train-config가 mismatch면 architecture 미스매치로 weight 로드 실패 가능.
#   - Press 'q' on the OAK display window to abort mid-episode.
#   - 'Ctrl-C' also stops cleanly (deactivates + disconnects robot).
#   - --use-sensor: ckpt가 use_sensor=True (proprio_dim=9)로 학습됐을 때만 사용.
#     현재 추천 ckpt 둘 다 proprio_dim=6이므로 불필요.

CHECKPOINT="${1:?provide checkpoint path as first argument}"
shift
EXTRA_ARGS=("$@")

CONFIG=config/sim_eval_align_config.yaml

# Default train-config는 추천하지 않음 — ckpt와 짝이 맞는 config를
# --train-config flag로 명시적으로 override해야 함. fallback은 frozen baseline에 맞춤.
DEFAULT_TRAIN_CONFIG=config/sim_train_align_siglip2_b24_ft10mm_HARD_cotrain_lr_low_config.yaml

# EXTRA_ARGS에 --train-config가 있으면 그걸 사용, 없으면 default
HAS_TRAIN_CFG=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--train-config" ]]; then HAS_TRAIN_CFG=1; break; fi
done

if [[ $HAS_TRAIN_CFG -eq 0 ]]; then
    echo "[Run_Real_Eval_Align] WARNING: --train-config 미지정, default 사용: ${DEFAULT_TRAIN_CONFIG}"
    echo "[Run_Real_Eval_Align] ckpt와 짝이 안 맞으면 architecture mismatch 가능."
    EXTRA_ARGS+=(--train-config "${DEFAULT_TRAIN_CONFIG}")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_eval_align \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    "${EXTRA_ARGS[@]}"
