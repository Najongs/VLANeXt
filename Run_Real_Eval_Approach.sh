#!/bin/bash
# Real-robot eval (approach phase) — applies inferred actions to a Mecademic Meca500.
#
# Usage:
#   bash Run_Real_Eval_Approach.sh <checkpoint_path> [extra_flags...]
#
# Examples:
#   # Dry-run (no robot motion — recommended first):
#   bash Run_Real_Eval_Approach.sh /home/najo/NAS/VLANeXt/output_dir_approach_2mix_0426 --dry-run --max-steps 10
#
#   # Live run:
#   bash Run_Real_Eval_Approach.sh /home/najo/NAS/VLANeXt/output_dir_approach_2mix_0426 --max-steps 50
#
#   # Swap camera roles if camera1/2 enumerated in unexpected order:
#   bash Run_Real_Eval_Approach.sh /path/to/ckpt --swap-cameras --max-steps 50
#

#   # 1단계 — dry-run (안전, 모션 X):
#   bash Run_Real_Eval_Approach.sh /path/to/ckpt --dry-run --max-steps 10

#   # 2단계 — 실제 동작:
#   bash Run_Real_Eval_Approach.sh /path/to/ckpt --max-steps 50

#   # 카메라 순서 swap 필요 시:
#   bash Run_Real_Eval_Approach.sh /path/to/ckpt --swap-cameras --max-steps 50

# Notes:
#   - Press 'q' on the OAK display window to abort mid-episode.
#   - 'Ctrl-C' also stops cleanly (deactivates + disconnects robot).

CHECKPOINT="${1:?provide checkpoint path as first argument}"
shift
EXTRA_ARGS=("$@")

CONFIG=config/sim_eval_approach_config.yaml
TRAIN_CONFIG=config/sim_train_approach_config.yaml

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_eval_approach \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --train-config "${TRAIN_CONFIG}" \
    "${EXTRA_ARGS[@]}"
