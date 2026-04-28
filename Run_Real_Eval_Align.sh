#!/bin/bash
# Real-robot eval (fine-alignment phase) — applies inferred actions to a Mecademic Meca500.
#
# Usage:
#   bash Run_Real_Eval_Align.sh <checkpoint_path> [extra_flags...]
#
# Recommended workflow:
#   1. Position the phantom and manually pre-align the needle near the trocar
#      (e.g. via Robot_action.py teleop or the Mecademic web UI), then exit.
#   2. Run this script with --skip-home so it starts from your manually set pose.
#
# Examples:
#   # Dry-run smoke test (no motion):
#   bash Run_Real_Eval_Align.sh /data/public/NAS/VLANeXt/output_dir_align_2_far_mix_0428 --skip-home --dry-run --max-steps 20
#
#   # Live align run (manual pre-positioned start):
#   bash Run_Real_Eval_Align.sh /data/public/NAS/VLANeXt/output_dir_align_2_far_mix_0428 --skip-home --max-steps 100
#
#   # If your checkpoint was trained with use_sensor=True (proprio dim 8):
#   bash Run_Real_Eval_Align.sh /path/to/ckpt --skip-home --use-sensor --max-steps 100
#
#   # Swap camera roles if camera1/2 enumerated in unexpected order:
#   bash Run_Real_Eval_Align.sh /path/to/ckpt --skip-home --swap-cameras --max-steps 100
#
# Notes:
#   - Press 'q' on the OAK display window to abort mid-episode.
#   - 'Ctrl-C' also stops cleanly (deactivates + disconnects robot).

CHECKPOINT="${1:?provide checkpoint path as first argument}"
shift
EXTRA_ARGS=("$@")

CONFIG=config/sim_eval_align_config.yaml
TRAIN_CONFIG=config/sim_train_align_config.yaml

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_eval_align \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --train-config "${TRAIN_CONFIG}" \
    "${EXTRA_ARGS[@]}"
