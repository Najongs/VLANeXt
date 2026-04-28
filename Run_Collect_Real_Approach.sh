#!/bin/bash
# Real-robot dataset collection (approach phase) via sim twin.
# Sim's scripted IK trajectory drives the real Mecademic Meca500;
# OAK cameras + real GetPose / GetJoints get recorded into HDF5
# in the same schema as Sim/Save_dataset_approach_only.py.
#
# Usage:
#   bash Run_Collect_Real_Approach.sh --phantom-pos 0.0 -0.4 [extra_flags...]
#
# Examples:
#   # Dry-run (no robot motion — recommended first):
#   bash Run_Collect_Real_Approach.sh --phantom-pos 0.0 -0.4 \
#       --dry-run --num-episodes 1 --max-steps 30
#
#   # Live, joint-space mirror, 10 episodes:
#   bash Run_Collect_Real_Approach.sh --phantom-pos 0.0 -0.4 --num-episodes 10
#
#   # Cartesian-delta mirror (alternative; compare against joint mode):
#   bash Run_Collect_Real_Approach.sh --phantom-pos 0.0 -0.4 \
#       --num-episodes 5 --mirror-mode cartesian
#
#   # Camera order swap if camera1/2 came up flipped:
#   bash Run_Collect_Real_Approach.sh --phantom-pos 0.0 -0.4 --swap-cameras
#
# Notes:
#   - --phantom-pos is REQUIRED (real phantom XY in robot base frame, meters).
#     CALIBRATE for your physical setup; defaults below are placeholders.
#   - Press 'q' on the OAK display window to abort the current episode.
#   - Ctrl-C also stops cleanly (deactivates + disconnects robot).

EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_collect_approach \
    "${EXTRA_ARGS[@]}"
