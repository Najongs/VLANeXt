#!/bin/bash
# Real-robot dataset collection (fine-alignment phase) via sim twin.
# Sim's scripted IK pre-aligns the needle near the trocar, applies a random
# perturbation, and then drives the real Mecademic Meca500 through the slow
# fine-alignment recovery; OAK cameras + real GetPose / GetJoints are recorded
# in the same HDF5 schema as Sim/Save_dataset_align_only.py.
#
# Usage:
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 [extra_flags...]
#
# Examples:
#   # Dry-run (no robot motion — recommended first):
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 \
#       --dry-run --num-episodes 1
#
#   # Live, joint-space mirror, 10 episodes:
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 --num-episodes 10
#
#   # Cartesian-delta mirror (alternative; compare against joint mode):
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 \
#       --num-episodes 5 --mirror-mode cartesian
#
#   # Reproducible perturbation sequence:
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 \
#       --num-episodes 20 --seed 42
#
# Notes:
#   - --phantom-pos is REQUIRED (real phantom XY in robot base frame, meters).
#     CALIBRATE for your physical setup.
#   - On the very first run, the real robot moves from its current pose (likely
#     HOME_JOINTS = (30,-20,20,0,30,60)) directly to the aligned pose. Verify
#     the joint-space path is collision-free for your phantom placement before
#     running live.
#   - Press 'q' on the OAK display window to abort the current episode.
#   - Ctrl-C also stops cleanly (deactivates + disconnects robot).

EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_collect_align \
    "${EXTRA_ARGS[@]}"
