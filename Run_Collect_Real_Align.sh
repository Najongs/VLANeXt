#!/bin/bash
# Real-robot dataset collection (fine-alignment phase) via sim twin.
# Sim's scripted IK pre-aligns the needle near the trocar, applies a random
# perturbation, and then drives the real Mecademic Meca500 through the slow
# fine-alignment recovery; OAK cameras + real GetPose / GetJoints are recorded
# in the same HDF5 schema as Sim/Save_dataset_align_only.py.
#
# Safety (sim dry-run validation, ON by default):
#   Every candidate perturbation is replayed in a sim copy *before* the real
#   robot moves. The check covers:
#     - Pre-align endpoint occlusion + HOME→aligned joint-linear path
#     - Per-episode: aligned→perturbed joint-linear path (real's actual
#       MoveJoints trajectory), perturbed pose, sim fine-align, sim hold
#   Failed candidates are resampled (up to --max-perturb-retries times); if all
#   attempts fail the episode is skipped and the next one samples fresh.
#
# Usage:
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 [extra_flags...]
#




#   # Live, joint-space mirror, 10 episodes:
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 0.0 --num-episodes 2 --stream-rate-hz 7.5
#




#   # Stricter safety (smaller occlusion tolerance → reject more aggressively):
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 \
#       --occlusion-tolerance-mm 0.5 --max-perturb-retries 10
#
#   # Disable safety (NOT recommended on real robot — needle can collide):
#   bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 --no-safety-validation
#
# Notes:
#   - --phantom-pos is REQUIRED (real phantom XY in robot base frame, meters).
#     CALIBRATE for your physical setup.
#   - On the very first run, the real robot moves from HOME_JOINTS=(30,-20,20,
#     0,30,60) to the aligned pose. The pre-flight safety check validates this
#     joint-linear path; a RuntimeError fail-stops before the robot moves if
#     phantom_pos places obstacles in the way.
#   - Per-episode: a perturbation is *only* sent to the real robot after sim
#     dry-run confirms the entire trajectory (perturb-move + fine-align + hold)
#     is occlusion/collision-free.
#   - Press 'q' on the OAK display window to abort the current episode.
#   - Ctrl-C also stops cleanly (deactivates + disconnects robot).

EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_collect_align \
    "${EXTRA_ARGS[@]}"
