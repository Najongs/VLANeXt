#!/bin/bash
# Real-robot dataset collection (fine-alignment) — sim-plan + real-replay variant.
#
# Difference from Run_Collect_Real_Align.sh:
#   - Sim does NOT drive real tick-by-tick. Sim only computes aligned_qpos +
#     perturbed_qpos. Real robot is sent ONE MoveJoints to perturbed (blocking),
#     then ONE MoveJoints to aligned (non-blocking) — Mecademic plans its own
#     smooth trajectory. Recording loop is wall-clock paced.
#   - Sim is downgraded to FK-mirror: sim qpos = real qpos, mj_forward → derive
#     aux keys (needle_tip_pos, keypoints, sensor_dist) from real's actual pose.
#     This guarantees aux labels are perfectly synced with real images/state.
#   - No streaming loop, no EMA, no jitter from sim IK noise.
#
# Tradeoff: trajectory shape is Mecademic's joint-linear/S-curve, not sim's
# IK trajectory. End-pose identical, intermediate path differs slightly.
# HDF5 schema is identical → drop-in compatible with sim_act_align dataset.
#
# Usage:
#   bash Run_Collect_Real_Align_Replay.sh --phantom-pos 0.0 -0.4 [extra_flags...]

EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m digital_twin.real_collect_align_replay \
    "${EXTRA_ARGS[@]}"

rsync -av --progress -e 'ssh -p 37840' /home/irom/NAS/VLANeXt/dataset/real_align/collected_data_real yohan@nayohan.iptime.org:/data/public/NAS/VLANeXt/dataset