"""
real_collect_align.py

Real-robot dataset collection for the *fine-alignment* phase, driven by a
MuJoCo "sim twin". Mirrors `Sim/Save_dataset_align_only.py` but routes every
control frame to the real Mecademic Meca500 + records real OAK frames.

Episode flow (matches sim):
  1) (Once) Pre-alignment: sim runs IK from sim-home (0,0,0,0,0,0) toward the
     trocar entry; the resulting `aligned_qpos` is sent to the real robot
     with a blocking `MoveJoints` + `WaitIdle`. Real ends up at the aligned pose.
  2) Per episode (no recording until step ④):
     ② Reset sim to `aligned_qpos`.
     ③ Sample perturbation (xy ±40mm, z ±20mm, angle ±10°), drive sim there
        with a fast smooth_step IK move; mirror the *final* perturbed qpos to
        the real robot with `MoveJoints` + `WaitIdle`.
     ④ **Recording — Fine alignment**: sim does smooth_step IK from perturbed
        back to aligned at FINE_ALIGN_SPEED (~0.005 m/s). Real mirrors per
        control frame (joint or cartesian mirror), captures real OAK frames +
        real GetPose/GetJoints, writes to HDF5.
     ⑤ **Recording — Hold**: HOLD_RECORD_STEPS more ctrl frames at aligned.
     ⑥ HDF5 saved.

HDF5 schema is identical to `Save_dataset_align_only.py:SimRecorder` →
`src/datasets/sim_act_align.SimActAlign` loads it as-is for sim+real merged
training.

Reuses `RealCollectEnv`, `SimHandles`, IK solver, etc. from
`digital_twin.real_collect_approach`.

  실 로봇 첫 실행 권장 절차:                     
                                                                                                                            
  # 1) Sim+OAK만 (real 미연결 / dry-run)                                                                                                                                   
  bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 \                                                                                                                  
      --dry-run --num-episodes 1                                                                                                                                           
                                                                                                                                                                           
  # 2) 실 로봇 라이브 1 에피소드 — 첫 큰 이동(HOME→aligned) collision-free 시각 확인                                                                                       
  bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 --num-episodes 1                                                                                                   
                                                                                                                                                                           
  # 3) 본격 수집                                                                                                                                                           
  bash Run_Collect_Real_Align.sh --phantom-pos 0.0 -0.4 --num-episodes 50 \                                                                                                
      --save-dir /data/public/NAS/VLANeXt/dataset/real_align/run_0429 --seed 42 
      
Usage:
    bash Run_Collect_Real_Align.sh --num-episodes 10 --phantom-pos 0.0 -0.4
    bash Run_Collect_Real_Align.sh --dry-run --num-episodes 1
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import argparse
import logging
import pathlib

import numpy as np
import cv2
import mujoco

# Path setup: project root + Sim/
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Sim"))

# Reuse from sim collector (no modification — single source of truth)
from Save_dataset_align_only import (  # noqa: E402
    SimRecorder,
    smooth_step,
    project_to_2d,
    ALIGN_SPEED,
    FINE_ALIGN_SPEED,
    PERTURB_POS_XY_MM,
    PERTURB_POS_Z_MIN_MM,
    PERTURB_POS_Z_MAX_MM,
    PERTURB_ANGLE_DEG,
    ALIGN_THRESHOLD_M,
    ALIGN_HOLD_STEPS,
    HOLD_RECORD_STEPS,
    ACTION_CLIP_MM,
    TIMEOUT_SEC,
    MAX_CTRL_STEPS,
    RETREAT_MM,
    IMG_WIDTH,
    IMG_HEIGHT,
    TASK_INSTRUCTION,
)

# Reuse the real-robot wrapper + sim helpers we already wrote for approach
from digital_twin.real_collect_approach import (  # noqa: E402
    RealCollectEnv,
    SimHandles,
    _solve_ik_step,
    _set_phantom,
    _wrap_pi,
    _build_display_frame,
)
from digital_twin.real_eval_approach import ROBOT_ADDRESS_DEFAULT, HOME_JOINTS  # noqa: E402


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Safety: sim-only dry-run validation (occlusion + collision proxy)
#
# Mirrors Save_dataset_align_only.py:310-322 (`check_tip_occluded`). The check
# ray-casts from `tool_camera` toward `needle_tip`; anything hit closer than
# the tip indicates the tip is hidden behind a phantom geom (or, by proxy,
# penetrating it). We use this as a *collision proxy* — a tip that ends up
# inside/behind the phantom would crash the real robot.
#
# Strategy: before sending any perturbed pose to the real robot, we replay the
# *entire planned trajectory* (perturb-move + fine-align + hold) in a copy of
# MjData and abort if any frame triggers the check. If validation fails, we
# resample a new perturbation (up to `cfg.max_perturb_retries` times). Episodes
# that pass validation are then executed live with sim+real mirror + recording.
# ─────────────────────────────────────────────────────────────────────────────
def _check_tip_occluded(model, data, h: SimHandles, tolerance_m: float = 0.001) -> bool:
    """`tool_camera → needle_tip` ray-cast. True if anything between the camera
    and the tip (within `tolerance_m`) — i.e., the tip is hidden behind a phantom
    geom (proxy for collision). Mirrors Save_dataset_align_only.py:310-322."""
    tool_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tool_camera")
    if tool_cam_id < 0:
        return False
    cam_pos = data.cam_xpos[tool_cam_id].copy()
    tip_pos = data.site_xpos[h.tip_id].copy()
    direction = tip_pos - cam_pos
    d_to_tip = np.linalg.norm(direction)
    if d_to_tip < 1e-6:
        return False
    dn = direction / d_to_tip
    geomid_out = np.zeros(1, dtype=np.int32)
    hit_dist = mujoco.mj_ray(model, data, cam_pos, dn, None, 1, -1, geomid_out)
    return bool(0 < hit_dist < d_to_tip - tolerance_m)


def _clone_state(model, src_data):
    """Fresh MjData with qpos/qvel/time copied from src. Phantom config lives
    on `model.body_pos` (shared) so it carries over automatically."""
    new_data = mujoco.MjData(model)
    new_data.qpos[:] = src_data.qpos.copy()
    new_data.qvel[:] = src_data.qvel.copy()
    new_data.time = float(src_data.time)
    mujoco.mj_forward(model, new_data)
    return new_data


def _check_qpos_path_safe(model, data, h: SimHandles, qpos_start, qpos_end,
                           n_steps: int = 20, tolerance_m: float = 0.001):
    """Validate Mecademic-style joint-linear interp `qpos_start → qpos_end`.

    `MoveJoints` interpolates linearly in joint space (not the smooth IK
    trajectory the sim took to *find* `qpos_end`). So we sample N intermediate
    qpos, set them via `mj_forward`, and run the occlusion check at each.
    Mutates `data.qpos`; caller is responsible for restoring state if needed.

    Returns (ok: bool, reason: str).
    """
    qpos_start = np.asarray(qpos_start, dtype=np.float64)
    qpos_end = np.asarray(qpos_end, dtype=np.float64)
    for i in range(n_steps + 1):
        alpha = i / n_steps
        data.qpos[: h.n_motors] = (1 - alpha) * qpos_start + alpha * qpos_end
        mujoco.mj_forward(model, data)
        if _check_tip_occluded(model, data, h, tolerance_m):
            return False, f"qpos_path_step_{i}/{n_steps}"
    return True, "ok"


def _dryrun_fine_align(model, data, h: SimHandles, goal_tip, goal_back,
                       max_ctrl_steps: int, tolerance_m: float = 0.001):
    """Sim-only replica of the live fine-align loop. Same IK + mj_step cadence
    (one occlusion check per 67 sim steps == one recorded frame). No real, no
    recording. Returns (ok, reason)."""
    start_tip_pos = data.site_xpos[h.tip_id].copy()
    start_back_pos = data.site_xpos[h.back_id].copy()
    fine_align_distance = np.linalg.norm(goal_tip - start_tip_pos)
    fine_duration = max(fine_align_distance / FINE_ALIGN_SPEED, 1e-3)
    fine_traj_start = data.time
    record_start_time = data.time
    align_timer = 0
    step_count = 0
    ctrl_step = 0

    while True:
        progress = smooth_step((data.time - fine_traj_start) / fine_duration)
        target_tip = (1 - progress) * start_tip_pos + progress * goal_tip
        target_back = (1 - progress) * start_back_pos + progress * goal_back
        _solve_ik_step(model, data, h, target_tip, target_back, 0.5)
        mujoco.mj_step(model, data)
        step_count += 1

        if step_count % 67 == 0:
            ctrl_step += 1
            if _check_tip_occluded(model, data, h, tolerance_m):
                return False, f"finealign_ctrl{ctrl_step}"
            if ctrl_step >= max_ctrl_steps:
                return False, f"finealign_max_steps_{max_ctrl_steps}"

        if progress >= 1.0:
            curr_tip = data.site_xpos[h.tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                return True, "ok"

        if data.time - record_start_time > TIMEOUT_SEC:
            return False, "finealign_timeout"


def _dryrun_hold(model, data, h: SimHandles, goal_tip, goal_back,
                 tolerance_m: float = 0.001):
    """Sim-only replica of the hold loop (HOLD_RECORD_STEPS × 67 sim steps).
    Returns (ok, reason)."""
    for hold_step in range(HOLD_RECORD_STEPS):
        for _ in range(67):
            _solve_ik_step(model, data, h, goal_tip, goal_back, 0.5)
            mujoco.mj_step(model, data)
        if _check_tip_occluded(model, data, h, tolerance_m):
            return False, f"hold_step_{hold_step}"
    return True, "ok"


def _dryrun_validate_episode(model, live_data, h: SimHandles,
                              goal_tip, goal_back,
                              perturb_xyz, perturb_angle_rad, perturb_axis,
                              max_ctrl_steps: int,
                              tolerance_m: float = 0.001,
                              qpos_path_steps: int = 20):
    """Full sim-only dry-run of one candidate episode. `live_data` is NOT
    mutated — all work happens on a clone. Returns (ok, reason, perturbed_qpos).

    Sequence (mirrors what the real robot will actually do):
      (a) `_move_to_perturbed_sim` on a clone — gives us `perturbed_qpos` and
          validates the smooth IK trajectory implicitly via per-step ncon
          (we only check occlusion at the endpoint here for speed).
      (b) `_check_qpos_path_safe(aligned → perturbed)` — checks the
          *joint-linear* path the real robot will take via `MoveJoints`.
          This is the safety-critical step: smooth IK and joint-linear are
          different trajectories in cartesian space.
      (c) `_dryrun_fine_align` from perturbed → aligned (slow streaming).
      (d) `_dryrun_hold` at aligned.
    """
    aligned_qpos = live_data.qpos[: h.n_motors].copy()

    # (a) Run perturb move on clone-A to derive perturbed_qpos
    sim_a = _clone_state(model, live_data)
    reached = _move_to_perturbed_sim(
        model, sim_a, h, goal_tip, goal_back,
        perturb_xyz, perturb_angle_rad, perturb_axis,
    )
    if not reached:
        return False, "perturb_ik_fail", None
    perturbed_qpos = sim_a.qpos[: h.n_motors].copy()
    if _check_tip_occluded(model, sim_a, h, tolerance_m):
        return False, "occluded_at_perturbed", None

    # (b) Joint-linear path check on clone-B (real's actual trajectory)
    sim_b = _clone_state(model, live_data)
    ok, reason = _check_qpos_path_safe(
        model, sim_b, h, aligned_qpos, perturbed_qpos,
        n_steps=qpos_path_steps, tolerance_m=tolerance_m,
    )
    if not ok:
        return False, f"path_aligned→perturbed_{reason}", None

    # (c) Fine-align dry-run continues from sim_a (already at perturbed)
    ok, reason = _dryrun_fine_align(
        model, sim_a, h, goal_tip, goal_back,
        max_ctrl_steps=max_ctrl_steps, tolerance_m=tolerance_m,
    )
    if not ok:
        return False, reason, None

    # (d) Hold dry-run
    ok, reason = _dryrun_hold(model, sim_a, h, goal_tip, goal_back, tolerance_m)
    if not ok:
        return False, reason, None

    return True, "ok", perturbed_qpos


# ─────────────────────────────────────────────────────────────────────────────
# Sim-side: pre-alignment, perturbation move (no recording)
# ─────────────────────────────────────────────────────────────────────────────
def _pre_align_sim(model, data, h: SimHandles, ik_speed=0.5) -> tuple:
    """One-time sim pre-alignment from sim-home (0,0,0,0,0,0) to trocar entry.

    Returns (aligned_qpos, aligned_qvel, p_entry_world, p_depth_world, needle_len_m).
    Assumes phantom has already been placed in `data` by the caller.
    """
    # Sim-home (matches Save_dataset_align_only.py:327, 394)
    data.qpos[:6] = np.zeros(6)
    mujoco.mj_forward(model, data)

    p_entry = data.site_xpos[h.target_entry_id].copy()
    p_depth = data.site_xpos[h.target_depth_id].copy()
    curr_tip = data.site_xpos[h.tip_id].copy()
    curr_back = data.site_xpos[h.back_id].copy()
    needle_len = np.linalg.norm(curr_tip - curr_back)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    retreat_m = RETREAT_MM / 1000.0
    goal_tip = p_entry - (axis_dir * retreat_m)
    goal_back = p_entry - (axis_dir * (retreat_m + needle_len))

    start_tip_pos = curr_tip.copy()
    start_back_pos = curr_back.copy()
    align_distance = np.linalg.norm(goal_tip - start_tip_pos)
    dynamic_duration = max(align_distance / ALIGN_SPEED, 1e-3)
    traj_start_time = data.time
    align_timer = 0

    while True:
        progress = smooth_step((data.time - traj_start_time) / dynamic_duration)
        target_tip = (1 - progress) * start_tip_pos + progress * goal_tip
        target_back = (1 - progress) * start_back_pos + progress * goal_back

        _solve_ik_step(model, data, h, target_tip, target_back, ik_speed)
        mujoco.mj_step(model, data)

        if progress >= 1.0:
            curr_tip = data.site_xpos[h.tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                break

        if data.time - traj_start_time > 50.0:
            raise RuntimeError("sim pre-alignment failed (>50s sim wall-time)")

    return (
        data.qpos[: h.n_motors].copy(),
        data.qvel[: h.n_motors].copy(),
        p_entry, p_depth, needle_len,
    )


def _solve_retreat_qpos(model, data, h: SimHandles, p_entry, p_depth, needle_len,
                         extra_retreat_mm: float, ik_speed: float = 0.5):
    """Solve qpos with needle tip retreated by (RETREAT_MM + extra_retreat_mm)
    from trocar entry along the trocar axis. Used as inter-episode safe
    waypoint — pulls the needle further from the phantom before traveling to
    HOME, so consecutive-episode joint-space transitions can't graze the
    phantom. Mirrors `_pre_align_sim`'s smooth-IK loop.

    `data` must currently be at the aligned pose. Mutates `data` (caller
    should pass a clone if needed)."""
    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    total_retreat_m = (RETREAT_MM + float(extra_retreat_mm)) / 1000.0
    goal_tip = p_entry - axis_dir * total_retreat_m
    goal_back = p_entry - axis_dir * (total_retreat_m + needle_len)

    start_tip = data.site_xpos[h.tip_id].copy()
    start_back = data.site_xpos[h.back_id].copy()
    dist = float(np.linalg.norm(goal_tip - start_tip))
    duration = max(dist / ALIGN_SPEED, 1e-3)
    t0 = data.time
    align_timer = 0
    while True:
        progress = smooth_step((data.time - t0) / duration)
        target_tip = (1 - progress) * start_tip + progress * goal_tip
        target_back = (1 - progress) * start_back + progress * goal_back
        _solve_ik_step(model, data, h, target_tip, target_back, ik_speed)
        mujoco.mj_step(model, data)
        if progress >= 1.0:
            curr_tip = data.site_xpos[h.tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                break
        if data.time - t0 > 30.0:
            raise RuntimeError("retreat IK timeout")
    return data.qpos[: h.n_motors].copy()


def _sample_perturbation(rng: np.random.Generator) -> tuple:
    """xy ±PERTURB_POS_XY_MM, z in [PERTURB_POS_Z_MIN_MM, PERTURB_POS_Z_MAX_MM],
    angle ±PERTURB_ANGLE_DEG, axis isotropic on S². Returns (perturb_xyz_m, angle_rad, axis)."""
    perturb_xyz = np.array([
        rng.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
        rng.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
        rng.uniform(PERTURB_POS_Z_MIN_MM, PERTURB_POS_Z_MAX_MM) / 1000.0,
    ], dtype=np.float64)
    angle_rad = np.deg2rad(rng.uniform(-PERTURB_ANGLE_DEG, PERTURB_ANGLE_DEG))
    axis = rng.standard_normal(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    return perturb_xyz, angle_rad, axis


def _move_to_perturbed_sim(model, data, h: SimHandles, goal_tip, goal_back,
                            perturb_xyz, angle_rad, axis, ik_speed=0.5,
                            move_speed_mps=0.05) -> bool:
    """Sim-only smooth IK move from current pose to perturbed pose. No recording.

    Returns True if perturbed pose reached within 1mm tip tolerance.
    """
    perturbed_tip = goal_tip + perturb_xyz
    rot_mat = np.eye(3)
    if abs(angle_rad) > 1e-6:
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        rot_mat = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    perturbed_back_dir = rot_mat @ (goal_back - goal_tip)
    perturbed_back = perturbed_tip + perturbed_back_dir

    move_dist = np.linalg.norm(perturbed_tip - goal_tip)
    move_duration = max(move_dist / move_speed_mps, 0.1)
    move_start_time = data.time

    for ps in range(5000):
        t = (data.time - move_start_time) / move_duration
        alpha = smooth_step(min(t, 1.0))
        interp_tip = (1 - alpha) * goal_tip + alpha * perturbed_tip
        interp_back = (1 - alpha) * goal_back + alpha * perturbed_back
        _solve_ik_step(model, data, h, interp_tip, interp_back, ik_speed)
        mujoco.mj_step(model, data)

        if t >= 1.0:
            if np.linalg.norm(data.site_xpos[h.tip_id] - perturbed_tip) < 0.001:
                # Settling
                for _ in range(200):
                    _solve_ik_step(model, data, h, perturbed_tip, perturbed_back, ik_speed)
                    mujoco.mj_step(model, data)
                return True
            if ps > 4500:
                return False
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main per-episode loop
# ─────────────────────────────────────────────────────────────────────────────
def run_collection_episode(model, data, env: RealCollectEnv, recorder: SimRecorder,
                           cfg, h: SimHandles, ep_idx: int, rng: np.random.Generator,
                           aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len) -> bool:
    """One sim-driven, real-recorded fine-alignment episode."""

    # ② Reset sim to aligned state
    mujoco.mj_resetData(model, data)
    data.qpos[: h.n_motors] = aligned_qpos
    data.qvel[: h.n_motors] = aligned_qvel
    # Re-set phantom in case mj_resetData cleared body_pos overrides
    _set_phantom(model, data, h, cfg.phantom_pos, cfg.phantom_rot)
    mujoco.mj_forward(model, data)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    retreat_m = RETREAT_MM / 1000.0
    goal_tip = p_entry - (axis_dir * retreat_m)
    goal_back = p_entry - (axis_dir * (retreat_m + needle_len))

    # ③ Perturbation: sim-only validation loop, then mirror final pose to real.
    #
    # Each candidate perturbation is dry-run in a sim copy (perturb-move +
    # joint-linear path + fine-align + hold). Only validated perturbations are
    # ever sent to the real robot. If all `cfg.max_perturb_retries` candidates
    # fail validation, this episode is skipped (the next call samples fresh).
    perturb_xyz = perturb_angle_rad = perturb_axis = None
    perturbed_qpos_validated = None  # rad, only set when validation enabled

    if cfg.skip_safety_validation:
        # Legacy behavior: single sample, no pre-flight check.
        perturb_xyz, perturb_angle_rad, perturb_axis = _sample_perturbation(rng)
        perturb_pos_mm = perturb_xyz * 1000.0
        logger.info(f"  Episode {ep_idx}: perturb_xyz_mm={perturb_pos_mm.round(1).tolist()}  "
                    f"angle_deg={np.rad2deg(perturb_angle_rad):.1f}  [SAFETY OFF]")
    else:
        n_attempts = max(1, int(cfg.max_perturb_retries))
        last_reason = "no_attempt"
        for attempt in range(1, n_attempts + 1):
            xyz, ang, ax = _sample_perturbation(rng)
            ok, reason, perturbed_qpos_validated = _dryrun_validate_episode(
                model, data, h, goal_tip, goal_back, xyz, ang, ax,
                max_ctrl_steps=cfg.max_steps,
                tolerance_m=cfg.occlusion_tol_m,
                qpos_path_steps=cfg.qpos_path_steps,
            )
            if ok:
                perturb_xyz, perturb_angle_rad, perturb_axis = xyz, ang, ax
                logger.info(
                    f"  ✓ Episode {ep_idx} validation pass on attempt {attempt}/{n_attempts}: "
                    f"perturb_xyz_mm={(xyz * 1000).round(1).tolist()} "
                    f"angle_deg={np.rad2deg(ang):.1f}"
                )
                break
            last_reason = reason
            logger.info(
                f"  ✗ Episode {ep_idx} attempt {attempt}/{n_attempts} rejected ({reason}): "
                f"perturb_xyz_mm={(xyz * 1000).round(1).tolist()} "
                f"angle_deg={np.rad2deg(ang):.1f}"
            )
        if perturb_xyz is None:
            logger.warning(
                f"  ⏭️  Episode {ep_idx} skipped: {n_attempts} validation attempts "
                f"all failed (last={last_reason}). Trying next episode with new randoms."
            )
            return False
        perturb_pos_mm = perturb_xyz * 1000.0

    # Re-run perturb on live data (deterministic; produces same perturbed_qpos
    # as the validation clone). This is what the sim trajectory recorder needs.
    reached = _move_to_perturbed_sim(
        model, data, h, goal_tip, goal_back,
        perturb_xyz, perturb_angle_rad, perturb_axis,
    )
    if not reached:
        logger.warning(f"  Episode {ep_idx} discarded: sim IK failed to reach perturbed pose (live)")
        return False

    perturbed_qpos_deg = np.rad2deg(data.qpos[:6].copy())
    # Sanity: live-derived pose should match the validated one
    if perturbed_qpos_validated is not None:
        delta_deg = float(np.max(np.abs(perturbed_qpos_deg - np.rad2deg(perturbed_qpos_validated))))
        if delta_deg > 0.5:
            logger.warning(
                f"  ⚠️  Episode {ep_idx}: live perturbed_qpos differs from validated by {delta_deg:.2f}° "
                f"(non-determinism?). Aborting episode for safety."
            )
            return False

    env.reset_to_joints(perturbed_qpos_deg)

    # Episode metadata (matches sim's episode_meta keys)
    episode_meta = {
        "aligned_qpos": np.rad2deg(aligned_qpos).astype(np.float32),
        "perturb_xyz_mm": perturb_pos_mm.astype(np.float32),
        "perturb_angle_deg": np.array(np.rad2deg(perturb_angle_rad), dtype=np.float32),
        "target_entry_world": p_entry.astype(np.float32),
        "target_depth_world": p_depth.astype(np.float32),
        "retreat_mm": np.float32(RETREAT_MM),
        # Real-only: trace the fixed phantom config used for this run
        "phantom_offset": np.array([cfg.phantom_pos[0], cfg.phantom_pos[1], 0.0], dtype=np.float32),
    }
    recorder.start(episode_meta)

    # ④ Recording — Fine alignment (perturbed → aligned, slow)
    last_real_state = env.read_state()
    last_real_ee = last_real_state["ee_pose"]

    start_tip_pos = data.site_xpos[h.tip_id].copy()
    start_back_pos = data.site_xpos[h.back_id].copy()
    fine_align_distance = np.linalg.norm(goal_tip - start_tip_pos)
    fine_duration = max(fine_align_distance / FINE_ALIGN_SPEED, 1e-3)
    fine_traj_start = data.time
    record_start_time = data.time

    last_ee_pose_sim = h.ee_pose_mm_rad(data)
    align_timer = 0
    success = False
    user_quit = False
    step_count = 0
    ctrl_step = 0

    while True:
        progress = smooth_step((data.time - fine_traj_start) / fine_duration)
        target_tip = (1 - progress) * start_tip_pos + progress * goal_tip
        target_back = (1 - progress) * start_back_pos + progress * goal_back

        _solve_ik_step(model, data, h, target_tip, target_back, 0.5)

        # sensor_dist via sim ray
        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()
        nlen = np.linalg.norm(curr_tip - curr_back) + 1e-10
        needle_dir = (curr_tip - curr_back) / nlen
        dist_to_surface = mujoco.mj_ray(
            model, data, curr_tip, needle_dir, None, 1, h.link6_id, np.zeros(1, dtype=np.int32)
        )
        current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

        mujoco.mj_step(model, data)
        step_count += 1

        # High-rate joint streaming for smooth Mecademic blending (joint mode only).
        # Recording rate (7.46 Hz) stays unchanged below; only the joint command rate
        # is decoupled and raised so consecutive MoveJoints chain without stop-start.
        if cfg.mirror_mode == "joint" and step_count % cfg.stream_every == 0:
            env.stream_joints(np.rad2deg(data.qpos[:6].copy()))
            time.sleep(cfg.stream_dt)

        if step_count % 67 == 0:
            ctrl_step += 1

            sim_ee_pose_mm = h.ee_pose_mm_rad(data)
            sim_delta_ee = sim_ee_pose_mm - last_ee_pose_sim
            sim_delta_ee[3:6] = _wrap_pi(sim_delta_ee[3:6])
            pos_mag = np.linalg.norm(sim_delta_ee[:3])
            if pos_mag > ACTION_CLIP_MM:
                sim_delta_ee[:3] *= ACTION_CLIP_MM / pos_mag

            # Cartesian mode drives real here (joint mode already streamed above).
            if cfg.mirror_mode == "cartesian":
                env.stream_cartesian_delta(sim_delta_ee)
                time.sleep(cfg.control_dt)

            real_frames = env.render_frames()
            real_state = env.read_state()
            real_qpos_deg = real_state["qpos_deg"]
            real_ee = real_state["ee_pose"]

            real_delta_ee = real_ee - last_real_ee
            real_delta_ee[3:6] = _wrap_pi(real_delta_ee[3:6])
            real_clip_mm = ACTION_CLIP_MM * 5.0
            real_pos_mag = np.linalg.norm(real_delta_ee[:3])
            if real_pos_mag > real_clip_mm:
                real_delta_ee[:3] *= real_clip_mm / real_pos_mag

            needle_tip_mm = data.site_xpos[h.tip_id].copy() * 1000.0
            trocar_entry_mm = data.site_xpos[h.target_entry_id].copy() * 1000.0
            tip_uv_wrist = project_to_2d(
                data.site_xpos[h.tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT
            )
            trocar_uv_wrist = project_to_2d(
                data.site_xpos[h.target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT
            )
            keypoints_wrist = np.concatenate([tip_uv_wrist, trocar_uv_wrist]).astype(np.float32)
            tip_visible = float(0.0 <= tip_uv_wrist[0] <= 1.0 and 0.0 <= tip_uv_wrist[1] <= 1.0)
            trocar_visible = float(0.0 <= trocar_uv_wrist[0] <= 1.0 and 0.0 <= trocar_uv_wrist[1] <= 1.0)
            keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

            frames_for_recorder = {
                "top_camera": real_frames["top_camera"],
                "tool_camera": real_frames["tool_camera"],
                "side_camera": real_frames["side_camera"],
            }
            if cfg.skip_side_camera:
                frames_for_recorder.pop("side_camera", None)

            recorder.add(
                frames_for_recorder,
                real_qpos_deg,
                real_ee.astype(np.float32),
                real_delta_ee.astype(np.float32),
                float(data.time),
                1,  # phase=1 throughout (align + hold)
                float(current_sensor_dist),
                needle_tip_mm=needle_tip_mm.astype(np.float32),
                trocar_entry_mm=trocar_entry_mm.astype(np.float32),
                keypoints_wrist=keypoints_wrist,
                keypoints_visibility=keypoints_visibility,
                instruction=TASK_INSTRUCTION,
            )
            last_ee_pose_sim = sim_ee_pose_mm.copy()
            last_real_ee = real_ee.copy()

            # Display + 'q' abort
            if cfg.show_preview:
                try:
                    disp = _build_display_frame(
                        real_frames, ctrl_step, cfg.max_steps, 1,
                        sim_delta_ee[:3], np.rad2deg(sim_delta_ee[3:6]),
                        cfg.mirror_mode, env.dry_run,
                    )
                    cv2.imshow("Real Collect Align (top | tool) — 'q' abort", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        logger.warning("🛑 'q' pressed — aborting episode")
                        user_quit = True
                        break
                except Exception as e:
                    logger.debug(f"Preview disabled: {e}")
                    cfg.show_preview = False

            if ctrl_step % 10 == 0:
                logger.info(
                    f"    align ctrl {ctrl_step:3d}  "
                    f"sim_dpos_mm={sim_delta_ee[:3].round(2).tolist()}  "
                    f"real_dpos_mm={real_delta_ee[:3].round(2).tolist()}"
                )

        # Success: reached aligned pose (after smooth_step done)
        if progress >= 1.0:
            curr_tip = data.site_xpos[h.tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                success = True
                break

        if data.time - record_start_time > TIMEOUT_SEC:
            logger.warning(f"  Episode {ep_idx}: align timeout (>{TIMEOUT_SEC}s sim)")
            break
        if len(recorder.buffer) >= cfg.max_steps:
            logger.warning(f"  Episode {ep_idx}: hit max_steps={cfg.max_steps}")
            break

    if user_quit or not success:
        recorder.discard()
        reason = "user_quit" if user_quit else "no_success"
        logger.warning(f"  ❌ Episode {ep_idx} discarded ({reason})")
        return False

    # ⑤ Recording — Hold
    for hold_step in range(HOLD_RECORD_STEPS):
        # Run 67 sim steps holding the aligned target. In joint mode, stream qpos
        # at high rate inside this inner loop so the robot stays continuously fed
        # (queue active) instead of going idle between recording boundaries.
        for inner_i in range(1, 68):
            _solve_ik_step(model, data, h, goal_tip, goal_back, 0.5)
            mujoco.mj_step(model, data)
            if cfg.mirror_mode == "joint" and inner_i % cfg.stream_every == 0:
                env.stream_joints(np.rad2deg(data.qpos[:6].copy()))
                time.sleep(cfg.stream_dt)

        # Sensor + sim state
        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()
        nlen = np.linalg.norm(curr_tip - curr_back) + 1e-10
        needle_dir = (curr_tip - curr_back) / nlen
        dist_to_surface = mujoco.mj_ray(
            model, data, curr_tip, needle_dir, None, 1, h.link6_id, np.zeros(1, dtype=np.int32)
        )
        current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

        sim_ee_pose_mm = h.ee_pose_mm_rad(data)
        sim_delta_ee = sim_ee_pose_mm - last_ee_pose_sim
        sim_delta_ee[3:6] = _wrap_pi(sim_delta_ee[3:6])
        pos_mag = np.linalg.norm(sim_delta_ee[:3])
        if pos_mag > ACTION_CLIP_MM:
            sim_delta_ee[:3] *= ACTION_CLIP_MM / pos_mag

        # Cartesian mode drives real here (joint mode streamed inside the inner
        # 67-step loop above). Sim is essentially still at aligned, so deltas tiny.
        if cfg.mirror_mode == "cartesian":
            env.stream_cartesian_delta(sim_delta_ee)
            time.sleep(cfg.control_dt)

        real_frames = env.render_frames()
        real_state = env.read_state()
        real_qpos_deg = real_state["qpos_deg"]
        real_ee = real_state["ee_pose"]
        real_delta_ee = real_ee - last_real_ee
        real_delta_ee[3:6] = _wrap_pi(real_delta_ee[3:6])
        real_clip_mm = ACTION_CLIP_MM * 5.0
        real_pos_mag = np.linalg.norm(real_delta_ee[:3])
        if real_pos_mag > real_clip_mm:
            real_delta_ee[:3] *= real_clip_mm / real_pos_mag

        needle_tip_mm = data.site_xpos[h.tip_id].copy() * 1000.0
        trocar_entry_mm = data.site_xpos[h.target_entry_id].copy() * 1000.0
        tip_uv_wrist = project_to_2d(
            data.site_xpos[h.tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT
        )
        trocar_uv_wrist = project_to_2d(
            data.site_xpos[h.target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT
        )
        keypoints_wrist = np.concatenate([tip_uv_wrist, trocar_uv_wrist]).astype(np.float32)
        tip_visible = float(0.0 <= tip_uv_wrist[0] <= 1.0 and 0.0 <= tip_uv_wrist[1] <= 1.0)
        trocar_visible = float(0.0 <= trocar_uv_wrist[0] <= 1.0 and 0.0 <= trocar_uv_wrist[1] <= 1.0)
        keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

        frames_for_recorder = {
            "top_camera": real_frames["top_camera"],
            "tool_camera": real_frames["tool_camera"],
            "side_camera": real_frames["side_camera"],
        }
        if cfg.skip_side_camera:
            frames_for_recorder.pop("side_camera", None)

        recorder.add(
            frames_for_recorder,
            real_qpos_deg,
            real_ee.astype(np.float32),
            real_delta_ee.astype(np.float32),
            float(data.time),
            1,  # phase=1 (hold is still annotated as align in sim)
            float(current_sensor_dist),
            needle_tip_mm=needle_tip_mm.astype(np.float32),
            trocar_entry_mm=trocar_entry_mm.astype(np.float32),
            keypoints_wrist=keypoints_wrist,
            keypoints_visibility=keypoints_visibility,
            instruction=TASK_INSTRUCTION,
        )
        last_ee_pose_sim = sim_ee_pose_mm.copy()
        last_real_ee = real_ee.copy()

    if len(recorder.buffer) > 0:
        recorder.save_async()
        logger.info(f"  ✅ Episode {ep_idx} saved ({ctrl_step + HOLD_RECORD_STEPS} ctrl frames)")
        return True

    recorder.discard()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Real-robot fine-alignment dataset collection via sim twin")
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--phantom-pos", type=float, nargs=2, required=True,
                    metavar=("X", "Y"), help="Real phantom XY in robot base frame (m)")
    ap.add_argument("--phantom-rot", type=float, default=None,
                    help="Phantom yaw (deg). Default: auto by Y (>=-0.25 → 0, else -90)")
    ap.add_argument("--mujoco-xml", type=str,
                    default=str(_PROJECT_ROOT / "Sim" / "meca_add.xml"))
    ap.add_argument("--save-dir", type=str,
                    default=str(_PROJECT_ROOT / "dataset" / "real_align" / "collected_data_real"))
    ap.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    ap.add_argument("--swap-cameras", action="store_true",
                    help="Swap camera1↔camera2 → top/tool mapping")
    ap.add_argument("--max-steps", type=int, default=MAX_CTRL_STEPS,
                    help=f"Max recording ctrl frames before discard (default: {MAX_CTRL_STEPS})")
    ap.add_argument("--mirror-mode", choices=["joint", "cartesian"], default="joint")
    ap.add_argument("--control-dt", type=float, default=0.134,
                    help="Wall-time per recorded frame (~67 mj_steps × 0.002s). "
                         "Used by cartesian mode only; joint mode paces via --stream-rate-hz.")
    ap.add_argument("--stream-rate-hz", type=float, default=15.0,
                    help="Joint-mode streaming rate to the robot (Hz). 15 Hz empirically "
                         "minimizes wrist-camera jitter (motion duration ≈ command interval). "
                         "Recording rate stays 7.46 Hz regardless.")
    ap.add_argument("--joint-vel-limit", type=float, default=50.0,
                    help="Mecademic SetJointVelLimit (deg/s).")
    ap.add_argument("--cart-lin-vel", type=float, default=50.0,
                    help="Mecademic SetCartLinVel (mm/s). Cartesian-mode only.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip robot connect; sim+OAK only")
    ap.add_argument("--skip-side-camera", action="store_true",
                    help="Don't record side_camera (real has only 2 cams)")
    ap.add_argument("--no-display", action="store_true",
                    help="Disable cv2.imshow live preview (auto-on if $DISPLAY unset)")
    ap.add_argument("--seed", type=int, default=None)
    # Safety: pre-flight sim dry-run validation
    ap.add_argument("--max-perturb-retries", type=int, default=5,
                    help="Per episode, resample the perturbation up to N times if "
                         "sim dry-run flags occlusion/collision risk. After N failures "
                         "the episode is skipped (next episode samples fresh).")
    ap.add_argument("--no-safety-validation", action="store_true",
                    help="Disable pre-flight sim dry-run check. NOT recommended on real "
                         "robot — needle may collide with phantom.")
    ap.add_argument("--occlusion-tolerance-mm", type=float, default=1.0,
                    help="Ray-cast tolerance: a hit closer than (tip_dist - tol) counts "
                         "as the tip being occluded by phantom. Smaller = stricter.")
    ap.add_argument("--qpos-path-steps", type=int, default=20,
                    help="Number of intermediate samples along the joint-linear path "
                         "real takes from aligned→perturbed (Mecademic MoveJoints).")
    ap.add_argument("--inter-episode-retreat-mm", type=float, default=30.0,
                    help="Between episodes, retract the needle by this many mm beyond "
                         "RETREAT_MM along the trocar axis, then go to HOME, then back "
                         "to ALIGN before the next episode. Avoids continuous joint-space "
                         "transitions near the phantom that risk collision. 0 disables.")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Save dir: {save_dir}")

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    # Sim model
    logger.info(f"📦 Loading MuJoCo model: {args.mujoco_xml}")
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    data = mujoco.MjData(model)
    h = SimHandles(model)
    if h.phantom_body_id < 0 or h.tip_id < 0 or h.target_entry_id < 0:
        raise RuntimeError("Required sim sites/bodies missing. Check meca_add.xml")

    # Real env
    env = RealCollectEnv(
        robot_address=args.robot_address,
        swap_cameras=args.swap_cameras,
        dry_run=args.dry_run,
        joint_vel_limit_deg_s=float(args.joint_vel_limit),
        cart_lin_vel_mm_s=float(args.cart_lin_vel),
    )

    # Recorder
    recorder = SimRecorder(str(save_dir))

    # cfg as a simple namespace
    class _Cfg: pass
    cfg = _Cfg()
    cfg.phantom_pos = tuple(args.phantom_pos)
    cfg.phantom_rot = args.phantom_rot
    cfg.mirror_mode = args.mirror_mode
    cfg.control_dt = float(args.control_dt)
    # Joint-mode high-rate streaming (Mecademic motion blending stays continuous).
    sim_dt = 0.002
    target_dt = 1.0 / max(1e-3, float(args.stream_rate_hz))
    cfg.stream_every = max(1, int(round(target_dt / sim_dt)))
    cfg.stream_dt = cfg.stream_every * sim_dt
    logger.info(
        f"⚙️  Joint streaming: every {cfg.stream_every} sim-steps → "
        f"{1.0/cfg.stream_dt:.1f} Hz (pace {cfg.stream_dt*1000:.1f} ms/tick)"
    )
    cfg.max_steps = int(args.max_steps)
    cfg.skip_side_camera = bool(args.skip_side_camera)
    cfg.show_preview = not (args.no_display or not os.environ.get("DISPLAY"))
    # Safety knobs (sim dry-run validation)
    cfg.skip_safety_validation = bool(args.no_safety_validation)
    cfg.max_perturb_retries = int(args.max_perturb_retries)
    cfg.occlusion_tol_m = float(args.occlusion_tolerance_mm) / 1000.0
    cfg.qpos_path_steps = int(args.qpos_path_steps)
    cfg.inter_episode_retreat_mm = float(args.inter_episode_retreat_mm)
    if cfg.skip_safety_validation:
        logger.warning("⚠️  Safety validation DISABLED — needle may collide with phantom on real robot.")
    else:
        logger.info(
            f"🛡  Safety: dry-run validation ON (retries≤{cfg.max_perturb_retries}, "
            f"occl_tol={cfg.occlusion_tol_m*1000:.1f}mm, "
            f"qpos_path_samples={cfg.qpos_path_steps})"
        )

    n_ok = 0
    try:
        # ① Pre-alignment (one-time, sim → real)
        logger.info("🎯 Pre-alignment (sim) — solving aligned pose…")
        # Place phantom for sim's pre-alignment (matches per-episode placement)
        mujoco.mj_resetData(model, data)
        _set_phantom(model, data, h, cfg.phantom_pos, cfg.phantom_rot)
        aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len = _pre_align_sim(model, data, h)
        logger.info(f"   sim aligned_qpos (deg) = {np.rad2deg(aligned_qpos).round(1).tolist()}")

        # Pre-flight safety: aligned endpoint + HOME_JOINTS→aligned joint-linear path
        # (the real robot's actual trajectory). Fail-fast on misplaced phantom before
        # the real robot ever moves.
        if not cfg.skip_safety_validation:
            if _check_tip_occluded(model, data, h, cfg.occlusion_tol_m):
                raise RuntimeError(
                    f"Pre-align tip is occluded at aligned pose — phantom_pos={cfg.phantom_pos} "
                    f"likely misplaced (trocar entry not visible from tool_camera). "
                    f"Verify --phantom-pos against the real phantom position before retrying."
                )
            home_rad = np.deg2rad(np.asarray(HOME_JOINTS, dtype=np.float64))
            sim_check = _clone_state(model, data)  # currently at aligned
            ok, reason = _check_qpos_path_safe(
                model, sim_check, h, home_rad, aligned_qpos,
                n_steps=cfg.qpos_path_steps, tolerance_m=cfg.occlusion_tol_m,
            )
            if not ok:
                raise RuntimeError(
                    f"Pre-align HOME→aligned joint-linear path occluded ({reason}). "
                    f"Real robot would collide on its first move. "
                    f"Adjust HOME_JOINTS or phantom placement."
                )
            logger.info("   ✓ Pre-align safety check passed (aligned endpoint + HOME→aligned path)")

        # Pre-compute inter-episode safe retreat waypoint (sim-only; doesn't
        # mutate live `data`). Pulled further back along the trocar axis than
        # the aligned pose so HOME→aligned and post-episode retreat→HOME paths
        # have a clean clearance margin.
        retreated_deg = None
        if cfg.inter_episode_retreat_mm > 0:
            sim_clone = _clone_state(model, data)  # data is at aligned post pre-align
            retreated_qpos = _solve_retreat_qpos(
                model, sim_clone, h, p_entry, p_depth, needle_len,
                extra_retreat_mm=cfg.inter_episode_retreat_mm,
            )
            retreated_deg = np.rad2deg(retreated_qpos)
            logger.info(
                f"   inter-episode retreat qpos (deg, +{cfg.inter_episode_retreat_mm:.0f}mm) = "
                f"{retreated_deg.round(2).tolist()}"
            )

        # Send the aligned pose to real robot (single big move).
        # CAUTION: real robot will travel from HOME_JOINTS to aligned_qpos.
        logger.info("🤖 Mirroring aligned_qpos to real robot (blocking MoveJoints)…")
        env.reset_to_joints(np.rad2deg(aligned_qpos))

        aligned_deg = np.rad2deg(aligned_qpos)
        home_deg = list(HOME_JOINTS)

        # ② Per-episode loop
        for ep in range(1, args.num_episodes + 1):
            logger.info("\n" + "=" * 60 + f"\n▶ Align Episode {ep}/{args.num_episodes}\n" + "=" * 60)
            ok = run_collection_episode(
                model, data, env, recorder, cfg, h, ep, rng,
                aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len,
            )
            if ok:
                n_ok += 1

            # Inter-episode safety: retreat → HOME → re-approach aligned for next ep.
            # Skip the re-approach after the very last episode (just retreat → HOME
            # for a clean shutdown state).
            if retreated_deg is not None:
                logger.info("↩️  Inter-episode safety: retreat along axis → HOME")
                env.reset_to_joints(retreated_deg)
                env.reset_to_joints(home_deg)
                if ep < args.num_episodes:
                    logger.info("🤖 Re-approach: HOME → ALIGN for next episode")
                    env.reset_to_joints(aligned_deg)
    except KeyboardInterrupt:
        logger.warning("\n🛑 KeyboardInterrupt — finishing pending saves and shutting down")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        recorder.wait_for_all()
        env.close()
        logger.info(f"\n✅ Done. {n_ok}/{args.num_episodes} episodes saved to {save_dir}")


if __name__ == "__main__":
    main()
