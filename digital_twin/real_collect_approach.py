"""
real_collect_approach.py

Real-robot dataset collection for the *approach* phase, driven by a MuJoCo "sim twin".

Architecture
------------
- A MuJoCo simulation runs alongside the real Mecademic Meca500.
- The sim's scripted approach trajectory (smooth_step + hierarchical IK toward the
  trocar entry, exactly as in `Sim/Save_dataset_approach_only.py`) decides each
  control frame.
- Per recording frame (every 67 sim steps ≈ 0.134 s) we either:
    * `joint`  mode: send sim's `data.qpos[:6]` (deg) to the robot via `MoveJoints`
    * `cartesian` mode: send sim's frame-to-frame `delta_ee` via `MovePose`
- After the real robot has had `control_dt` to move, we capture **real** OAK frames
  + **real** GetPose / GetJoints, compute `delta_ee = real_ee_now - real_ee_last`,
  and append to HDF5 in the **same schema** used by the sim collector
  (`Sim/Save_dataset_approach_only.py:SimRecorder`).

The `needle_tip_pos`, `trocar_entry_pos`, `keypoints_wrist`, `keypoints_visibility`,
`phase`, and `metadata/*` fields are filled from sim — sim is running anyway and
real cannot observe these directly. This keeps the HDF5 100% drop-in compatible
with `src/datasets/sim_act_approach.py` for sim+real merged training.

Usage
-----
    bash Run_Collect_Real_Approach.sh --num-episodes 10 --phantom-pos 0.0 -0.4
    bash Run_Collect_Real_Approach.sh --dry-run --num-episodes 1 --max-steps 30
"""

import os
# Headless MuJoCo (we never render sim images, but the model's cam_xpos / cam_xmat
# get computed regardless; setting EGL avoids any GL-related import surprises).
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import argparse
import logging
import pathlib

import numpy as np
import cv2
import mujoco

import mecademicpy.robot as mdr

# Path setup: project root + Sim/ for direct module import
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Sim"))

# Reuse from sim collector (do NOT modify Save_dataset_approach_only.py)
from Save_dataset_approach_only import (  # noqa: E402
    SimRecorder,
    smooth_step,
    project_to_2d,
    ALIGN_SPEED,
    ACTION_CLIP_MM,
    HOLD_STEPS as SIM_HOLD_STEPS,
    RETREAT_MM as SIM_RETREAT_MM,
    WARMUP_STEPS,
    IMG_WIDTH,
    IMG_HEIGHT,
    TASK_INSTRUCTION,
)

# Reuse OAK camera helper from the eval script (already battle-tested with our cameras)
from digital_twin.real_eval_approach import (  # noqa: E402
    OAKCameraManager,
    HOME_JOINTS,
    ROBOT_ADDRESS_DEFAULT,
    SAFETY_CLAMP_POS_MM,
    SAFETY_CLAMP_ROT_RAD,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Real env: thin extension of ApproachRealEnv with joint-mirror + state read APIs
# ─────────────────────────────────────────────────────────────────────────────
class RealCollectEnv:
    """Mecademic + OAK wrapper for sim-twin data collection.

    Joint mirror (`stream_joints`) and Cartesian delta (`stream_cartesian_delta`)
    are both supported; the caller picks per `cfg.mirror_mode`.
    """

    def __init__(self, robot_address=ROBOT_ADDRESS_DEFAULT, swap_cameras=False,
                 dry_run=False, joint_vel_limit_deg_s=20.0, cart_lin_vel_mm_s=50.0):
        self.swap_cameras = swap_cameras
        self.dry_run = dry_run
        self.robot = None
        self.cam_mgr = None

        # Robot
        if not dry_run:
            logger.info(f"🔌 Connecting to robot at {robot_address}…")
            self.robot = mdr.Robot()
            self.robot.Connect(address=robot_address)
            if not self.robot.IsConnected():
                raise RuntimeError(f"Failed to connect to robot at {robot_address}")
            logger.info("✅ Robot connected. Activating + homing…")
            self.robot.ActivateAndHome()
            self.robot.SetRealTimeMonitoring(1)
            try:
                # Conservative speed limits for sim-driven motion; user can override
                # downstream if 7.5 Hz frame rate produces visible lag.
                self.robot.SetJointVelLimit(joint_vel_limit_deg_s)
                self.robot.SetCartLinVel(cart_lin_vel_mm_s)
            except Exception as e:
                logger.warning(f"SetJointVelLimit/SetCartLinVel failed (non-fatal): {e}")
            self.robot.MoveJoints(*HOME_JOINTS)
            self.robot.WaitIdle()
            logger.info(f"🏠 Home joints {HOME_JOINTS} reached")
        else:
            logger.warning("⚠️ DRY-RUN: skipping robot connect; commands will be no-ops")

        # Cameras (always — even in dry-run we want real frames recorded if available)
        self.cam_mgr = OAKCameraManager(width=IMG_WIDTH, height=IMG_HEIGHT)
        try:
            n_cams = self.cam_mgr.initialize_cameras()
            logger.info(f"📷 {n_cams} OAK camera(s): {self.cam_mgr.camera_ids}")
            if n_cams < 1:
                logger.warning("⚠️ No OAK cameras — frames will be black")
            elif n_cams < 2:
                logger.warning("⚠️ Only 1 camera — tool_camera will be blank")
            # Warm cameras up
            for _ in range(15):
                self.cam_mgr.get_frames(blocking_timeout=0.1)
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Camera init failed: {e}; falling back to blank frames")
            self.cam_mgr = None

    # ─── lifecycle ──────────────────────────────────────────────────────────
    def reset_to_joints(self, joints_deg):
        """Move to a specific home pose synchronously (used at episode start only)."""
        if self.dry_run or self.robot is None:
            return
        try:
            self.robot.MoveJoints(*[float(j) for j in joints_deg])
            self.robot.WaitIdle()
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"reset_to_joints failed: {e}; ResetError + retry")
            self._recover()
            try:
                self.robot.MoveJoints(*[float(j) for j in joints_deg])
                self.robot.WaitIdle()
            except Exception as e2:
                logger.error(f"reset_to_joints retry failed: {e2}")

    def _recover(self):
        if self.robot is None:
            return
        try:
            self.robot.ResetError()
            self.robot.ResumeMotion()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    # ─── streaming APIs (per control frame) ─────────────────────────────────
    def stream_joints(self, joints_deg):
        """Joint-space mirror — queued, no WaitIdle."""
        if self.dry_run or self.robot is None:
            logger.debug(f"[DRY] MoveJoints {[round(float(j), 2) for j in joints_deg]}")
            return
        try:
            self.robot.MoveJoints(*[float(j) for j in joints_deg])
        except Exception as e:
            logger.warning(f"MoveJoints failed: {e}")
            self._recover()

    def stream_cartesian_delta(self, delta_ee_6d):
        """Cartesian delta mirror — target = current_pose + delta, MovePose absolute."""
        delta = np.asarray(delta_ee_6d, dtype=np.float32).copy()
        delta[:3] = np.clip(delta[:3], -SAFETY_CLAMP_POS_MM, SAFETY_CLAMP_POS_MM)
        delta[3:6] = np.clip(delta[3:6], -SAFETY_CLAMP_ROT_RAD, SAFETY_CLAMP_ROT_RAD)
        if self.dry_run or self.robot is None:
            logger.debug(f"[DRY] MovePose delta={delta.round(3).tolist()}")
            return
        try:
            current = list(self.robot.GetPose())[:6]
        except Exception as e:
            logger.warning(f"GetPose failed in stream_cartesian_delta: {e}")
            return
        target = [
            float(current[0] + delta[0]),
            float(current[1] + delta[1]),
            float(current[2] + delta[2]),
            float(current[3] + np.rad2deg(delta[3])),
            float(current[4] + np.rad2deg(delta[4])),
            float(current[5] + np.rad2deg(delta[5])),
        ]
        try:
            self.robot.MovePose(*target)
        except Exception as e:
            logger.warning(f"MovePose failed: {e}")
            self._recover()

    # ─── observation ─────────────────────────────────────────────────────────
    def read_state(self) -> dict:
        """Return real robot state in sim-compatible units.

        - qpos_deg: (6,) float32 — degrees (Meca native)
        - ee_pose:  (6,) float32 — [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
        """
        if self.dry_run or self.robot is None:
            return {
                "qpos_deg": np.zeros(6, dtype=np.float32),
                "ee_pose": np.zeros(6, dtype=np.float32),
            }
        for _ in range(5):
            try:
                q = list(self.robot.GetJoints())[:6]
                p = list(self.robot.GetPose())[:6]
                qpos_deg = np.asarray(q, dtype=np.float32)
                ee = np.asarray(p, dtype=np.float32)
                ee_pose = np.concatenate([ee[:3], np.deg2rad(ee[3:6])]).astype(np.float32)
                return {"qpos_deg": qpos_deg, "ee_pose": ee_pose}
            except Exception:
                time.sleep(0.01)
        logger.warning("read_state: GetJoints/GetPose failed; returning zeros")
        return {
            "qpos_deg": np.zeros(6, dtype=np.float32),
            "ee_pose": np.zeros(6, dtype=np.float32),
        }

    def render_frames(self) -> dict:
        """Return raw BGR uint8 frames keyed by sim role names.

        Schema is fixed to match `Sim/Save_dataset_approach_only.py:CAMERA_LIST`:
            top_camera, tool_camera, side_camera.
        side_camera does not exist in our real rig → filled with zeros so the
        HDF5 schema stays drop-in.
        """
        blank = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        cam1 = cam2 = None
        if self.cam_mgr is not None:
            try:
                raw = self.cam_mgr.get_frames(blocking_timeout=0.2)
                cam1 = raw.get("camera1")
                cam2 = raw.get("camera2")
            except Exception as e:
                logger.debug(f"OAK get_frames error: {e}")
        if self.swap_cameras:
            cam1, cam2 = cam2, cam1
        return {
            "top_camera": cam1 if cam1 is not None else blank,
            "tool_camera": cam2 if cam2 is not None else blank,
            "side_camera": blank,
        }

    def close(self):
        if self.cam_mgr is not None:
            try:
                self.cam_mgr.close()
            except Exception as e:
                logger.error(f"Camera close failed: {e}")
        if self.robot is not None:
            try:
                if self.robot.IsConnected():
                    self.robot.DeactivateRobot()
                    self.robot.Disconnect()
                    logger.info("🔌 Robot disconnected")
            except Exception as e:
                logger.error(f"Robot close failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sim helpers — site IDs + FK ee_pose (matches Save_dataset_approach_only.py)
# ─────────────────────────────────────────────────────────────────────────────
class SimHandles:
    """Bundle of MuJoCo IDs + EE FK helper. Mirrors the locals in sim main()."""

    def __init__(self, model):
        m = mujoco
        self.tip_id = m.mj_name2id(model, m.mjtObj.mjOBJ_SITE, "needle_tip")
        self.back_id = m.mj_name2id(model, m.mjtObj.mjOBJ_SITE, "needle_back")
        self.target_entry_id = m.mj_name2id(model, m.mjtObj.mjOBJ_SITE, "trocar_target")
        self.target_depth_id = m.mj_name2id(model, m.mjtObj.mjOBJ_SITE, "trocar_depth")
        self.phantom_body_id = m.mj_name2id(model, m.mjtObj.mjOBJ_BODY, "phantom_assembly")
        self.rotating_id = m.mj_name2id(model, m.mjtObj.mjOBJ_BODY, "rotating_assembly")
        self.link6_id = m.mj_name2id(model, m.mjtObj.mjOBJ_BODY, "6_Link")
        self.n_motors = model.nu
        self.dof = model.nv

    def ee_pose_mm_rad(self, data):
        """Same convention as Save_dataset_approach_only.get_ee_pose_6d_scaled (xyz mm, rpy rad)."""
        if self.link6_id < 0:
            return np.zeros(6, dtype=np.float32)
        pos = data.xpos[self.link6_id].copy() * 1000.0
        mat = data.xmat[self.link6_id].reshape(3, 3)
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        if sy > 1e-6:
            r = np.arctan2(mat[2, 1], mat[2, 2])
            p = np.arctan2(-mat[2, 0], sy)
            y = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            r = np.arctan2(-mat[1, 2], mat[1, 1])
            p = np.arctan2(-mat[2, 0], sy)
            y = 0.0
        return np.concatenate([pos, [r, p, y]]).astype(np.float32)


def _solve_ik_step(model, data, h: SimHandles, target_tip, target_back, current_speed):
    """Hierarchical null-space IK identical to Save_dataset_approach_only.py:500-531."""
    curr_tip = data.site_xpos[h.tip_id].copy()
    curr_back = data.site_xpos[h.back_id].copy()
    err_tip = target_tip - curr_tip
    err_back = target_back - curr_back

    tip_rot_mat = data.site_xmat[h.tip_id].reshape(3, 3)
    offset_angle = np.deg2rad(180 + 30)
    offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
    current_side_vec = tip_rot_mat @ offset_local_vec

    needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
    target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
    nrm = np.linalg.norm(target_side_vec)
    target_side_vec = target_side_vec / nrm if nrm > 1e-3 else np.array([1.0, 0.0, 0.0])
    err_roll = np.cross(current_side_vec, target_side_vec)

    jac_tip_full = np.zeros((6, h.dof))
    jac_back = np.zeros((3, h.dof))
    mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], h.tip_id)
    mujoco.mj_jacSite(model, data, jac_back, None, h.back_id)

    J1, e1 = jac_tip_full[:3, : h.n_motors], err_tip * 50.0
    if np.linalg.norm(e1) > 1.0:
        e1 = e1 / np.linalg.norm(e1) * 1.0
    J1_pinv = np.linalg.pinv(J1, rcond=1e-4)
    dq1 = J1_pinv @ e1
    P1 = np.eye(h.n_motors) - (J1_pinv @ J1)

    J2_proj = jac_back[:, : h.n_motors] @ P1
    dq2 = np.linalg.pinv(J2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, : h.n_motors] @ dq1)
    P2 = P1 - (np.linalg.pinv(J2_proj, rcond=1e-4) @ J2_proj)

    J3_proj = jac_tip_full[3:, : h.n_motors] @ P2
    dq3 = np.linalg.pinv(J3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, : h.n_motors] @ (dq1 + dq2))

    data.ctrl[: h.n_motors] = data.qpos[: h.n_motors] + (dq1 + dq2 + dq3) * current_speed


# ─────────────────────────────────────────────────────────────────────────────
# Episode helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sample_home_pose(randomize: bool, seed_rng: np.random.Generator) -> np.ndarray:
    """Match Save_dataset_approach_only.py:351-358 distribution (radians).

    Sim has J3 written as `np.random.uniform(0.75, 0.25)` — legacy numpy's
    `np.random.uniform` silently swapped reversed bounds, but the new
    `Generator.uniform` raises. Normalizing to `(0.25, 0.75)` reproduces
    the same effective range without changing the distribution.
    """
    if not randomize:
        return np.array([0.0, -0.5, 0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    return np.array([
        seed_rng.uniform(-0.5, 0.5),
        seed_rng.uniform(-0.6, -0.4),
        seed_rng.uniform(0.25, 0.75),
        seed_rng.uniform(-0.3, 0.3),
        seed_rng.uniform(0.4, 0.6),
        seed_rng.uniform(0.9, 1.1),
    ], dtype=np.float64)


def _set_phantom(model, data, h: SimHandles, phantom_xy, phantom_rot_deg=None):
    """Place the phantom at (px, py, 0) and rotate per sim convention."""
    if h.phantom_body_id < 0:
        return np.zeros(3, np.float32), np.array([1, 0, 0, 0], np.float32), np.float32(0.0)
    px, py = float(phantom_xy[0]), float(phantom_xy[1])
    model.body_pos[h.phantom_body_id] = np.array([px, py, 0.0])

    if phantom_rot_deg is None:
        rand_angle = 0 if py >= -0.25 else -90
    else:
        rand_angle = float(phantom_rot_deg)

    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(rand_angle)], "xyz")
    if h.rotating_id >= 0:
        model.body_quat[h.rotating_id] = new_quat
    mujoco.mj_forward(model, data)
    return (
        np.array([px, py, 0.0], np.float32),
        new_quat.astype(np.float32),
        np.float32(rand_angle),
    )


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _build_display_frame(real_frames, ctrl_step, max_steps, task_state,
                         sim_delta_pos_mm, sim_delta_rot_deg, mirror_mode,
                         dry_run):
    top = real_frames["top_camera"]
    tool = real_frames["tool_camera"]
    combined = np.concatenate([top, tool], axis=1)
    cv2.putText(combined, f"step {ctrl_step}/{max_steps}  phase {task_state}  mode={mirror_mode}",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(combined, f"sim_dpos_mm={sim_delta_pos_mm.round(2).tolist()}",
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(combined, f"sim_drot_deg={sim_delta_rot_deg.round(2).tolist()}",
                (5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    if dry_run:
        cv2.putText(combined, "DRY-RUN", (5, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop (mirrors Sim/Save_dataset_approach_only.py main loop,
# with real-robot streaming + real frame/state recording at each control frame)
# ─────────────────────────────────────────────────────────────────────────────
def run_collection_episode(model, data, env: RealCollectEnv, recorder: SimRecorder,
                           cfg, h: SimHandles, ep_idx: int, rng: np.random.Generator) -> bool:
    """One sim-driven, real-recorded episode. Returns True on success."""

    # === Reset sim ===
    mujoco.mj_resetData(model, data)
    home_pose_rad = _sample_home_pose(cfg.randomize_home, rng)
    data.qpos[:6] = home_pose_rad
    phantom_offset, phantom_quat, phantom_angle_deg = _set_phantom(
        model, data, h, cfg.phantom_pos, cfg.phantom_rot
    )
    mujoco.mj_forward(model, data)

    # === Reset real to same home (deg) ===
    home_pose_deg = np.rad2deg(home_pose_rad)
    logger.info(f"  Episode {ep_idx}: home (deg) = {home_pose_deg.round(1).tolist()}")
    env.reset_to_joints(home_pose_deg)

    p_entry = data.site_xpos[h.target_entry_id].copy()
    p_depth = data.site_xpos[h.target_depth_id].copy()
    start_tip = data.site_xpos[h.tip_id].copy()
    start_back = data.site_xpos[h.back_id].copy()
    needle_len = np.linalg.norm(start_tip - start_back)
    current_speed = 0.5  # matches sim

    recorder.start({
        "initial_qpos": np.rad2deg(data.qpos[:h.n_motors].copy()).astype(np.float32),
        "phantom_offset": phantom_offset,
        "phantom_quat": phantom_quat,
        "phantom_angle_deg": np.array(phantom_angle_deg, dtype=np.float32),
        "target_entry_world": p_entry.astype(np.float32),
        "target_depth_world": p_depth.astype(np.float32),
    })

    # === Warm-up (sim only — settle J6 before recording starts) ===
    for _ in range(WARMUP_STEPS):
        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()
        _solve_ik_step(model, data, h, curr_tip, curr_back, current_speed)
        mujoco.mj_step(model, data)

    last_ee_pose_sim = h.ee_pose_mm_rad(data)
    # Sync real to post-warmup sim pose so the first recorded delta isn't a giant jump
    post_warmup_qpos_deg = np.rad2deg(data.qpos[:6].copy())
    if cfg.mirror_mode == "joint":
        env.stream_joints(post_warmup_qpos_deg)
    time.sleep(0.4)
    last_real_state = env.read_state()
    last_real_ee = last_real_state["ee_pose"]

    # === Trajectory state ===
    task_state = 1
    traj_initialized = False
    traj_start_time = data.time
    align_timer = 0
    hold_frame_count = 0
    success = False
    user_quit = False
    step_count = 0
    ctrl_step = 0
    dynamic_duration = 1.0  # set on first pass

    while True:
        t_curr = data.time
        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()

        # ─── 1. Trajectory logic (approach-only: states 1 → 3) ────────────
        if task_state == 1:  # Align
            if not traj_initialized:
                traj_start_time = t_curr
                start_tip_pos = curr_tip.copy()
                start_back_pos = curr_back.copy()
                traj_initialized = True
                axis_dir_init = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                retreat_m = SIM_RETREAT_MM / 1000.0
                goal_tip_init = p_entry - (axis_dir_init * retreat_m)
                align_distance = np.linalg.norm(goal_tip_init - start_tip_pos)
                dynamic_duration = max(align_distance / ALIGN_SPEED, 1e-3)
            progress = smooth_step((t_curr - traj_start_time) / dynamic_duration) if dynamic_duration > 0 else 1.0
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            retreat_m = SIM_RETREAT_MM / 1000.0
            goal_tip = p_entry - (axis_dir * retreat_m)
            goal_back = p_entry - (axis_dir * (retreat_m + needle_len))
            target_tip = (1 - progress) * start_tip_pos + progress * goal_tip
            target_back = (1 - progress) * start_back_pos + progress * goal_back

            if progress >= 1.0:
                if np.linalg.norm(curr_tip - goal_tip) < 0.002:
                    align_timer += 1
                else:
                    align_timer = 0
                if align_timer > 50:
                    # approach-only collection: transition to Hold (state 3)
                    task_state = 3
        elif task_state == 3:  # Hold (approach reached, keep still)
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            retreat_m = SIM_RETREAT_MM / 1000.0
            target_tip = p_entry - (axis_dir * retreat_m)
            target_back = target_tip - (axis_dir * needle_len)
        else:  # safety: shouldn't happen
            target_tip = curr_tip
            target_back = curr_back

        # ─── 2. IK solve → data.ctrl ─────────────────────────────────────
        _solve_ik_step(model, data, h, target_tip, target_back, current_speed)

        # ─── 3. sensor_dist (sim ray) ────────────────────────────────────
        p_sensor = data.site_xpos[h.tip_id].copy()
        nlen = np.linalg.norm(curr_tip - curr_back) + 1e-10
        needle_dir = (curr_tip - curr_back) / nlen
        dist_to_surface = mujoco.mj_ray(
            model, data, p_sensor, needle_dir, None, 1, h.link6_id, np.zeros(1, dtype=np.int32)
        )
        current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

        # ─── 4. Step sim ─────────────────────────────────────────────────
        mujoco.mj_step(model, data)
        step_count += 1

        # ─── 5. Recording frame (every 67 sim steps) ─────────────────────
        if step_count % 67 == 0:
            ctrl_step += 1

            # Sim state at this frame
            sim_ee_pose_mm = h.ee_pose_mm_rad(data)
            sim_delta_ee = sim_ee_pose_mm - last_ee_pose_sim
            sim_delta_ee[3:6] = _wrap_pi(sim_delta_ee[3:6])
            pos_mag = np.linalg.norm(sim_delta_ee[:3])
            if pos_mag > ACTION_CLIP_MM:
                sim_delta_ee[:3] *= ACTION_CLIP_MM / pos_mag

            # 5a. Drive real
            if cfg.mirror_mode == "joint":
                env.stream_joints(np.rad2deg(data.qpos[:6].copy()))
            else:
                env.stream_cartesian_delta(sim_delta_ee)

            # 5b. Wall-time matched to sim (67 mj_step ≈ 0.134 s with default timestep)
            time.sleep(cfg.control_dt)

            # 5c. Capture real frames + real state (post-motion)
            real_frames = env.render_frames()
            real_state = env.read_state()
            real_qpos_deg = real_state["qpos_deg"]
            real_ee = real_state["ee_pose"]

            real_delta_ee = real_ee - last_real_ee
            real_delta_ee[3:6] = _wrap_pi(real_delta_ee[3:6])
            real_pos_mag = np.linalg.norm(real_delta_ee[:3])
            # Real allows wider clip than sim (tracking lag) — 5x the sim clip
            real_clip_mm = ACTION_CLIP_MM * 5.0
            if real_pos_mag > real_clip_mm:
                real_delta_ee[:3] *= real_clip_mm / real_pos_mag

            # 5d. Sim aux (needle / trocar / wrist keypoints — sim-frame only)
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

            # 5e. Append to recorder — REAL frames, REAL state, REAL action
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
                int(task_state),
                float(current_sensor_dist),
                needle_tip_mm=needle_tip_mm.astype(np.float32),
                trocar_entry_mm=trocar_entry_mm.astype(np.float32),
                keypoints_wrist=keypoints_wrist,
                keypoints_visibility=keypoints_visibility,
                instruction=TASK_INSTRUCTION,
            )

            last_ee_pose_sim = sim_ee_pose_mm.copy()
            last_real_ee = real_ee.copy()

            # 5f. Display + 'q' abort (skip silently in headless mode)
            if cfg.show_preview:
                try:
                    disp = _build_display_frame(
                        real_frames, ctrl_step, cfg.max_steps, task_state,
                        sim_delta_ee[:3], np.rad2deg(sim_delta_ee[3:6]),
                        cfg.mirror_mode, env.dry_run,
                    )
                    cv2.imshow("Real Collect (top | tool) — 'q' abort", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        logger.warning("🛑 'q' pressed — aborting episode")
                        user_quit = True
                        break
                except Exception as e:
                    logger.debug(f"Preview disabled after error: {e}")
                    cfg.show_preview = False

            if ctrl_step % 10 == 0:
                logger.info(
                    f"    ctrl {ctrl_step:3d}/{cfg.max_steps}  phase {task_state}  "
                    f"sim_dpos_mm={sim_delta_ee[:3].round(2).tolist()}  "
                    f"real_dpos_mm={real_delta_ee[:3].round(2).tolist()}"
                )

            # 5g. Hold completion
            if task_state == 3:
                hold_frame_count += 1
                if hold_frame_count >= cfg.hold_steps:
                    success = True
                    break

        # ─── Termination guards ───────────────────────────────────────────
        if data.time - traj_start_time > 50.0:
            logger.warning(f"  Episode {ep_idx}: timeout (>50s sim)")
            break
        if ctrl_step >= cfg.max_steps:
            logger.warning(f"  Episode {ep_idx}: hit max_steps={cfg.max_steps}")
            break

    # Flush episode → HDF5
    if success and not user_quit:
        recorder.save_async()
        logger.info(f"  ✅ Episode {ep_idx} saved ({ctrl_step} ctrl frames)")
    else:
        recorder.discard()
        reason = "user_quit" if user_quit else (
            "timeout" if data.time - traj_start_time > 50.0 else "max_steps"
        )
        logger.warning(f"  ❌ Episode {ep_idx} discarded ({reason})")

    return success and not user_quit


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Real-robot dataset collection (approach phase) via sim twin")
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--phantom-pos", type=float, nargs=2, required=True,
                    metavar=("X", "Y"), help="Real phantom XY in robot base frame (m)")
    ap.add_argument("--phantom-rot", type=float, default=None,
                    help="Phantom yaw (deg). Default: auto by Y (>=-0.25 → 0, else -90)")
    ap.add_argument("--mujoco-xml", type=str,
                    default=str(_PROJECT_ROOT / "Sim" / "meca_add.xml"))
    ap.add_argument("--save-dir", type=str,
                    default=str(_PROJECT_ROOT / "dataset" / "real_approach" / "collected_data_real"))
    ap.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    ap.add_argument("--swap-cameras", action="store_true",
                    help="Swap camera1↔camera2 → top/tool mapping")
    ap.add_argument("--randomize-home", dest="randomize_home", action="store_true", default=True)
    ap.add_argument("--fixed-home", dest="randomize_home", action="store_false")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--hold-steps", type=int, default=SIM_HOLD_STEPS,
                    help="Frames to record at hold (post-approach)")
    ap.add_argument("--mirror-mode", choices=["joint", "cartesian"], default="joint")
    ap.add_argument("--control-dt", type=float, default=0.134,
                    help="Wall-time per recorded frame (~67 mj_steps × 0.002s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip robot connect; sim+OAK only")
    ap.add_argument("--skip-side-camera", action="store_true",
                    help="Don't record side_camera (real has only 2 cams)")
    ap.add_argument("--no-display", action="store_true",
                    help="Disable cv2.imshow live preview (auto-on if $DISPLAY unset)")
    ap.add_argument("--seed", type=int, default=None)
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
    cfg.randomize_home = bool(args.randomize_home)
    cfg.max_steps = int(args.max_steps)
    cfg.hold_steps = int(args.hold_steps)
    cfg.skip_side_camera = bool(args.skip_side_camera)
    # Auto-disable cv2 preview when there's no display server (headless boxes)
    cfg.show_preview = not (args.no_display or not os.environ.get("DISPLAY"))

    n_ok = 0
    try:
        for ep in range(1, args.num_episodes + 1):
            logger.info("\n" + "=" * 60 + f"\n▶ Episode {ep}/{args.num_episodes}\n" + "=" * 60)
            ok = run_collection_episode(model, data, env, recorder, cfg, h, ep, rng)
            if ok:
                n_ok += 1
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
