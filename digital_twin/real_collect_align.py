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
from digital_twin.real_eval_approach import ROBOT_ADDRESS_DEFAULT  # noqa: E402


logger = logging.getLogger(__name__)


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

    # ③ Perturbation: sim moves to perturbed; real receives final perturbed_qpos
    perturb_xyz, perturb_angle_rad, perturb_axis = _sample_perturbation(rng)
    perturb_pos_mm = perturb_xyz * 1000.0
    logger.info(f"  Episode {ep_idx}: perturb_xyz_mm={perturb_pos_mm.round(1).tolist()}  "
                f"angle_deg={np.rad2deg(perturb_angle_rad):.1f}")

    reached = _move_to_perturbed_sim(
        model, data, h, goal_tip, goal_back,
        perturb_xyz, perturb_angle_rad, perturb_axis,
    )
    if not reached:
        logger.warning(f"  Episode {ep_idx} discarded: sim IK failed to reach perturbed pose")
        return False

    # Mirror the perturbed pose to real (single blocking MoveJoints).
    # Mecademic interpolates linearly in joint-space, which is the same trajectory
    # the sim's IK produces for a small perturbation, just faster.
    perturbed_qpos_deg = np.rad2deg(data.qpos[:6].copy())
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

        if step_count % 67 == 0:
            ctrl_step += 1

            sim_ee_pose_mm = h.ee_pose_mm_rad(data)
            sim_delta_ee = sim_ee_pose_mm - last_ee_pose_sim
            sim_delta_ee[3:6] = _wrap_pi(sim_delta_ee[3:6])
            pos_mag = np.linalg.norm(sim_delta_ee[:3])
            if pos_mag > ACTION_CLIP_MM:
                sim_delta_ee[:3] *= ACTION_CLIP_MM / pos_mag

            # Drive real (joint mirror or cartesian)
            if cfg.mirror_mode == "joint":
                env.stream_joints(np.rad2deg(data.qpos[:6].copy()))
            else:
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
        # Run 67 sim steps holding the aligned target
        for _ in range(67):
            _solve_ik_step(model, data, h, goal_tip, goal_back, 0.5)
            mujoco.mj_step(model, data)

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

        # Drive real (sim is essentially still at aligned, so deltas are tiny)
        if cfg.mirror_mode == "joint":
            env.stream_joints(np.rad2deg(data.qpos[:6].copy()))
        else:
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
    cfg.max_steps = int(args.max_steps)
    cfg.skip_side_camera = bool(args.skip_side_camera)
    cfg.show_preview = not (args.no_display or not os.environ.get("DISPLAY"))

    n_ok = 0
    try:
        # ① Pre-alignment (one-time, sim → real)
        logger.info("🎯 Pre-alignment (sim) — solving aligned pose…")
        # Place phantom for sim's pre-alignment (matches per-episode placement)
        mujoco.mj_resetData(model, data)
        _set_phantom(model, data, h, cfg.phantom_pos, cfg.phantom_rot)
        aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len = _pre_align_sim(model, data, h)
        logger.info(f"   sim aligned_qpos (deg) = {np.rad2deg(aligned_qpos).round(1).tolist()}")

        # Send the aligned pose to real robot (single big move).
        # CAUTION: real robot will travel from HOME_JOINTS to aligned_qpos.
        # Verify path is collision-free for your phantom placement first.
        logger.info("🤖 Mirroring aligned_qpos to real robot (blocking MoveJoints)…")
        env.reset_to_joints(np.rad2deg(aligned_qpos))

        # ② Per-episode loop
        for ep in range(1, args.num_episodes + 1):
            logger.info("\n" + "=" * 60 + f"\n▶ Align Episode {ep}/{args.num_episodes}\n" + "=" * 60)
            ok = run_collection_episode(
                model, data, env, recorder, cfg, h, ep, rng,
                aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len,
            )
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
