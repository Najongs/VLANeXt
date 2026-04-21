"""
sim_eval_basic_motion.py

Evaluates a basic-motion pre-trained VLANeXt model.
For each of 10 directions, gives the direction instruction and checks
whether the robot actually moves in the commanded direction.

Usage:
    python -m scripts.sim_eval_basic_motion \
        --config config/sim_eval_basic_config.yaml \
        --checkpoint /path/to/checkpoint \
        --train-config config/sim_train_basic_wrist_config.yaml
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import argparse
import yaml
import pathlib
import csv
import time

import mujoco
import numpy as np
import cv2
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.sim_act_basic import action_min_sim_basic as action_min_sim, action_max_sim_basic as action_max_sim

from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, set_seed,
    SIM_MODEL_PATH, IMG_WIDTH, IMG_HEIGHT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 10 directions (must match Save_dataset_basic_motion.py)
# ═══════════════════════════════════════════════════════════════════════════════
DIRECTIONS = {
    "left":       np.array([ 0.0,  1.0,  0.0]),
    "right":      np.array([ 0.0, -1.0,  0.0]),
    "up":         np.array([ 0.0,  0.0,  1.0]),
    "down":       np.array([ 0.0,  0.0, -1.0]),
    "left_up":    np.array([ 0.0,  1.0,  1.0]) / sqrt(2),
    "left_down":  np.array([ 0.0,  1.0, -1.0]) / sqrt(2),
    "right_up":   np.array([ 0.0, -1.0,  1.0]) / sqrt(2),
    "right_down": np.array([ 0.0, -1.0, -1.0]) / sqrt(2),
    "forward":    np.array([ 1.0,  0.0,  0.0]),
    "backward":   np.array([-1.0,  0.0,  0.0]),
}

DIRECTION_INSTRUCTIONS = {
    "left": "Left", "right": "Right", "up": "Up", "down": "Down",
    "left_up": "Left Up", "left_down": "Left Down",
    "right_up": "Right Up", "right_down": "Right Down",
    "forward": "Forward", "backward": "Backward",
}

# Random start joint ranges (same as data collection)
JOINT_RANGES = [
    (-0.5, 0.5),
    (-0.3, 0.3),
    (-0.5, 0.2),
    (-0.3, 0.3),
    (0.4, 1.0),
    (-1.0, 1.0),
]


class BasicMotionSimEnv:
    """MuJoCo env for basic motion evaluation."""

    def __init__(self, model_xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMG_HEIGHT, width=IMG_WIDTH)

        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        self.back_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        self.link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        self.n_motors = self.model.nu
        self.dof = self.model.nv

    def _run_ik_step(self, target_tip_pos, target_back_pos, speed=0.5):
        """IK solver 1 step (same as data collection)."""
        curr_tip = self.data.site_xpos[self.tip_id].copy()
        curr_back = self.data.site_xpos[self.back_id].copy()
        n = self.n_motors

        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back

        tip_rot_mat = self.data.site_xmat[self.tip_id].reshape(3, 3)
        offset_angle = np.deg2rad(180 + 30)
        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
        current_side_vec = tip_rot_mat @ offset_local_vec

        needle_axis = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side_vec = np.cross(needle_axis, np.array([0, 0, 1]))
        target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
        err_roll = np.cross(current_side_vec, target_side_vec)

        jac_tip_full = np.zeros((6, self.dof))
        jac_back = np.zeros((3, self.dof))
        mujoco.mj_jacSite(self.model, self.data, jac_tip_full[:3], jac_tip_full[3:], self.tip_id)
        mujoco.mj_jacSite(self.model, self.data, jac_back, None, self.back_id)

        J_p1 = jac_tip_full[:3, :n]
        e_p1 = err_tip * 50.0
        if np.linalg.norm(e_p1) > 1.0:
            e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
        dq_p1 = J_p1_pinv @ e_p1

        P_null_1 = np.eye(n) - (J_p1_pinv @ J_p1)
        J_p2_proj = jac_back[:, :n] @ P_null_1
        dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n] @ dq_p1)

        P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
        J_p3_proj = jac_tip_full[3:, :n] @ P_null_2
        dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n] @ (dq_p1 + dq_p2))

        self.data.ctrl[:n] = self.data.qpos[:n] + (dq_p1 + dq_p2 + dq_p3) * speed

    def reset(self, direction_vec):
        """Reset with random joint angles + IK warmup (same as data collection)."""
        mujoco.mj_resetData(self.model, self.data)
        home_pose = np.array([
            np.random.uniform(*JOINT_RANGES[i]) for i in range(6)
        ])
        self.data.qpos[:6] = home_pose
        mujoco.mj_forward(self.model, self.data)

        # Stabilize
        for _ in range(200):
            self.data.ctrl[:self.n_motors] = self.data.qpos[:self.n_motors]
            mujoco.mj_step(self.model, self.data)

        # Warm-up: IK movement without recording (same as data collection)
        warmup_tip = self.data.site_xpos[self.tip_id].copy()
        warmup_back = self.data.site_xpos[self.back_id].copy()
        fine_align_speed = 0.005  # matches FINE_ALIGN_SPEED
        for ws in range(500):
            disp = direction_vec * fine_align_speed * self.model.opt.timestep * (ws + 1)
            self._run_ik_step(warmup_tip + disp, warmup_back + disp)
            mujoco.mj_step(self.model, self.data)

        self._initial_ee_pos = self.data.xpos[self.link6_id].copy() * 1000.0  # mm

    def get_ee_pose(self):
        pos = self.data.xpos[self.link6_id].copy() * 1000.0
        mat = self.data.xmat[self.link6_id].reshape(3, 3)
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        if sy > 1e-6:
            r = np.arctan2(mat[2, 1], mat[2, 2])
            p = np.arctan2(-mat[2, 0], sy)
            y = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            r = np.arctan2(-mat[1, 2], mat[1, 1])
            p = np.arctan2(-mat[2, 0], sy)
            y = 0.0
        return np.concatenate([pos, [r, p, y]])

    def get_ee_pos_mm(self):
        return self.data.xpos[self.link6_id].copy() * 1000.0

    def render_cameras(self):
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            self.renderer.update_scene(self.data, camera=cam_name)
            frames[cam_name] = self.renderer.render().copy()
        return frames

    def apply_delta_ee(self, delta_ee_6d, n_sim_steps=67, gain=0.5):
        current_ee = self.get_ee_pose()
        target_ee = current_ee + delta_ee_6d
        target_pos_m = target_ee[:3] / 1000.0
        target_rpy = target_ee[3:]

        for _ in range(n_sim_steps):
            cur_pos = self.data.xpos[self.link6_id].copy()
            cur_mat = self.data.xmat[self.link6_id].reshape(3, 3)

            err_pos = target_pos_m - cur_pos
            target_mat = self._rpy_to_rotmat(target_rpy)
            err_rot_mat = target_mat @ cur_mat.T
            err_rot = self._rotmat_to_axisangle(err_rot_mat)
            err = np.concatenate([err_pos * 50.0, err_rot * 10.0])

            jac_pos = np.zeros((3, self.dof))
            jac_rot = np.zeros((3, self.dof))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.link6_id)
            J = np.vstack([jac_pos[:, :self.n_motors], jac_rot[:, :self.n_motors]])

            dq = np.linalg.pinv(J, rcond=1e-4) @ err
            self.data.ctrl[:self.n_motors] = self.data.qpos[:self.n_motors] + dq * gain
            mujoco.mj_step(self.model, self.data)

    def get_sensor_dist(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        needle_dir = (tip_pos - back_pos)
        nd_len = np.linalg.norm(needle_dir)
        if nd_len > 1e-8:
            needle_dir /= nd_len
        dist = mujoco.mj_ray(
            self.model, self.data, tip_pos, needle_dir,
            None, 1, self.link6_id, np.zeros(1, dtype=np.int32),
        )
        return dist * 1000.0 if dist >= 0 else -1.0

    @staticmethod
    def _rpy_to_rotmat(rpy):
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _rotmat_to_axisangle(R):
        angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if abs(angle) < 1e-6:
            return np.zeros(3)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(angle))
        return axis * angle


def draw_basic_overlay(frame, direction_name, displacement_mm, cosine_sim, ctrl_step):
    """Draw direction + metrics overlay on replay frame."""
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Dir: {direction_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Step: {ctrl_step}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Disp: {displacement_mm:.1f}mm", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Cos: {cosine_sim:.3f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _save_summary_plot(eval_dir, results_per_direction):
    """Bar chart of per-direction success rate + cosine similarity."""
    directions = list(results_per_direction.keys())
    success_rates = []
    avg_cosines = []
    avg_displacements = []

    for d in directions:
        episodes = results_per_direction[d]
        n = len(episodes)
        sr = sum(1 for e in episodes if e["success"]) / n * 100 if n > 0 else 0
        avg_cos = np.mean([e["cosine_sim"] for e in episodes]) if n > 0 else 0
        avg_disp = np.mean([e["displacement_mm"] for e in episodes]) if n > 0 else 0
        success_rates.append(sr)
        avg_cosines.append(avg_cos)
        avg_displacements.append(avg_disp)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Basic Motion Eval Summary", fontsize=14)

    x = np.arange(len(directions))

    # Success rate
    colors_sr = ['green' if s >= 50 else 'orange' if s >= 25 else 'red' for s in success_rates]
    axes[0].bar(x, success_rates, color=colors_sr)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(directions, rotation=45, ha='right')
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate per Direction")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=8)

    # Avg cosine similarity
    colors_cos = ['green' if c >= 0.5 else 'orange' if c >= 0.0 else 'red' for c in avg_cosines]
    axes[1].bar(x, avg_cosines, color=colors_cos)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(directions, rotation=45, ha='right')
    axes[1].set_ylabel("Avg Cosine Similarity")
    axes[1].set_title("Direction Accuracy")
    axes[1].set_ylim(-1, 1.1)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Avg displacement
    axes[2].bar(x, avg_displacements, color='steelblue')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(directions, rotation=45, ha='right')
    axes[2].set_ylabel("Avg Displacement (mm)")
    axes[2].set_title("Movement Magnitude")

    plt.tight_layout()
    out_path = eval_dir / "eval_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main eval loop
# ═══════════════════════════════════════════════════════════════════════════════
def run_eval(cfg):
    checkpoint_path = cfg.eval.finetuned_checkpoint
    assert checkpoint_path, "eval.finetuned_checkpoint must be set!"

    # Shard support
    shard_id = getattr(cfg, "shard_id", None)
    num_shards = getattr(cfg, "num_shards", None)

    seed = cfg.eval.seed
    if shard_id is not None:
        seed = seed + shard_id * 1000
    set_seed(seed)

    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)
    model = load_model(checkpoint_path, diffusion_steps=diff_steps, scheduler_type=sched_type,
                       train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    image_size = getattr(cfg.eval, "image_size", 256)
    num_ep_per_dir = getattr(cfg.eval, "num_episodes_per_direction", 10)
    max_steps = getattr(cfg.eval, "max_steps_per_episode", 80)
    num_steps_execute = getattr(cfg.eval, "num_steps_execute", 1)
    sim_steps_per_ctrl = getattr(cfg.eval, "sim_steps_per_control", 67)
    save_video = getattr(cfg.eval, "save_video", True)
    video_fps = getattr(cfg.eval, "video_fps", 15)
    cosine_threshold = getattr(cfg.eval, "cosine_threshold", 0.5)
    min_displacement_mm = getattr(cfg.eval, "min_displacement_mm", 2.0)

    # Build direction+episode list
    all_tasks = []
    for dir_name in DIRECTIONS:
        for ep_idx in range(num_ep_per_dir):
            all_tasks.append((dir_name, ep_idx))

    if shard_id is not None and num_shards is not None:
        all_tasks = [t for i, t in enumerate(all_tasks) if i % num_shards == shard_id]
        shard_suffix = f"_shard{shard_id}"
    else:
        shard_suffix = ""

    # Output directory
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1]
    except (ValueError, IndexError):
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"basic_eval_step{step_str}_exec{num_steps_execute}_diff{diff_steps}{shard_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    model_xml = os.path.abspath(SIM_MODEL_PATH)
    env = BasicMotionSimEnv(model_xml)

    # CSV
    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "direction", "episode", "success", "steps",
        "cosine_sim", "displacement_mm",
        "disp_x_mm", "disp_y_mm", "disp_z_mm",
    ])

    results_per_direction = {d: [] for d in DIRECTIONS}
    total_successes = 0
    total_episodes = 0

    for task_idx, (dir_name, ep_idx) in enumerate(all_tasks):
        direction_vec = DIRECTIONS[dir_name]
        instruction = DIRECTION_INSTRUCTIONS[dir_name]

        env.reset(direction_vec)
        start_pos = env.get_ee_pos_mm().copy()

        image_history = []
        image_history_wrist = []
        image_history_top = []
        state_history = []
        action_history = []
        action_buffer = []
        replay_images = []

        for ctrl_step in range(max_steps):
            frames = env.render_cameras()
            img_ext = preprocess_image(frames["side_camera"], (image_size, image_size))
            img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))
            img_top = preprocess_image(frames["top_camera"], (image_size, image_size))

            # Primary view depends on train config (wrist for wrist model)
            img_primary = img_wrist

            image_history.append(img_primary)
            image_history_wrist.append(img_wrist)
            image_history_top.append(img_top)

            ee_pose = env.get_ee_pose()
            use_sensor = getattr(cfg.model, "use_sensor", False)
            if use_sensor:
                sensor_dist = env.get_sensor_dist()
                sensor_dist_clipped = min(sensor_dist, 20.0) if sensor_dist >= 0 else 20.0
                proprio = np.concatenate([ee_pose, [0.0], [sensor_dist_clipped]])
            else:
                proprio = np.concatenate([ee_pose, [0.0]])
            state_history.append(proprio)

            # Current displacement for overlay
            current_pos = env.get_ee_pos_mm()
            disp_vec = current_pos - start_pos
            disp_mag = np.linalg.norm(disp_vec)
            cos_sim = np.dot(disp_vec, direction_vec) / (disp_mag + 1e-8)

            replay_frame = np.concatenate([img_ext, img_wrist, img_top], axis=1)
            draw_basic_overlay(replay_frame, instruction, disp_mag, cos_sim, ctrl_step)
            replay_images.append(replay_frame)

            observation = {
                "full_image": img_primary,
                "full_image_wrist": img_wrist,
                "full_image_top": img_top,
                "image_history": image_history,
                "image_history_wrist": image_history_wrist,
                "image_history_top": image_history_top,
                "state_history": state_history,
                "action_history": action_history,
            }

            if len(action_buffer) == 0:
                raw_chunk, _ = predict_action(model, processor, observation, instruction)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk[None, :]
                steps_exec = min(num_steps_execute, len(raw_chunk))
                action_buffer = list(raw_chunk[:steps_exec])

            raw_action = action_buffer.pop(0)
            action_history.append(raw_action)

            a_min = np.array(action_min_sim, dtype=np.float32)
            a_max = np.array(action_max_sim, dtype=np.float32)
            denorm_action = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
            delta_ee = denorm_action[:6]

            env.apply_delta_ee(delta_ee, n_sim_steps=sim_steps_per_ctrl)

        # Episode done — compute final metrics
        final_pos = env.get_ee_pos_mm()
        displacement = final_pos - start_pos
        displacement_mm = np.linalg.norm(displacement)
        if displacement_mm > 1e-6:
            cosine_sim = np.dot(displacement, direction_vec) / displacement_mm
        else:
            cosine_sim = 0.0

        success = cosine_sim > cosine_threshold and displacement_mm > min_displacement_mm
        if success:
            total_successes += 1
        total_episodes += 1

        ep_result = {
            "success": success,
            "cosine_sim": cosine_sim,
            "displacement_mm": displacement_mm,
            "displacement": displacement.copy(),
        }
        results_per_direction[dir_name].append(ep_result)

        sr = total_successes / total_episodes * 100
        msg = (f"[{task_idx+1}/{len(all_tasks)}] {dir_name} ep{ep_idx} | "
               f"{'SUCCESS' if success else 'FAIL'} | "
               f"cos={cosine_sim:.3f} disp={displacement_mm:.1f}mm | "
               f"SR: {sr:.1f}% ({total_successes}/{total_episodes})")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        csv_writer.writerow([
            dir_name, ep_idx, int(success), max_steps,
            f"{cosine_sim:.4f}", f"{displacement_mm:.2f}",
            f"{displacement[0]:.2f}", f"{displacement[1]:.2f}", f"{displacement[2]:.2f}",
        ])
        csv_file.flush()

        if save_video:
            vid_dir = eval_dir / dir_name
            vid_dir.mkdir(exist_ok=True)
            tag = "S" if success else "F"
            vid_path = vid_dir / f"ep{ep_idx:03d}_{tag}.mp4"
            writer = imageio.get_writer(str(vid_path), fps=video_fps)
            for img in replay_images:
                writer.append_data(img)
            writer.close()

        # Save trajectory
        ee_traj = np.array([s[:3] for s in state_history])
        traj_dir = eval_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            traj_dir / f"traj_{dir_name}_ep{ep_idx:03d}_{'S' if success else 'F'}.npz",
            ee_pose=ee_traj,
            direction=direction_vec,
        )

    csv_file.close()

    # Per-direction summary
    print(f"\n{'='*70}")
    print(f"Per-Direction Results:")
    print(f"{'='*70}")
    dir_summary = {}
    for d in DIRECTIONS:
        episodes = results_per_direction[d]
        n = len(episodes)
        if n == 0:
            continue
        sr = sum(1 for e in episodes if e["success"]) / n * 100
        avg_cos = np.mean([e["cosine_sim"] for e in episodes])
        avg_disp = np.mean([e["displacement_mm"] for e in episodes])
        line = f"  {d:12s}: SR={sr:5.1f}%  cos={avg_cos:+.3f}  disp={avg_disp:.1f}mm  (n={n})"
        print(line)
        log_file = open(log_path, "a")
        log_file.write(line + "\n")
        log_file.close()
        dir_summary[d] = {"sr": sr, "avg_cos": avg_cos, "avg_disp": avg_disp}

    overall_sr = total_successes / total_episodes * 100 if total_episodes > 0 else 0
    summary = f"\nOverall Success Rate: {overall_sr:.2f}% ({total_successes}/{total_episodes})"
    print(summary)
    with open(log_path, "a") as f:
        f.write(summary + "\n")

    # Generate summary plot
    if shard_id is None:
        _save_summary_plot(eval_dir, results_per_direction)

    # Rename directory with SR (only if not a shard)
    if shard_id is None:
        new_dir = eval_dir.parent / f"{eval_dir.name}_SR{overall_sr:.2f}"
        try:
            eval_dir.rename(new_dir)
            print(f"Results saved to: {new_dir}")
        except Exception as e:
            print(f"Could not rename directory: {e}")
    else:
        print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLANeXt on basic motion task")
    parser.add_argument("--config", type=str, default="config/sim_eval_basic_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Override eval.finetuned_checkpoint")
    parser.add_argument("--train-config", type=str, default=None, help="Path to train config")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if args.checkpoint:
        config_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint

    cfg = DictConfig(config_dict)
    cfg.train_config_path = args.train_config
    cfg.shard_id = args.shard_id
    cfg.num_shards = args.num_shards

    run_eval(cfg)
