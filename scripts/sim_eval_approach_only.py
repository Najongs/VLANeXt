"""
sim_eval_approach_only.py

Evaluates a fine-tuned VLANeXt model on approach task.
(Random home pose → navigate to trocar vicinity, no insertion)

Usage:
    python -m scripts.sim_eval_approach_only \
        --config config/sim_eval_approach_config.yaml \
        --checkpoint /path/to/checkpoint \
        --train-config config/sim_train_align_config.yaml
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import argparse
import yaml
from omegaconf import OmegaConf
import random
import time
import pathlib

import mujoco
import numpy as np
import cv2
import torch
import imageio
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, SiglipImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.VLANeXt import VLANeXt, LlamaProcessorWrapper
from src.datasets.sim_act_approach import action_min_sim_approach as action_min_sim, action_max_sim_approach as action_max_sim

# Reuse model loading / inference from sim_eval
from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, save_episode_plot, draw_overlay,
    smooth_step, project_to_2d, set_seed,
    SIM_MODEL_PATH, IMG_WIDTH, IMG_HEIGHT,
)

import glob as _glob


def _save_trajectory_plot(eval_dir):
    """Load all saved trajectory npz files and plot 3D + 2D projections."""
    npz_files = sorted(_glob.glob(str(eval_dir / "traj_ep*.npz")))
    if not npz_files:
        return

    trajectories = []
    successes = []
    for f in npz_files:
        data = np.load(f)
        trajectories.append(data["ee_pose"])
        successes.append("_S." in f)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Eval Trajectories (n={len(trajectories)})", fontsize=14)

    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    for i, traj in enumerate(trajectories):
        color = 'green' if successes[i] else 'red'
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5, linewidth=0.8, color=color)
        ax1.scatter(*traj[0], marker='o', s=30, color=color, alpha=0.6)
        ax1.scatter(*traj[-1], marker='x', s=30, color=color, alpha=0.8)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D View')
    ax1.grid(True)

    # 2D projections
    proj_configs = [
        (222, 0, 1, 'X (mm)', 'Y (mm)', 'XY (Top View)'),
        (223, 0, 2, 'X (mm)', 'Z (mm)', 'XZ (Front View)'),
        (224, 1, 2, 'Y (mm)', 'Z (mm)', 'YZ (Side View)'),
    ]
    for subplot, ax_a, ax_b, xlabel, ylabel, title in proj_configs:
        ax = fig.add_subplot(subplot)
        for i, traj in enumerate(trajectories):
            color = 'green' if successes[i] else 'red'
            ax.plot(traj[:, ax_a], traj[:, ax_b], alpha=0.5, linewidth=0.8, color=color)
            ax.scatter(traj[0, ax_a], traj[0, ax_b], marker='o', s=20, color=color, alpha=0.6)
            ax.scatter(traj[-1, ax_a], traj[-1, ax_b], marker='x', s=20, color=color, alpha=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True); ax.axis('equal')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', label='Success'),
        Line2D([0], [0], color='red', label='Fail'),
        Line2D([0], [0], marker='o', color='gray', label='Start', linestyle='None', markersize=6),
        Line2D([0], [0], marker='x', color='gray', label='End', linestyle='None', markersize=6),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = eval_dir / "eval_trajectories.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Approach eval config
# ═══════════════════════════════════════════════════════════════════════════════
TASK_INSTRUCTION = "Approach the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"

# Home pose joint ranges (same as Save_dataset.py)
HOME_POSE_RANGES = [
    (-0.5, 0.5),   # joint 0
    (-0.6, -0.4),    # joint 1
    (0.75, 0.25),      # joint 2
    (-0.3, 0.3),      # joint 3 (fixed)
    (0.4, 0.6),      # joint 4
    (0.9, 1.1),      # joint 5
]

# Success: needle tip within distance of trocar entry
APPROACH_SUCCESS_THRESHOLD_M = 0.015   # 15mm
APPROACH_SUCCESS_ANGLE_DEG = 10.0      # needle-trocar axis angle < 10deg
APPROACH_SUCCESS_HOLD_STEPS = 10       # consecutive steps within threshold


class ApproachSimEnv:
    """MuJoCo env for approach evaluation.

    Reset:
      1. Optionally randomize phantom position
      2. Set robot to random home pose (far from trocar)
    Success: needle tip within threshold of trocar entry
    """

    def __init__(self, model_xml_path: str, randomize_phantom: bool = False, phantom_pos: tuple = None):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMG_HEIGHT, width=IMG_WIDTH)

        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        self.back_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        self.target_entry_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        self.target_depth_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        self.link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        self.n_motors = self.model.nu
        self.dof = self.model.nv

        # Phantom randomization
        self.randomize_phantom = randomize_phantom
        self.phantom_pos = phantom_pos
        self._phantom_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        self._rotating_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")

        self.approach_hold_counter = 0
        self.last_phantom_info = None

    def _randomize_phantom(self):
        """Randomize phantom position and rotation (same logic as Save_dataset.py)."""
        offset_x = np.random.uniform(-0.05, 0.05)
        # Y=-0.26~-0.17 제외 (회전 전환 경계, IK 실패 다발 구간)
        if np.random.random() < 0.6:
            offset_y = np.random.uniform(-0.4, -0.26)
        else:
            offset_y = np.random.uniform(-0.17, 0.0)
        offset_z = 0.0
        self.model.body_pos[self._phantom_body_id] = np.array([offset_x, offset_y, offset_z])

        if offset_y >= -0.25:
            random_angle_deg = np.random.uniform(-15, 15)
        else:
            random_angle_deg = np.random.uniform(-15 - 90, 15 - 90)

        new_quat = np.zeros(4)
        mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
        self.model.body_quat[self._rotating_id] = new_quat
        mujoco.mj_forward(self.model, self.data)

        self.last_phantom_info = {
            "phantom_x": offset_x,
            "phantom_y": offset_y,
            "phantom_angle_deg": random_angle_deg,
        }
        print(f"  Phantom: pos=({offset_x:.3f}, {offset_y:.3f}), angle={random_angle_deg:.1f}deg")

    def _set_fixed_phantom(self, pos):
        """Set phantom to a fixed (x, y) position with fixed rotation."""
        px, py = pos
        self.model.body_pos[self._phantom_body_id] = np.array([px, py, 0.0])

        if py >= -0.25:
            random_angle_deg = 0
        else:
            random_angle_deg = -90

        new_quat = np.zeros(4)
        mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
        self.model.body_quat[self._rotating_id] = new_quat
        mujoco.mj_forward(self.model, self.data)

        self.last_phantom_info = {
            "phantom_x": px,
            "phantom_y": py,
            "phantom_angle_deg": random_angle_deg,
        }
        print(f"  Phantom (fixed): pos=({px:.3f}, {py:.3f}), angle={random_angle_deg:.1f}deg")

    def reset(self):
        """Reset to random home pose (far from trocar)."""
        mujoco.mj_resetData(self.model, self.data)

        if self.phantom_pos is not None:
            self._set_fixed_phantom(self.phantom_pos)
        elif self.randomize_phantom:
            self._randomize_phantom()

        # Random home pose (same ranges as Save_dataset.py)
        home_pose = np.array([
            np.random.uniform(*HOME_POSE_RANGES[i]) for i in range(6)
        ])
        self.data.qpos[:6] = home_pose
        mujoco.mj_forward(self.model, self.data)

        # Let physics settle
        for _ in range(100):
            self.data.ctrl[:self.n_motors] = self.data.qpos[:self.n_motors]
            mujoco.mj_step(self.model, self.data)

        self.approach_hold_counter = 0

        initial_dist = np.linalg.norm(
            self.data.site_xpos[self.tip_id] - self.data.site_xpos[self.target_entry_id]
        ) * 1000.0

        self.last_start_info = {
            "home_pose": home_pose.tolist(),
            "initial_dist_mm": initial_dist,
        }
        print(f"  Home pose: [{', '.join(f'{v:.3f}' for v in home_pose)}], initial_dist={initial_dist:.1f}mm")

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

    def check_success(self):
        """Check if needle tip is close enough to trocar entry."""
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()

        dist = np.linalg.norm(tip_pos - entry_pos)

        # Needle-trocar axis angle
        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        axis_dir = depth_pos - entry_pos
        axis_len = np.linalg.norm(axis_dir)
        if needle_len > 1e-8 and axis_len > 1e-8:
            cos_angle = abs(np.dot(needle_dir / needle_len, axis_dir / axis_len))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
        else:
            angle_deg = 90.0

        if dist < APPROACH_SUCCESS_THRESHOLD_M and angle_deg < APPROACH_SUCCESS_ANGLE_DEG:
            self.approach_hold_counter += 1
        else:
            self.approach_hold_counter = 0

        return self.approach_hold_counter >= APPROACH_SUCCESS_HOLD_STEPS

    def get_alignment_dist_mm(self):
        """Distance from needle tip to trocar entry in mm."""
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        return np.linalg.norm(tip_pos - entry_pos) * 1000.0

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

    def get_spatial_metrics(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()

        dist_mm = np.linalg.norm((entry_pos - tip_pos) * 1000.0)

        axis = depth_pos - entry_pos
        axis_dir = axis / (np.linalg.norm(axis) + 1e-10)

        tip_offset = tip_pos - entry_pos
        insertion_depth_mm = np.dot(tip_offset, axis_dir) * 1000.0
        projection = tip_offset - np.dot(tip_offset, axis_dir) * axis_dir
        lateral_mm = np.linalg.norm(projection) * 1000.0

        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        if needle_len > 1e-8:
            needle_dir /= needle_len
            cos_angle = abs(np.dot(needle_dir, axis_dir))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
        else:
            angle_deg = 90.0

        tip_uv = project_to_2d(tip_pos, self.model, self.data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
        trocar_uv = project_to_2d(entry_pos, self.model, self.data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)

        return {
            "dist_mm": dist_mm,
            "insertion_depth_mm": insertion_depth_mm,
            "lateral_mm": lateral_mm,
            "angle_deg": angle_deg,
            "tip_uv": tip_uv,
            "trocar_uv": trocar_uv,
        }

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main eval loop
# ═══════════════════════════════════════════════════════════════════════════════
def run_eval(cfg):
    checkpoint_path = cfg.eval.finetuned_checkpoint
    assert checkpoint_path, "eval.finetuned_checkpoint must be set!"

    # Shard support for parallel eval
    shard_id = getattr(cfg, "shard_id", None)
    num_shards = getattr(cfg, "num_shards", None)

    seed = cfg.eval.seed
    if shard_id is not None:
        seed = seed + shard_id * 1000
    set_seed(seed)

    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)
    model = load_model(checkpoint_path, diffusion_steps=diff_steps, scheduler_type=sched_type, train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    use_sensor = getattr(cfg.model, "use_sensor", False)
    image_size = getattr(cfg.eval, "image_size", 256)
    num_episodes = getattr(cfg.eval, "num_episodes", 50)
    max_steps = getattr(cfg.eval, "max_steps_per_episode", 500)
    num_steps_execute = getattr(cfg.eval, "num_steps_execute", 1)
    sim_steps_per_ctrl = getattr(cfg.eval, "sim_steps_per_control", 67)
    save_video = getattr(cfg.eval, "save_video", True)
    video_fps = getattr(cfg.eval, "video_fps", 15)

    # Build episode list
    all_episodes = list(range(1, num_episodes + 1))
    if shard_id is not None and num_shards is not None:
        all_episodes = [ep for ep in all_episodes if (ep - 1) % num_shards == shard_id]
        shard_suffix = f"_shard{shard_id}"
    else:
        shard_suffix = ""

    # Output directory
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1]
    except (ValueError, IndexError):
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"approach_eval_step{step_str}_exec{num_steps_execute}_diff{diff_steps}{shard_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    model_xml = os.path.abspath(SIM_MODEL_PATH)
    randomize_phantom = getattr(cfg, "randomize_phantom", False)
    phantom_pos = getattr(cfg, "phantom_pos", None)
    env = ApproachSimEnv(model_xml, randomize_phantom=randomize_phantom, phantom_pos=phantom_pos)

    total_successes = 0

    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = ["episode", "success", "steps", "final_dist_mm",
                   "final_lateral_mm", "final_angle_deg", "min_dist_mm",
                   "initial_dist_mm"]
    if randomize_phantom or phantom_pos is not None:
        csv_header.extend(["phantom_x", "phantom_y", "phantom_angle_deg"])
    csv_writer.writerow(csv_header)

    for ep in all_episodes:
        env.reset()

        image_history = []
        image_history_wrist = []
        image_history_top = []
        state_history = []
        action_history = []
        action_buffer = []
        replay_images = []
        metrics_history = []

        success = False

        for ctrl_step in range(max_steps):
            frames = env.render_cameras()
            img_ext = preprocess_image(frames["side_camera"], (image_size, image_size))
            img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))
            img_top = preprocess_image(frames["top_camera"], (image_size, image_size))

            # Map sim cameras to train config camera roles
            # Train config: cam_exterior=top_camera, cam_wrist=tool_camera, cam_top=""
            img_primary = img_top       # exterior = top_camera
            img_secondary = img_wrist   # wrist = tool_camera

            image_history.append(img_primary)
            image_history_wrist.append(img_secondary)
            image_history_top.append(img_top)  # kept for replay only

            metrics = env.get_spatial_metrics()
            metrics_history.append(metrics)

            replay_frame = np.concatenate([img_ext, img_wrist, img_top], axis=1)

            ee_pose = env.get_ee_pose()
            proprio_parts = [ee_pose, [0.0]]  # ee_pose(6) + gripper(1) = 7
            if use_sensor:
                sensor_dist = env.get_sensor_dist()
                sensor_dist_clipped = min(sensor_dist, 20.0) if sensor_dist >= 0 else 20.0
                proprio_parts.append([sensor_dist_clipped])
            proprio = np.concatenate(proprio_parts)
            state_history.append(proprio)

            observation = {
                "full_image": img_primary,
                "full_image_wrist": img_secondary,
                "image_history": image_history,
                "image_history_wrist": image_history_wrist,
                "state_history": state_history,
                "action_history": action_history,
            }
            # Note: full_image_top / image_history_top intentionally excluded
            # because training used cam_top="" (only 2-view: top + tool)

            spatial_pred = None
            if len(action_buffer) == 0:
                raw_chunk, spatial_pred = predict_action(model, processor, observation, TASK_INSTRUCTION)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk[None, :]
                steps_exec = min(num_steps_execute, len(raw_chunk))
                action_buffer = list(raw_chunk[:steps_exec])
            if spatial_pred is not None:
                metrics["spatial_pred"] = spatial_pred

            draw_overlay(replay_frame, metrics, ctrl_step)
            replay_images.append(replay_frame)

            raw_action = action_buffer.pop(0)
            action_history.append(raw_action)

            a_min = np.array(action_min_sim, dtype=np.float32)
            a_max = np.array(action_max_sim, dtype=np.float32)
            denorm_action = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
            delta_ee = denorm_action[:6]

            env.apply_delta_ee(delta_ee, n_sim_steps=sim_steps_per_ctrl)

            if env.check_success():
                success = True
                metrics_history.append(env.get_spatial_metrics())
                break

        if success:
            total_successes += 1

        ep_done = all_episodes.index(ep) + 1
        sr = total_successes / ep_done * 100
        final_m = metrics_history[-1]
        min_dist = min(m["dist_mm"] for m in metrics_history)
        msg = (f"Episode {ep}/{num_episodes} | {'SUCCESS' if success else 'FAIL'} | "
               f"Steps: {ctrl_step + 1} | SR: {sr:.1f}% ({total_successes}/{ep_done}) | "
               f"dist={final_m['dist_mm']:.1f}mm lateral={final_m['lateral_mm']:.1f}mm "
               f"angle={final_m['angle_deg']:.1f}deg min_dist={min_dist:.1f}mm")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        si = env.last_start_info
        row = [
            ep, int(success), ctrl_step + 1,
            f"{final_m['dist_mm']:.2f}",
            f"{final_m['lateral_mm']:.2f}", f"{final_m['angle_deg']:.2f}",
            f"{min_dist:.2f}",
            f"{si['initial_dist_mm']:.2f}",
        ]
        if (randomize_phantom or phantom_pos is not None) and env.last_phantom_info:
            ph = env.last_phantom_info
            row.extend([f"{ph['phantom_x']:.4f}", f"{ph['phantom_y']:.4f}", f"{ph['phantom_angle_deg']:.1f}"])
        csv_writer.writerow(row)
        csv_file.flush()

        save_episode_plot(metrics_history, ep, success, str(eval_dir))

        if save_video:
            save_rollout_video(replay_images, ep, success, str(eval_dir), fps=video_fps)

        # Save trajectory (ee_pose only, without gripper/sensor)
        ee_traj = np.array([s[:6] for s in state_history])  # (T, 6): xyz + rpy
        np.savez_compressed(
            eval_dir / f"traj_ep{ep:03d}_{'S' if success else 'F'}.npz",
            ee_pose=ee_traj[:, :3],  # position (mm)
        )

    csv_file.close()

    # Generate trajectory visualization
    _save_trajectory_plot(eval_dir)

    final_sr = total_successes / len(all_episodes) * 100
    summary = f"\n{'='*60}\nFinal Approach Success Rate: {final_sr:.2f}% ({total_successes}/{len(all_episodes)})\n{'='*60}"
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()

    # Only rename directory when NOT running as a shard (merge handles final naming)
    if shard_id is None:
        new_dir = eval_dir.parent / f"{eval_dir.name}_SR{final_sr:.2f}"
        try:
            eval_dir.rename(new_dir)
            print(f"Results saved to: {new_dir}")
        except Exception as e:
            print(f"Could not rename directory: {e}")
    else:
        print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLANeXt on approach task")
    parser.add_argument("--config", type=str, default="config/sim_eval_align_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Override eval.finetuned_checkpoint")
    parser.add_argument("--train-config", type=str, default=None, help="Path to train config")
    parser.add_argument("--shard-id", type=int, default=None, help="Shard index for parallel eval (0-based)")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--randomize-phantom", action="store_true",
                        help="Randomize phantom position/rotation each episode")
    parser.add_argument("--phantom-pos", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Fixed phantom position (x, y). e.g. --phantom-pos 0.0 -0.4")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if args.checkpoint:
        config_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint

    cfg = DictConfig(config_dict)
    cfg.train_config_path = args.train_config
    cfg.shard_id = args.shard_id
    cfg.num_shards = args.num_shards
    cfg.randomize_phantom = args.randomize_phantom
    cfg.phantom_pos = tuple(args.phantom_pos) if args.phantom_pos is not None else None
    run_eval(cfg)
