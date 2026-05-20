"""
sim_eval_align_only.py

Evaluates a fine-tuned VLANeXt model on fine-alignment task.
(No insertion — only alignment from perturbed state near trocar)

Usage:
    python -m scripts.sim_eval_align_only \
        --config config/sim_eval_align_config.yaml \
        --checkpoint /path/to/checkpoint \
        --train-config config/sim_train_spatial_config.yaml
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

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
# Heavy imports — lazy so lerobot bridges can reuse AlignSimEnv/grid utilities
# without pulling VLANeXt's transformers/peft/qwen-vl-utils dependency tree.
try:
    from transformers import AutoProcessor, AutoTokenizer, SiglipImageProcessor
except ImportError:
    AutoProcessor = AutoTokenizer = SiglipImageProcessor = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.models.VLANeXt import VLANeXt, LlamaProcessorWrapper
except ImportError:
    VLANeXt = LlamaProcessorWrapper = None
from src.datasets.sim_act_align import action_min_sim_align as action_min_sim, action_max_sim_align as action_max_sim
from src.utils.sensor_proc import process_sensor_dist_scalar
from src.datasets.euler_convention import (
    mujoco_to_mecademic_euler,
    mecademic_to_mujoco_euler,
)

# Reuse model loading / inference from sim_eval
from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, save_episode_plot, draw_overlay,
    smooth_step, project_to_2d, set_seed,
    SIM_MODEL_PATH, IMG_WIDTH, IMG_HEIGHT,
)

import glob as _glob
import torch.nn as _nn


class _KPHeadUV(_nn.Module):
    def __init__(self, h):
        super().__init__()
        self.mlp = _nn.Sequential(
            _nn.Linear(h, 512), _nn.GELU(), _nn.Dropout(0.1),
            _nn.Linear(512, 256), _nn.GELU(), _nn.Dropout(0.1),
            _nn.Linear(256, 2),
        )
    def forward(self, x): return torch.sigmoid(self.mlp(x.mean(dim=1)))


class _KPHeadDist(_nn.Module):
    def __init__(self, h):
        super().__init__()
        self.mlp = _nn.Sequential(
            _nn.Linear(h, 512), _nn.GELU(), _nn.Dropout(0.1),
            _nn.Linear(512, 256), _nn.GELU(), _nn.Dropout(0.1),
            _nn.Linear(256, 1),
        )
    def forward(self, x): return self.mlp(x.mean(dim=1)).squeeze(-1)


class KeypointInferencer:
    """Load frozen SigLIP2 + uv head + dist head. Predict (troc_u, troc_v, dist_norm) from tool_camera frame.

    Used for VLA proprio inference when proprio_dim == 9 (ee_pose 6 + uv 2 + dist_norm 1).

    Projection bias correction: HDF5 GT projection has systematic offset vs visual feature
    center (measured 2026-05-14 — sim 0/0, real -5/+10 px). Apply via `domain` arg.
    """
    DIST_NORM = 50.0  # mm — must match dataset / training
    VISION_MODEL = "google/siglip2-so400m-patch16-512"
    # Per-domain offset (du, dv) in pixels — added to predicted UV to align with visual feature.
    # Source: project_keypoint_projection_bias memory.
    PROJECTION_OFFSET_PX = {"sim": (0.0, 0.0), "real": (-5.0, 10.0), "none": (0.0, 0.0)}

    def __init__(self, uv_ckpt_path, dist_ckpt_path, device="cuda", domain="sim"):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        self.device = device
        self.proc = SiglipImageProcessor.from_pretrained(self.VISION_MODEL)
        self.vm = SiglipVisionModel.from_pretrained(self.VISION_MODEL, dtype=torch.bfloat16).to(device).eval()
        for p in self.vm.parameters(): p.requires_grad_(False)
        hidden = self.vm.config.hidden_size
        self.uv_head = _KPHeadUV(hidden).to(device).to(torch.bfloat16).eval()
        self.dist_head = _KPHeadDist(hidden).to(device).to(torch.bfloat16).eval()
        self.uv_head.load_state_dict(torch.load(uv_ckpt_path, map_location=device)["head_state"])
        self.dist_head.load_state_dict(torch.load(dist_ckpt_path, map_location=device)["head_state"])
        if domain not in self.PROJECTION_OFFSET_PX:
            raise ValueError(f"domain must be one of {list(self.PROJECTION_OFFSET_PX)}, got {domain}")
        self.domain = domain
        off = self.PROJECTION_OFFSET_PX[domain]
        self._uv_offset_norm = np.array([off[0] / 256.0, off[1] / 256.0], dtype=np.float32)
        print(f"[KeypointInferencer] loaded uv={uv_ckpt_path} dist={dist_ckpt_path} domain={domain} uv_offset_px={off}")

    @torch.no_grad()
    def predict(self, img_uint8_HW3):
        """img: uint8 HxWx3 numpy. Returns (troc_u, troc_v, dist_norm).
        Applies domain-specific UV offset to align prediction with visual feature center.
        IMPORTANT: input is resized to 256x256 (LANCZOS) to match training distribution
        (HDF5 stored 256x256 frames). Live MuJoCo renders (640x480) MUST be resized first
        — direct pass causes distribution mismatch (processor's internal resize differs).
        """
        from PIL import Image as _Image
        pil = _Image.fromarray(img_uint8_HW3).convert("RGB")
        if pil.size != (256, 256):
            pil = pil.resize((256, 256), _Image.LANCZOS)
        pv = self.proc(images=pil, return_tensors="pt")["pixel_values"].to(self.device, dtype=torch.bfloat16)
        feats = self.vm(pixel_values=pv).last_hidden_state
        uv = self.uv_head(feats).float().cpu().numpy()[0]
        dnorm = float(self.dist_head(feats).float().cpu().numpy()[0])
        # Apply projection-bias correction: predictions match GT distribution (biased),
        # add domain offset to get visually-accurate UV.
        uv_corrected = uv + self._uv_offset_norm
        return float(uv_corrected[0]), float(uv_corrected[1]), max(0.0, min(dnorm, 2.0))

    def predict_world_seed(self, img_uint8_HW3, cam_pos, cam_mat, fovy_deg=58.0,
                            tip_uv=(0.48789975, 0.32647642)):
        """Predict a world-frame XYZ lateral offset that moves tip toward predicted trocar.

        Used to SEED handoff grid search — instead of blind grid around current tip,
        first apply this delta, then refine via sensor.

        Returns: (world_delta_mm 3-vec, predicted_dist_mm, predicted_lateral_norm)
        """
        tu, tv, dnorm = self.predict(img_uint8_HW3)
        du_img = tu - tip_uv[0]
        dv_img = tv - tip_uv[1]
        dist_mm = dnorm * self.DIST_NORM
        # mm per normalized image unit at the predicted distance.
        # Vertical FOV; image is square so same for u.
        mm_per_norm = 2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0) * dist_mm
        # MuJoCo project_to_2d: u = -f*(px/pz) → px (cam x) flipped in image.
        # So moving robot in cam +x = trocar moves in image -u.
        # To bring trocar (at troc_uv) onto tip (at tip_uv): need image content shift
        # by -(du_img, dv_img). Camera moves by +(cam_x_offset, cam_y_offset) accordingly:
        cam_x = -du_img * mm_per_norm
        cam_y =  dv_img * mm_per_norm
        cam_offset_mm = np.array([cam_x, cam_y, 0.0], dtype=np.float64)
        # Camera frame → world frame. cam_mat columns are camera basis in world.
        world_delta_mm = (cam_mat @ cam_offset_mm).astype(np.float32)
        lateral_norm = float(np.hypot(du_img, dv_img))
        return world_delta_mm, dist_mm, lateral_norm


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
# Fine-alignment eval config
# ═══════════════════════════════════════════════════════════════════════════════
TASK_INSTRUCTION = "Align the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"

# Phantom variation grid (matches training data ranges; mm)
# X(-25~25), Y(-25~75), Z(0e~50), angle ±25°
PERTURB_POS_XY_MM = 25.0     # Note: Y range is asymmetric (-25~75), see build_perturb_grid
PERTURB_POS_Y_MIN_MM = -25.0
PERTURB_POS_Y_MAX_MM = 75.0
PERTURB_POS_Z_MIN_MM = 0.0
PERTURB_POS_Z_MAX_MM = 25.0
PERTURB_ANGLE_DEG = 25.0

# Success: needle tip within distance + angle threshold
ALIGN_SUCCESS_THRESHOLD_M = 0.005  # 5mm
ALIGN_SUCCESS_ANGLE_DEG = 10.0      # needle-trocar axis angle < 10deg
ALIGN_SUCCESS_HOLD_STEPS = 20       # consecutive steps within threshold (stable hold, was 5)
ALIGN_SUCCESS_SENSOR_MIN_MM = 20.0   # snsor must see through hole (> this value)

# --- Sensor-stop trigger (independent of distance criterion) ---
# When --sensor-stop is on, episode terminates EARLY with success=True after
# a "approach → cross hole" pattern is detected:
#   1. The needle must have been close to the trocar surface
#      (raw sensor ≤ SENSOR_STOP_CLOSE_MM) at SOME prior step
#   2. After that, raw sensor must jump to ≥ SENSOR_STOP_HOLE_MM and STAY there
#      for SENSOR_STOP_HOLD_STEPS consecutive control steps
#
# Why both conditions: during early approach a far reading (e.g. 8-15mm to
# table or off-axis surface) can spuriously satisfy a single "raw ≥ 8mm"
# threshold even though the needle is nowhere near the hole. The state
# machine ensures we only trigger on the genuine "approached → punched
# through" transition, which is what hole-through really looks like.
SENSOR_STOP_CLOSE_MM = 5.0   # must dip below this once to "arm" the trigger
SENSOR_STOP_HOLE_MM = 15.0   # then jump above this to fire (hole reads ~20mm)
SENSOR_STOP_HOLD_STEPS = 2   # how many consecutive frames to confirm the spike


def build_perturb_grid(xy_steps, z_steps, angle_steps, repeats,
                       x_range=PERTURB_POS_XY_MM,
                       y_range=(PERTURB_POS_Y_MIN_MM, PERTURB_POS_Y_MAX_MM),
                       z_range=(PERTURB_POS_Z_MIN_MM, PERTURB_POS_Z_MAX_MM),
                       angle_range=PERTURB_ANGLE_DEG,
                       x_steps=None, y_steps=None):
    """Phantom-grid: enumerate phantom (x, y, z, angle) cells × repeats. Values in mm/deg.
    x_steps/y_steps override xy_steps when provided (for asymmetric grids)."""
    xs_n = x_steps if x_steps is not None else xy_steps
    ys_n = y_steps if y_steps is not None else xy_steps
    if xs_n >= 2:
        xs = np.linspace(-x_range, x_range, xs_n)
    else:
        xs = np.array([0.0])
    if ys_n >= 2:
        ys = np.linspace(y_range[0], y_range[1], ys_n)
    else:
        ys = np.array([(y_range[0] + y_range[1]) / 2.0])
    if z_steps >= 2:
        zs = np.linspace(z_range[0], z_range[1], z_steps)
    elif z_steps == 1:
        zs = np.array([(z_range[0] + z_range[1]) / 2.0])
    else:
        zs = np.array([z_range[0]])
    if angle_steps >= 2:
        angles = np.linspace(-angle_range, angle_range, angle_steps)
    else:
        angles = np.array([0.0])

    cells = []
    cell_idx = 0
    for x in xs:
        for y in ys:
            for z in zs:
                for ang in angles:
                    for r in range(max(1, repeats)):
                        cells.append({
                            "cell_idx": cell_idx,
                            "repeat_id": r,
                            "x_mm": float(x),
                            "y_mm": float(y),
                            "z_mm": float(z),
                            "angle_deg": float(ang),
                        })
                    cell_idx += 1
    return cells


class AlignSimEnv:
    """MuJoCo env for fine-alignment evaluation.

    Reset:
      1. Pre-align needle to trocar (IK, cached per phantom position)
      2. Apply random perturbation
    Success: needle tip within threshold of trocar entry
    """

    def __init__(self, model_xml_path: str, randomize_phantom: bool = False, use_sensor_success: bool = False, phantom_pos: tuple = None, retreat_mm: float = 10.0):
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

        # Phantom randomization / fixed position
        self.randomize_phantom = randomize_phantom
        self.phantom_pos = phantom_pos
        self.use_sensor_success = use_sensor_success
        self.retreat_mm = retreat_mm
        # New XML structure: trocar_assembly (X,Y, rotation parent), phantom_assembly (Z lift)
        self._phantom_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trocar_assembly")
        self._phantom_assembly_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        self._rotating_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        self._phantom_base_pos = self.model.body_pos[self._phantom_body_id].copy() if self._phantom_body_id >= 0 else np.zeros(3)
        self._phantom_assembly_base_pos = self.model.body_pos[self._phantom_assembly_id].copy() if self._phantom_assembly_id >= 0 else np.zeros(3)

        # tool_camera ID (occlusion check용)
        self._tool_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "tool_camera")

        # Z 방향 균등 배분 카운터
        self._z_dir_counts = {"pos": 0, "neg": 0}

        # Cached aligned state
        self._aligned_qpos = None
        self._aligned_qvel = None
        self._goal_tip = None
        self._goal_back = None
        self._p_entry = None
        self._p_depth = None

        self.align_hold_counter = 0
        self.last_phantom_info = None

    def _run_ik_step(self, target_tip_pos, target_back_pos, speed=0.5):
        """One IK step (same as data collection)."""
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

    def _check_tip_occluded(self):
        """tool_camera → needle_tip ray cast로 팬텀에 가려지는지 확인."""
        cam_pos = self.data.cam_xpos[self._tool_cam_id].copy()
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        direction = tip_pos - cam_pos
        dist_to_tip = np.linalg.norm(direction)
        direction_norm = direction / (dist_to_tip + 1e-10)
        geomid_out = np.zeros(1, dtype=np.int32)
        hit_dist = mujoco.mj_ray(self.model, self.data, cam_pos, direction_norm,
                                  None, 1, -1, geomid_out)
        if hit_dist > 0 and hit_dist < dist_to_tip - 0.001:
            return True
        return False

    def _apply_phantom(self, px, py, pz, angle_deg):
        """Apply phantom params (matches training-time logic).
        X,Y → trocar_assembly offset; Z → phantom_assembly offset; angle → rotating_assembly quat.
        """
        self.model.body_pos[self._phantom_body_id] = self._phantom_base_pos + np.array([px, py, 0.0])
        self.model.body_pos[self._phantom_assembly_id] = self._phantom_assembly_base_pos + np.array([0.0, 0.0, pz])
        new_quat = np.zeros(4)
        mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(angle_deg)], "xyz")
        self.model.body_quat[self._rotating_id] = new_quat
        mujoco.mj_forward(self.model, self.data)
        self.last_phantom_info = {
            "phantom_x": float(px),
            "phantom_y": float(py),
            "phantom_z": float(pz),
            "phantom_angle_deg": float(angle_deg),
        }

    def _randomize_phantom(self):
        """Randomize phantom (matches Save_dataset_approach_only.py)."""
        offset_x = np.random.uniform(-0.025, 0.025)
        offset_y = np.random.uniform(-0.025, 0.075)
        offset_z = np.random.uniform(0.0, 0.05)  # optical_plate + trocar 통째로 상승
        random_angle_deg = float(np.random.uniform(-25, 25))
        self._apply_phantom(offset_x, offset_y, offset_z, random_angle_deg)
        print(f"  Phantom: pos=({offset_x:.3f}, {offset_y:.3f}, Z+{offset_z:.3f}), angle={random_angle_deg:.1f}deg")

    def _set_fixed_phantom(self, pos, pz=0.0, angle_deg=None):
        """Set phantom to fixed (x, y[, z[, angle]]); angle random ±25° if not given."""
        px, py = pos[0], pos[1]
        if angle_deg is None:
            angle_deg = float(np.random.uniform(-25, 25))
        self._apply_phantom(px, py, pz, angle_deg)
        print(f"  Phantom (fixed): pos=({px:.3f}, {py:.3f}, Z+{pz:.3f}), angle={angle_deg:.1f}deg")

    def _ensure_aligned_state(self):
        """Pre-align and cache. Re-runs if phantom is randomized or fixed with random rotation."""
        if self._aligned_qpos is not None and not self.randomize_phantom:
            return

        label = "Re-aligning for new phantom..." if self._aligned_qpos is not None else "Running initial pre-alignment..."
        print(label)
        mujoco.mj_resetData(self.model, self.data)
        home_pose = np.array([0.75, -0.5, 0.5, 0.0, 0.6, 1.0])
        self.data.qpos[:6] = home_pose
        mujoco.mj_forward(self.model, self.data)

        if self.phantom_pos is not None:
            self._set_fixed_phantom(self.phantom_pos)
        elif self.randomize_phantom:
            self._randomize_phantom()

        p_entry = self.data.site_xpos[self.target_entry_id].copy()
        p_depth = self.data.site_xpos[self.target_depth_id].copy()
        curr_tip = self.data.site_xpos[self.tip_id].copy()
        curr_back = self.data.site_xpos[self.back_id].copy()
        needle_len = np.linalg.norm(curr_tip - curr_back)

        axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
        retreat_m = self.retreat_mm / 1000.0
        goal_tip = p_entry - (axis_dir * retreat_m)
        goal_back = p_entry - (axis_dir * (retreat_m + needle_len))

        start_tip = curr_tip.copy()
        start_back = curr_back.copy()
        align_dist = np.linalg.norm(goal_tip - start_tip)
        duration = align_dist / 0.1  # fast pre-alignment
        t_start = self.data.time
        timer = 0

        while True:
            progress = smooth_step((self.data.time - t_start) / duration) if duration > 0 else 1.0
            t_tip = (1 - progress) * start_tip + progress * goal_tip
            t_back = (1 - progress) * start_back + progress * goal_back
            self._run_ik_step(t_tip, t_back)
            mujoco.mj_step(self.model, self.data)

            if progress >= 1.0:
                if np.linalg.norm(self.data.site_xpos[self.tip_id] - goal_tip) < 0.002:
                    timer += 1
                else:
                    timer = 0
                if timer > 20:
                    break
            if self.data.time - t_start > 50.0:
                raise RuntimeError("Pre-alignment failed!")

        self._aligned_qpos = self.data.qpos[:self.n_motors].copy()
        self._aligned_qvel = self.data.qvel[:self.n_motors].copy()
        self._goal_tip = goal_tip
        self._goal_back = goal_back
        self._p_entry = p_entry
        self._p_depth = p_depth
        print("Pre-alignment cached.")

    def reset(self, max_retries=10, grid_cell=None):
        """Reset env. Two modes:
        - grid_cell: phantom (x,y,z,angle) varies per cell; robot starts at fixed home_pose (NO pre-align, NO robot perturb).
        - else: legacy random — pre-align then perturb robot pose.
        """
        # ============ NEW: phantom-grid mode ============
        if grid_cell is not None:
            # Set phantom from cell (mm → m)
            px = grid_cell["x_mm"] / 1000.0
            py = grid_cell["y_mm"] / 1000.0
            pz = grid_cell["z_mm"] / 1000.0
            ang = grid_cell["angle_deg"]
            mujoco.mj_resetData(self.model, self.data)
            self._apply_phantom(px, py, pz, ang)
            # Robot: fixed home_pose (matches training)
            home_pose = np.array([0.75, -0.5, 0.5, 0, 0.6, 1.0])
            self.data.qpos[:6] = home_pose
            self.data.qvel[:self.n_motors] = 0.0
            self.data.ctrl[:self.n_motors] = home_pose[:self.n_motors]
            mujoco.mj_forward(self.model, self.data)
            # Cache trocar sites + goal_tip (success criterion uses these)
            self._p_entry = self.data.site_xpos[self.target_entry_id].copy()
            self._p_depth = self.data.site_xpos[self.target_depth_id].copy()
            axis_dir = (self._p_depth - self._p_entry) / (np.linalg.norm(self._p_depth - self._p_entry) + 1e-10)
            retreat_m = self.retreat_mm / 1000.0
            self._goal_tip = self._p_entry - axis_dir * retreat_m
            curr_tip = self.data.site_xpos[self.tip_id].copy()
            curr_back = self.data.site_xpos[self.back_id].copy()
            needle_len = np.linalg.norm(curr_tip - curr_back)
            self._goal_back = self._p_entry - axis_dir * (retreat_m + needle_len)
            actual_dist = np.linalg.norm(self.data.site_xpos[self.tip_id] - self._p_entry) * 1000.0
            self.align_hold_counter = 0
            self.last_perturb_info = {
                "perturb_x_mm": float(grid_cell["x_mm"]),
                "perturb_y_mm": float(grid_cell["y_mm"]),
                "perturb_z_mm": float(grid_cell["z_mm"]),
                "perturb_angle_deg": float(ang),
                "perturb_dist_mm": float(np.sqrt(grid_cell["x_mm"]**2 + grid_cell["y_mm"]**2 + grid_cell["z_mm"]**2)),
                "initial_dist_mm": float(actual_dist),
            }
            print(f"  PhantomCell: pos=({grid_cell['x_mm']:.1f},{grid_cell['y_mm']:.1f},{grid_cell['z_mm']:.1f})mm "
                  f"angle={ang:.1f}deg | tip_to_trocar={actual_dist:.1f}mm")
            return
        # ============ Legacy: random perturbation around pre-align ============
        if self.randomize_phantom:
            # Invalidate cache so _ensure_aligned_state re-runs
            self._aligned_qpos = None
        self._ensure_aligned_state()

        for attempt in range(max_retries):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:self.n_motors] = self._aligned_qpos
            self.data.qvel[:self.n_motors] = self._aligned_qvel
            mujoco.mj_forward(self.model, self.data)

            if grid_cell is not None:
                # Deterministic perturbation from grid cell
                perturb_xyz = np.array([
                    grid_cell["x_mm"] / 1000.0,
                    grid_cell["y_mm"] / 1000.0,
                    grid_cell["z_mm"] / 1000.0,
                ])
                perturb_angle_rad = np.deg2rad(grid_cell["angle_deg"])
                random_axis = np.array([1.0, 0.0, 0.0])  # fixed axis for reproducibility
            else:
                # Random perturbation — Z 방향 균등 배분
                pick_z_neg = self._z_dir_counts["neg"] <= self._z_dir_counts["pos"]
                if pick_z_neg and PERTURB_POS_Z_MIN_MM < 0:
                    z_val = np.random.uniform(PERTURB_POS_Z_MIN_MM, 0) / 1000.0
                else:
                    z_val = np.random.uniform(0, PERTURB_POS_Z_MAX_MM) / 1000.0
                perturb_xyz = np.array([
                    np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                    np.random.uniform(-PERTURB_POS_XY_MM, PERTURB_POS_XY_MM) / 1000.0,
                    z_val,
                ])
                perturb_angle_rad = np.deg2rad(np.random.uniform(-PERTURB_ANGLE_DEG, PERTURB_ANGLE_DEG))
                random_axis = np.random.randn(3)
                random_axis = random_axis / (np.linalg.norm(random_axis) + 1e-10)

            perturbed_tip = self._goal_tip + perturb_xyz
            rot_mat_perturb = np.eye(3)
            if abs(perturb_angle_rad) > 1e-6:
                K = np.array([
                    [0, -random_axis[2], random_axis[1]],
                    [random_axis[2], 0, -random_axis[0]],
                    [-random_axis[1], random_axis[0], 0],
                ])
                rot_mat_perturb = np.eye(3) + np.sin(perturb_angle_rad) * K + (1 - np.cos(perturb_angle_rad)) * (K @ K)
            perturbed_back_dir = rot_mat_perturb @ (self._goal_back - self._goal_tip)
            perturbed_back = perturbed_tip + perturbed_back_dir

            # IK to perturbed position (smooth interpolation to avoid singularity)
            converged = False
            move_speed = 0.05  # m/s
            move_dist = np.linalg.norm(perturbed_tip - self._goal_tip)
            move_duration = max(move_dist / move_speed, 0.1)
            move_start_time = self.data.time

            for ps in range(5000):
                t = (self.data.time - move_start_time) / move_duration
                alpha = smooth_step(min(t, 1.0))
                interp_tip = (1 - alpha) * self._goal_tip + alpha * perturbed_tip
                interp_back = (1 - alpha) * self._goal_back + alpha * perturbed_back

                self._run_ik_step(interp_tip, interp_back)
                mujoco.mj_step(self.model, self.data)

                if t >= 1.0:
                    if np.linalg.norm(self.data.site_xpos[self.tip_id] - perturbed_tip) < 0.001:
                        for _ in range(200):
                            self._run_ik_step(perturbed_tip, perturbed_back)
                            mujoco.mj_step(self.model, self.data)
                        converged = True
                        break
                    if ps > 4500:
                        break

            # Verify: actual tip distance to trocar entry should be reasonable
            actual_dist = np.linalg.norm(self.data.site_xpos[self.tip_id] - self._p_entry) * 1000.0
            z_abs_max = max(abs(PERTURB_POS_Z_MIN_MM), abs(PERTURB_POS_Z_MAX_MM))
            max_expected = np.sqrt(PERTURB_POS_XY_MM**2 * 2 + z_abs_max**2) + 5.0  # margin

            if converged and actual_dist < max_expected:
                # Occlusion check: Z 음수일 때 바늘팁이 팬텀에 가려지면 재시도
                if perturb_xyz[2] < 0 and self._check_tip_occluded():
                    print(f"  Perturbation attempt {attempt+1}: Z<0 occluded, retrying...")
                    continue
                perturb_dist = np.linalg.norm(perturb_xyz) * 1000
                # Z 방향 카운터 업데이트
                z_dir_key = "neg" if perturb_xyz[2] < 0 else "pos"
                self._z_dir_counts[z_dir_key] += 1
                print(f"  Perturbation: pos={perturb_dist:.1f}mm, angle={np.rad2deg(perturb_angle_rad):.1f}deg, "
                      f"actual_dist={actual_dist:.1f}mm, z_dir={z_dir_key}")
                break
            else:
                print(f"  Perturbation attempt {attempt+1} failed (converged={converged}, "
                      f"actual_dist={actual_dist:.1f}mm), retrying...")

        if not converged or actual_dist >= max_expected:
            print(f"  WARNING: Could not find valid perturbation after {max_retries} retries, "
                  f"using last attempt (dist={actual_dist:.1f}mm)")

        self.align_hold_counter = 0

        # Store perturbation info for analysis
        self.last_perturb_info = {
            "perturb_x_mm": perturb_xyz[0] * 1000,
            "perturb_y_mm": perturb_xyz[1] * 1000,
            "perturb_z_mm": perturb_xyz[2] * 1000,
            "perturb_angle_deg": np.rad2deg(perturb_angle_rad),
            "perturb_dist_mm": np.linalg.norm(perturb_xyz) * 1000,
            "initial_dist_mm": actual_dist,
        }

    def get_ee_pose(self):
        # TCP shifted to needle_tip site (matches Save_dataset_align_only.py).
        # Tip is +177.5mm along flange Z; rotation identical to flange.
        pos = self.data.site_xpos[self.tip_id].copy() * 1000.0
        mat = self.data.site_xmat[self.tip_id].reshape(3, 3)
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
        # Model output is delta in TIP frame (matches dataset). Apply: build tip_target,
        # then invert to flange_target for the link6 Jacobian IK below.
        # Tip → flange: p_flange = p_tip - R_target @ TIP_OFFSET, R unchanged.
        from src.utils.tip_frame import TIP_OFFSET_M  # local import to avoid top-level coupling

        current_tip_ee = self.get_ee_pose()
        target_tip_ee = current_tip_ee + delta_ee_6d
        target_rpy = target_tip_ee[3:]
        target_R = self._rpy_to_rotmat(target_rpy)
        # tip target pos (m) → flange target pos (m)
        target_tip_pos_m = target_tip_ee[:3] / 1000.0
        target_pos_m = target_tip_pos_m - target_R @ TIP_OFFSET_M

        for _ in range(n_sim_steps):
            cur_pos = self.data.xpos[self.link6_id].copy()
            cur_mat = self.data.xmat[self.link6_id].reshape(3, 3)

            err_pos = target_pos_m - cur_pos
            target_mat = target_R
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
        """Check if needle tip is aligned to goal_tip (retreated from entry)."""
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()

        dist = np.linalg.norm(tip_pos - self._goal_tip)

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

        aligned = dist < ALIGN_SUCCESS_THRESHOLD_M and angle_deg < ALIGN_SUCCESS_ANGLE_DEG

        if self.use_sensor_success and aligned:
            sensor_dist = self.get_sensor_dist()
            aligned = aligned and (sensor_dist >= ALIGN_SUCCESS_SENSOR_MIN_MM or sensor_dist < 0)

        if aligned:
            self.align_hold_counter += 1
        else:
            self.align_hold_counter = 0

        return self.align_hold_counter >= ALIGN_SUCCESS_HOLD_STEPS

    def get_alignment_dist_mm(self):
        """Distance from needle tip to goal_tip (retreated from entry) in mm."""
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        return np.linalg.norm(tip_pos - self._goal_tip) * 1000.0

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

        sensor_dist = self.get_sensor_dist()

        return {
            "dist_mm": dist_mm,
            "insertion_depth_mm": insertion_depth_mm,
            "lateral_mm": lateral_mm,
            "angle_deg": angle_deg,
            "sensor_dist_mm": sensor_dist,
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

    # Seed: CLI override > config (reproducibility for random mode)
    eval_seed_override = getattr(cfg, "eval_seed", None)
    seed = eval_seed_override if eval_seed_override is not None else cfg.eval.seed
    if shard_id is not None:
        seed = seed + shard_id * 1000
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[seed] eval seed = {seed} (shard_id={shard_id})")

    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)
    model = load_model(checkpoint_path, diffusion_steps=diff_steps, scheduler_type=sched_type, train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    # Keypoint inferencer — provides per-step (troc_u, troc_v, dist_norm).
    # Used for: (a) VLA proprio when proprio_dim=9, (b) keypoint-based handoff trigger.
    kp_inferencer = None
    uv_ckpt = getattr(cfg, "uv_ckpt", None)
    dist_ckpt = getattr(cfg, "dist_ckpt", None)
    pdim = getattr(model, "proprio_dim", 8)
    if uv_ckpt and dist_ckpt:
        kp_inferencer = KeypointInferencer(uv_ckpt, dist_ckpt, device="cuda",
                                            domain=getattr(cfg, "kp_domain", "sim"))
        print(f"[kp_inferencer] active (pdim={pdim}, use_kp_handoff={getattr(cfg, 'use_kp_handoff', False)})")
    elif pdim == 9 and not getattr(cfg, "use_oracle_kp", False):
        raise RuntimeError(f"model.proprio_dim=9 requires --uv-ckpt+--dist-ckpt OR --oracle-kp")
    if getattr(cfg, "use_oracle_kp", False):
        print(f"[oracle-kp] ACTIVE — VLA proprio uses GT keypoints (diagnostic upper bound)")

    image_size = getattr(cfg.eval, "image_size", 256)
    num_episodes = getattr(cfg.eval, "num_episodes", 50)
    max_steps = getattr(cfg, "max_steps", None) or getattr(cfg.eval, "max_steps_per_episode", 200)
    num_steps_execute = getattr(cfg.eval, "num_steps_execute", 1)
    sim_steps_per_ctrl = getattr(cfg.eval, "sim_steps_per_control", 67)
    save_video = getattr(cfg.eval, "save_video", True)
    video_fps = getattr(cfg.eval, "video_fps", 15)

    # Perturbation mode: random (legacy) or grid (deterministic)
    perturb_mode = getattr(cfg, "perturb_mode", "random")
    grid_cells_all = None
    if perturb_mode == "grid":
        # Optional override of perturb ranges (realistic eval).
        x_rng = getattr(cfg, "perturb_xy_mm", PERTURB_POS_XY_MM)
        y_rng = (getattr(cfg, "perturb_y_min_mm", PERTURB_POS_Y_MIN_MM),
                 getattr(cfg, "perturb_y_max_mm", PERTURB_POS_Y_MAX_MM))
        z_rng = (getattr(cfg, "perturb_z_min_mm", PERTURB_POS_Z_MIN_MM),
                 getattr(cfg, "perturb_z_max_mm", PERTURB_POS_Z_MAX_MM))
        a_rng = getattr(cfg, "perturb_angle_deg", PERTURB_ANGLE_DEG)
        grid_cells_all = build_perturb_grid(
            xy_steps=getattr(cfg, "xy_steps", 5),
            z_steps=getattr(cfg, "z_steps", 2),
            angle_steps=getattr(cfg, "angle_steps", 1),
            repeats=getattr(cfg, "repeats", 1),
            x_range=x_rng, y_range=y_rng, z_range=z_rng, angle_range=a_rng,
            x_steps=getattr(cfg, "x_steps", None),
            y_steps=getattr(cfg, "y_steps", None),
        )
        num_episodes = len(grid_cells_all)
        print(f"[grid] perturb_mode=grid → {num_episodes} cells "
              f"(xy={getattr(cfg, 'xy_steps', 5)}, z={getattr(cfg, 'z_steps', 2)}, "
              f"angle={getattr(cfg, 'angle_steps', 1)}, repeats={getattr(cfg, 'repeats', 1)})")

    # Build episode list (1-based) and assign grid cell per episode
    all_episodes = list(range(1, num_episodes + 1))
    if shard_id is not None and num_shards is not None:
        all_episodes = [ep for ep in all_episodes if (ep - 1) % num_shards == shard_id]
        shard_suffix = f"_shard{shard_id}"
    else:
        shard_suffix = ""

    # Map episode index → grid cell (None for random mode)
    if grid_cells_all is not None:
        ep_to_cell = {ep: grid_cells_all[ep - 1] for ep in all_episodes}
    else:
        ep_to_cell = {ep: None for ep in all_episodes}

    # Output directory
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1]
    except (ValueError, IndexError):
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"align_eval_step{step_str}_exec{num_steps_execute}_diff{diff_steps}{shard_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    model_xml = os.path.abspath(SIM_MODEL_PATH)
    randomize_phantom = getattr(cfg, "randomize_phantom", False)
    phantom_pos = getattr(cfg, "phantom_pos", None)
    use_sensor_success = getattr(cfg, 'use_sensor_success', False)
    use_sensor_stop = bool(getattr(cfg, 'use_sensor_stop', False))
    retreat_mm = getattr(cfg, 'retreat_mm', 10.0)
    env = AlignSimEnv(model_xml, randomize_phantom=randomize_phantom, use_sensor_success=use_sensor_success, phantom_pos=phantom_pos, retreat_mm=retreat_mm)
    if use_sensor_stop:
        print(f"⚡ sensor-stop ON: trigger requires (1) sensor ≤ {SENSOR_STOP_CLOSE_MM:.1f}mm at some prior step "
              f"AND (2) sensor ≥ {SENSOR_STOP_HOLE_MM:.1f}mm sustained for {SENSOR_STOP_HOLD_STEPS} steps")

    total_successes = 0

    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = ["episode", "success", "success_reason", "steps", "final_dist_mm",
                   "final_lateral_mm", "final_angle_deg", "min_dist_mm",
                   "final_sensor_dist_mm",
                   "perturb_x_mm", "perturb_y_mm", "perturb_z_mm",
                   "perturb_angle_deg", "perturb_dist_mm", "initial_dist_mm"]
    if grid_cells_all is not None:
        csv_header.extend(["cell_idx", "repeat_id"])
    if randomize_phantom or phantom_pos is not None:
        csv_header.extend(["phantom_x", "phantom_y", "phantom_angle_deg"])
    csv_writer.writerow(csv_header)

    # VQA dump setup (one frame per episode, best within target lateral band).
    vqa_dump_dir = getattr(cfg, "dump_vqa_out", None)
    vqa_records = []
    if vqa_dump_dir:
        os.makedirs(os.path.join(vqa_dump_dir, "frames"), exist_ok=True)
        print(f"[VQA dump] enabled — saving frames with lateral in "
              f"[{cfg.vqa_band_lo}, {cfg.vqa_band_hi}] mm to {vqa_dump_dir}/")

    for ep in all_episodes:
        env.reset(grid_cell=ep_to_cell[ep])

        image_history = []
        image_history_wrist = []
        image_history_top = []
        state_history = []
        action_history = []
        action_buffer = []
        replay_images = []
        metrics_history = []
        vqa_best = None  # (score, dict) for this ep

        success = False
        success_reason = ""  # "" | "dist" | "sensor_stop"
        sensor_spike_count = 0
        sensor_was_close = False  # arms the sensor_stop trigger
        # Per-step keypoint-predicted dist/lateral history (for handoff trigger)
        kp_dist_history = []  # predicted dist_mm per step
        kp_lateral_history = []  # |troc_uv - tip_uv| in normalized [0,1] space per step

        overlay_source = getattr(cfg, "overlay_source", "off")
        overlay_color = tuple(getattr(cfg, "overlay_color", (255, 0, 0)))
        overlay_radius_px = int(getattr(cfg, "overlay_radius_px", 3))
        # Aliased import to avoid shadowing the module-level `draw_overlay`
        # (sim_eval.draw_overlay = trajectory replay overlay used later at line ~1125).
        if overlay_source != "off":
            from src.utils.overlay_utils import draw_overlay as _draw_goal_overlay

        for ctrl_step in range(max_steps):
            frames = env.render_cameras()

            # SutureBot goal-pixel overlay (applied BEFORE resize/preprocess).
            # Modifies frames["tool_camera"] in place.
            if overlay_source == "gt":
                _m = env.get_spatial_metrics()
                _troc_uv = _m.get("trocar_uv")
                if _troc_uv is not None:
                    _draw_goal_overlay(frames["tool_camera"],
                                       (float(_troc_uv[0]), float(_troc_uv[1])),
                                       color=overlay_color, radius_px=overlay_radius_px)
            elif overlay_source == "predicted" and kp_inferencer is not None:
                _tu, _tv, _ = kp_inferencer.predict(frames["tool_camera"])
                _draw_goal_overlay(frames["tool_camera"], (_tu, _tv),
                                   color=overlay_color, radius_px=overlay_radius_px)

            img_ext = preprocess_image(frames["side_camera"], (image_size, image_size))
            img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))
            img_top = preprocess_image(frames["top_camera"], (image_size, image_size))

            # Use wrist as primary view (configured via train config view_mode)
            img_primary = img_wrist

            image_history.append(img_primary)
            image_history_wrist.append(img_wrist)
            image_history_top.append(img_top)

            metrics = env.get_spatial_metrics()
            metrics_history.append(metrics)

            # VQA frame capture: keep frame closest to band center, with trocar in-frame.
            if vqa_dump_dir:
                lat = float(metrics["lateral_mm"])
                if cfg.vqa_band_lo <= lat <= cfg.vqa_band_hi:
                    tip_uv_n = metrics.get("tip_uv"); troc_uv_n = metrics.get("trocar_uv")
                    if tip_uv_n is not None and troc_uv_n is not None:
                        tip_uv_px = (float(tip_uv_n[0]) * 256.0, float(tip_uv_n[1]) * 256.0)
                        troc_uv_px = (float(troc_uv_n[0]) * 256.0, float(troc_uv_n[1]) * 256.0)
                        in_frame = (0 <= troc_uv_px[0] < 256) and (0 <= troc_uv_px[1] < 256)
                        if in_frame:
                            band_mid = (cfg.vqa_band_lo + cfg.vqa_band_hi) / 2.0
                            score = abs(lat - band_mid)
                            if vqa_best is None or score < vqa_best[0]:
                                vqa_best = (score, {
                                    "lateral_mm": lat,
                                    "angle_deg": float(metrics["angle_deg"]),
                                    "dist_mm": float(metrics["dist_mm"]),
                                    "sensor_dist_mm": float(metrics["sensor_dist_mm"]) if metrics["sensor_dist_mm"] is not None else None,
                                    "tip_uv": list(tip_uv_px),
                                    "trocar_uv": list(troc_uv_px),
                                    "delta_uv": [troc_uv_px[0] - tip_uv_px[0], troc_uv_px[1] - tip_uv_px[1]],
                                    "img": frames["tool_camera"].copy(),
                                })

            replay_frame = np.concatenate([img_ext, img_wrist, img_top], axis=1)

            ee_pose = env.get_ee_pose()  # mujoco extrinsic XYZ
            # Match training convention: proprio orientation in Mecademic intrinsic XYZ
            ee_pose_mec = ee_pose.copy()
            ee_pose_mec[3:6] = mujoco_to_mecademic_euler(ee_pose[3:6])
            # Proprio: 6-DoF EE pose + sensor feat (binary 2D or continuous 1D, matched to train cfg).
            proprio6 = ee_pose_mec[:6].astype(np.float32)
            sensor_raw_mm = float(env.get_sensor_dist())
            pdim = getattr(model, "proprio_dim", 8)
            if pdim <= 6:
                proprio = proprio6  # (6,)
            elif pdim == 7:
                # continuous 1D — must match dataset clip (default 30mm)
                clip_mm = float(getattr(model, "sensor_clip_mm", 30.0))
                s = max(0.0, min(sensor_raw_mm, clip_mm)) / clip_mm
                proprio = np.concatenate([proprio6, np.array([s], dtype=np.float32)])  # (7,)
            elif pdim == 9:
                # Keypoint signal source: oracle (GT) or kp_inferencer (learned).
                # Oracle = upper bound (matches training distribution exactly).
                if getattr(cfg, "use_oracle_kp", False):
                    om = env.get_spatial_metrics()
                    tu, tv = float(om["trocar_uv"][0]), float(om["trocar_uv"][1])
                    dnorm = min(float(om["dist_mm"]) / 50.0, 2.0)
                elif kp_inferencer is not None:
                    tu, tv, dnorm = kp_inferencer.predict(frames["tool_camera"])
                else:
                    raise RuntimeError("proprio_dim=9 requires --uv-ckpt+--dist-ckpt OR --oracle-kp")
                proprio = np.concatenate([proprio6, np.array([tu, tv, dnorm], dtype=np.float32)])  # (9,)
                kp_dist_history.append(dnorm * 50.0)  # mm
                kp_lateral_history.append(float(np.hypot(tu - 0.488, tv - 0.326)))  # vs constant tip
            else:
                sensor_close = 1.0 if (0.0 <= sensor_raw_mm <= 5.0) else 0.0
                hole_through = 1.0 if (sensor_raw_mm >= 15.0) else 0.0
                proprio = np.concatenate([proprio6, np.array([sensor_close, hole_through], dtype=np.float32)])  # (8,)
            # When pdim != 9 but kp_inferencer available: still predict for handoff trigger.
            if kp_inferencer is not None and pdim != 9:
                tu, tv, dnorm = kp_inferencer.predict(frames["tool_camera"])
                kp_dist_history.append(dnorm * 50.0)
                kp_lateral_history.append(float(np.hypot(tu - 0.488, tv - 0.326)))
            state_history.append(proprio)

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

            # Angle-dominated action gate: when model issues large rx/ry rotation, suppress
            # the tip-frame Z retreat component (positive Z = away from trocar) to prevent
            # the "tries to fix angle but distance grows" failure mode.
            # Env-gated: ANGLE_Z_GATE=1 enables (default off — empirically no SR change vs baseline 67%).
            import os as _os
            if _os.getenv("ANGLE_Z_GATE", "0") == "1":
                _thr = float(_os.getenv("ANGLE_Z_GATE_THR", "0.05"))
                _angmag = float(abs(denorm_action[3]) + abs(denorm_action[4]))
                if _angmag > _thr and denorm_action[2] > 0:
                    denorm_action = denorm_action.copy()
                    denorm_action[2] = 0.0

            # Close-range action damping: when sensor reads close, scale action down.
            # scale = clamp((sensor - MIN)/(MAX - MIN), SCALE_MIN, 1.0)
            # Env-gated: ACTION_SCALE_NEAR=1, defaults MAX=8mm, MIN=2mm, SCALE_MIN=0.3.
            if _os.getenv("ACTION_SCALE_NEAR", "0") == "1":
                _smax = float(_os.getenv("ACTION_SCALE_MAX_MM", "8.0"))
                _smin = float(_os.getenv("ACTION_SCALE_MIN_MM", "2.0"))
                _floor = float(_os.getenv("ACTION_SCALE_FLOOR", "0.3"))
                _s = float(sensor_raw_mm) if sensor_raw_mm is not None else _smax
                if 0.0 < _s < _smax:
                    _scale = max(_floor, min(1.0, (_s - _smin) / (_smax - _smin)))
                    denorm_action = denorm_action.copy()
                    denorm_action *= _scale

            # denorm_action[3:6] is delta in Mecademic intrinsic XYZ (training convention).
            # Convert to delta in MuJoCo extrinsic XYZ for apply_delta_ee.
            target_mec_rpy = ee_pose_mec[3:6] + denorm_action[3:6]
            target_mujoco_rpy = mecademic_to_mujoco_euler(target_mec_rpy)
            delta_rpy_mujoco = target_mujoco_rpy - ee_pose[3:6]
            delta_rpy_mujoco = np.arctan2(np.sin(delta_rpy_mujoco), np.cos(delta_rpy_mujoco))
            delta_ee = np.concatenate([denorm_action[:3], delta_rpy_mujoco]).astype(np.float32)

            env.apply_delta_ee(delta_ee, n_sim_steps=sim_steps_per_ctrl)

            # Sensor-stop trigger (state machine):
            #   Phase A — approaching: wait for raw sensor to dip below CLOSE_MM
            #            (= needle has reached the trocar surface vicinity).
            #   Phase B — armed: once sensor was close, watch for sustained
            #            spike ≥ HOLE_MM = ray punched through into the hole.
            # Avoids false positives from "far reading happens to be ≥ HOLE_MM
            # mid-approach" because we only trigger after a confirmed
            # close-then-spike transition.
            if use_sensor_stop:
                raw_sensor = env.get_sensor_dist()
                if 0.0 <= raw_sensor <= SENSOR_STOP_CLOSE_MM:
                    sensor_was_close = True
                    sensor_spike_count = 0
                elif sensor_was_close and raw_sensor >= SENSOR_STOP_HOLE_MM:
                    sensor_spike_count += 1
                    if sensor_spike_count >= SENSOR_STOP_HOLD_STEPS:
                        success = True
                        success_reason = "sensor_stop"
                        metrics_history.append(env.get_spatial_metrics())
                        break
                else:
                    sensor_spike_count = 0

            if env.check_success():
                success = True
                success_reason = "dist"
                metrics_history.append(env.get_spatial_metrics())
                break

            # === Mid-trajectory keypoint-based handoff trigger ===
            # If KP-handoff enabled AND current predicted dist < threshold AND lateral OK:
            # break out of VLA loop immediately at the close state. Handoff fires below.
            if (kp_inferencer is not None
                    and getattr(cfg, "use_kp_handoff", False)
                    and getattr(cfg, "kp_inline_trigger", True)
                    and len(kp_dist_history) > 0):
                cur_kp_dist = kp_dist_history[-1]
                cur_kp_lat = kp_lateral_history[-1]
                lat_thr = getattr(cfg, "kp_lateral_thresh", None)
                if cur_kp_dist <= getattr(cfg, "handoff_trigger_mm", 15.0):
                    if lat_thr is None or cur_kp_lat <= lat_thr:
                        print(f"  [inline-trigger step {ctrl_step}] kp_dist={cur_kp_dist:.2f}mm "
                              f"kp_lat_norm={cur_kp_lat:.3f} → break VLA, run handoff")
                        metrics_history.append(env.get_spatial_metrics())
                        break

        # --- Sensor handoff (only if VLA did NOT succeed, and got close enough) ---
        # Trigger source: oracle dist (default) or keypoint-predicted dist (calibration-free).
        # Keypoint trigger = realistic real-world scenario; oracle = upper bound.
        handoff_log = None
        oracle_min_dist = min(m["dist_mm"] for m in metrics_history)
        kp_min_dist = None
        kp_min_lateral_norm = None
        if kp_inferencer is not None and len(kp_dist_history) > 0:
            kp_min_dist = min(kp_dist_history)
            kp_min_lateral_norm = min(kp_lateral_history)
        use_kp_trigger = getattr(cfg, "use_kp_handoff", False) and kp_min_dist is not None
        trigger_dist = kp_min_dist if use_kp_trigger else oracle_min_dist
        trigger_thresh = getattr(cfg, "handoff_trigger_mm", 15.0)
        # Smart trigger: dist + lateral both close (keypoint mode only)
        if use_kp_trigger and getattr(cfg, "kp_lateral_thresh", None) is not None:
            kp_lateral_ok = kp_min_lateral_norm <= cfg.kp_lateral_thresh
        else:
            kp_lateral_ok = True
        if (getattr(cfg, "use_handoff", False)
                and not success
                and trigger_dist <= trigger_thresh
                and kp_lateral_ok):
            print(f"  [handoff trigger] mode={'kp' if use_kp_trigger else 'oracle'} "
                  f"dist={trigger_dist:.2f}mm thr={trigger_thresh:.1f}mm "
                  f"lat={kp_min_lateral_norm if kp_min_lateral_norm is not None else 'n/a'}")
            from scripts.sensor_handoff import run_sensor_handoff
            # Compute keypoint-seeded lateral offset for handoff grid steering.
            kp_seed_mm = None
            if kp_inferencer is not None and getattr(cfg, "kp_seed_handoff", True):
                try:
                    frames_now = env.render_cameras()
                    cam_pos = env.data.cam_xpos[env._tool_cam_id].copy()
                    cam_mat = env.data.cam_xmat[env._tool_cam_id].reshape(3, 3)
                    fovy = float(env.model.cam_fovy[env._tool_cam_id])
                    kp_seed_mm, kp_pred_dist, kp_pred_lat = kp_inferencer.predict_world_seed(
                        frames_now["tool_camera"], cam_pos, cam_mat, fovy_deg=fovy
                    )
                    print(f"  [kp_seed] world_delta={kp_seed_mm.round(2)}mm "
                          f"predicted_dist={kp_pred_dist:.2f}mm lat_norm={kp_pred_lat:.3f}")
                except Exception as e:
                    print(f"  [kp_seed] failed: {e}; skipping seed")
                    kp_seed_mm = None
            kp_track_fn = None
            kp_track_iters = int(getattr(cfg, "kp_track_iters", 1))
            if kp_inferencer is not None and kp_track_iters > 1:
                def _kp_track_fn(_env, _kpi=kp_inferencer):
                    _frames = _env.render_cameras()
                    _cpos = _env.data.cam_xpos[_env._tool_cam_id].copy()
                    _cmat = _env.data.cam_xmat[_env._tool_cam_id].reshape(3, 3)
                    _fovy = float(_env.model.cam_fovy[_env._tool_cam_id])
                    wdelta, _d, latn = _kpi.predict_world_seed(_frames["tool_camera"], _cpos, _cmat, fovy_deg=_fovy)
                    return wdelta, latn
                kp_track_fn = _kp_track_fn
            try:
                # KP query for sweep video labels (only used when HANDOFF_VIDEO_SWEEP=1).
                kp_query_fn = None
                if kp_inferencer is not None and os.environ.get("HANDOFF_VIDEO_SWEEP", "0") == "1":
                    def _kp_query(_env, _kpi=kp_inferencer):
                        _frames = _env.render_cameras()
                        _cpos = _env.data.cam_xpos[_env._tool_cam_id].copy()
                        _cmat = _env.data.cam_xmat[_env._tool_cam_id].reshape(3, 3)
                        _fovy = float(_env.model.cam_fovy[_env._tool_cam_id])
                        _wd, _d, _ln = _kpi.predict_world_seed(_frames["tool_camera"], _cpos, _cmat, fovy_deg=_fovy)
                        return float(_d), float(_ln)
                    kp_query_fn = _kp_query
                handoff_log = run_sensor_handoff(env, verbose=True, frames_out=replay_images,
                                                  keypoint_seed_world_mm=kp_seed_mm,
                                                  keypoint_track_fn=kp_track_fn,
                                                  kp_track_iters=kp_track_iters,
                                                  kp_query_fn=kp_query_fn)
                metrics_history.append(env.get_spatial_metrics())
                # Re-evaluate success after handoff. Use stricter "insertion achieved" criterion:
                #   - did_align (sensor through-hole confirmed)
                #   - insertion_depth_mm > 2 (needle pushed into trocar)
                if not success and handoff_log["aligned"]["achieved"]:
                    post_m = env.get_spatial_metrics()
                    if post_m.get("insertion_depth_mm", 0) > 2.0 or env.check_success():
                        success = True
                        success_reason = "handoff"
            except Exception as e:
                print(f"  [handoff] ERROR: {e}")

        if success:
            total_successes += 1

        # Save VQA frame for this ep if captured.
        if vqa_dump_dir and vqa_best is not None:
            _, d = vqa_best
            fname = f"ep{ep:03d}.png"
            from PIL import Image as _PILImage
            _PILImage.fromarray(d["img"]).save(os.path.join(vqa_dump_dir, "frames", fname))
            du, dv = d["delta_uv"]
            mag = float(np.hypot(du, dv))
            if mag < 8.0:
                gt_dir = "centered"
            else:
                ang_img = (np.degrees(np.arctan2(-dv, du)) + 360.0) % 360.0
                sectors = [("right", 0.0), ("up-right", 45.0), ("up", 90.0), ("up-left", 135.0),
                           ("left", 180.0), ("down-left", 225.0), ("down", 270.0), ("down-right", 315.0)]
                gt_dir = min(sectors, key=lambda s: min(abs(ang_img - s[1]), 360.0 - abs(ang_img - s[1])))[0]
            lat = d["lateral_mm"]
            gt_mag = "tiny" if lat < 2 else ("small" if lat < 5 else ("medium" if lat < 10 else "large"))
            vqa_records.append({
                "ep": int(ep), "frame": f"frames/{fname}",
                "lateral_mm": lat, "angle_deg": d["angle_deg"], "dist_mm": d["dist_mm"],
                "sensor_dist_mm": d["sensor_dist_mm"],
                "tip_uv": d["tip_uv"], "trocar_uv": d["trocar_uv"], "delta_uv": d["delta_uv"],
                "trocar_in_frame": True,
                "gt_direction": gt_dir, "gt_magnitude": gt_mag,
            })
            print(f"  [VQA] saved {fname}: lat={lat:.2f} ang={d['angle_deg']:.1f} -> {gt_dir}/{gt_mag}")

        ep_done = all_episodes.index(ep) + 1
        sr = total_successes / ep_done * 100
        final_m = metrics_history[-1]
        min_dist = min(m["dist_mm"] for m in metrics_history)
        outcome = f"SUCCESS[{success_reason}]" if success else "FAIL"
        msg = (f"Episode {ep}/{num_episodes} | {outcome} | "
               f"Steps: {ctrl_step + 1} | SR: {sr:.1f}% ({total_successes}/{ep_done}) | "
               f"dist={final_m['dist_mm']:.1f}mm lateral={final_m['lateral_mm']:.1f}mm "
               f"angle={final_m['angle_deg']:.1f}deg min_dist={min_dist:.1f}mm "
               f"sensor={final_m.get('sensor_dist_mm', -1):.1f}mm")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        pi = env.last_perturb_info
        final_sensor = final_m.get('sensor_dist_mm', -1.0)
        row = [
            ep, int(success), success_reason or "fail", ctrl_step + 1,
            f"{final_m['dist_mm']:.2f}",
            f"{final_m['lateral_mm']:.2f}", f"{final_m['angle_deg']:.2f}",
            f"{min_dist:.2f}",
            f"{final_sensor:.2f}",
            f"{pi['perturb_x_mm']:.2f}", f"{pi['perturb_y_mm']:.2f}", f"{pi['perturb_z_mm']:.2f}",
            f"{pi['perturb_angle_deg']:.2f}", f"{pi['perturb_dist_mm']:.2f}", f"{pi['initial_dist_mm']:.2f}",
        ]
        if grid_cells_all is not None:
            cell = ep_to_cell[ep]
            row.extend([cell["cell_idx"], cell["repeat_id"]])
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
        dist_arr = np.array([m.get("dist_mm", np.nan) for m in metrics_history], dtype=np.float32)
        lat_arr = np.array([m.get("lateral_mm", np.nan) for m in metrics_history], dtype=np.float32)
        ang_arr = np.array([m.get("angle_deg", np.nan) for m in metrics_history], dtype=np.float32)
        np.savez_compressed(
            eval_dir / f"traj_ep{ep:03d}_{'S' if success else 'F'}.npz",
            ee_pose=ee_traj[:, :3],     # position (mm)
            dist_mm=dist_arr,           # tip→trocar distance per step
            lateral_mm=lat_arr,         # lateral distance per step
            angle_deg=ang_arr,          # tip-axis vs trocar-axis angle per step
        )

    csv_file.close()

    # Generate trajectory visualization
    _save_trajectory_plot(eval_dir)

    final_sr = total_successes / len(all_episodes) * 100
    summary = f"\n{'='*60}\nFinal Alignment Success Rate: {final_sr:.2f}% ({total_successes}/{len(all_episodes)})\n{'='*60}"
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()

    if vqa_dump_dir and vqa_records:
        import json as _json
        with open(os.path.join(vqa_dump_dir, "ground_truth.json"), "w") as _f:
            _json.dump({"samples": vqa_records, "image_size": [256, 256]}, _f, indent=2)
        print(f"[VQA dump] {len(vqa_records)} frames saved to {vqa_dump_dir}/")

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
    parser = argparse.ArgumentParser(description="Evaluate VLANeXt on fine-alignment task")
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
    parser.add_argument("--retreat-mm", type=float, default=10.0,
                        help="Retreat goal_tip from trocar entry along -axis_dir (mm, default: 10)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override eval.max_steps_per_episode (default: use config value)")
    parser.add_argument("--sensor-success", action="store_true",
                        help="Require sensor to see through trocar hole for success")
    parser.add_argument("--sensor-stop", action="store_true",
                        help=(f"Terminate episode early with success when raw sensor "
                              f">= {SENSOR_STOP_HOLE_MM:.1f}mm for "
                              f"{SENSOR_STOP_HOLD_STEPS} consecutive steps "
                              f"(needle staring through hole = aligned)"))
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="Override eval.seed for reproducibility (random mode)")
    parser.add_argument("--perturb-mode", type=str, default="random",
                        choices=["random", "grid"],
                        help="random: legacy uniform sampling; grid: deterministic 4D grid")
    parser.add_argument("--xy-steps", type=int, default=5,
                        help="(grid) XY axis steps per side. Default 5 → 5x5=25 cells")
    parser.add_argument("--x-steps", type=int, default=None,
                        help="(grid) Override X axis step count (independent of --xy-steps)")
    parser.add_argument("--y-steps", type=int, default=None,
                        help="(grid) Override Y axis step count (independent of --xy-steps)")
    parser.add_argument("--z-steps", type=int, default=2,
                        help="(grid) Z axis steps. Default 2 (near/far)")
    parser.add_argument("--angle-steps", type=int, default=1,
                        help="(grid) Angle steps. Default 1 → angle=0 fixed")
    parser.add_argument("--repeats", type=int, default=1,
                        help="(grid) Repeats per cell (for stochastic policy variance)")
    parser.add_argument("--handoff", action="store_true",
                        help="After VLA loop ends, run sensor-based handoff (lateral sweep + insertion)")
    parser.add_argument("--dump-vqa-out", type=str, default=None,
                        help="If set, dump tool_camera frames + GT JSON whenever lateral_mm in [vqa-band-lo, vqa-band-hi] (at most 1 per ep)")
    parser.add_argument("--vqa-band-lo", type=float, default=1.0)
    parser.add_argument("--vqa-band-hi", type=float, default=15.0)
    parser.add_argument("--uv-ckpt", type=str, default=None,
                        help="Keypoint head ckpt for UV prediction. Used for proprio_dim=9 AND/OR keypoint handoff.")
    parser.add_argument("--dist-ckpt", type=str, default=None,
                        help="Keypoint head ckpt for dist_mm prediction.")
    parser.add_argument("--use-kp-handoff", action="store_true",
                        help="Use keypoint-predicted dist as handoff trigger (vs oracle dist). Calibration-free.")
    parser.add_argument("--kp-lateral-thresh", type=float, default=None,
                        help="Smart trigger: also require predicted lateral (normalized) ≤ this. e.g. 0.05 = 5% image.")
    parser.add_argument("--kp-domain", type=str, default="sim", choices=["sim", "real", "none"],
                        help="Projection bias correction domain. Sim eval default = sim.")
    parser.add_argument("--no-kp-seed-handoff", dest="kp_seed_handoff", action="store_false",
                        help="Disable keypoint-seeded handoff (default ON when kp_inferencer active).")
    parser.set_defaults(kp_seed_handoff=True)
    parser.add_argument("--no-kp-inline-trigger", dest="kp_inline_trigger", action="store_false",
                        help="Disable mid-trajectory KP handoff trigger (default ON: interrupts VLA loop when close).")
    parser.set_defaults(kp_inline_trigger=True)
    parser.add_argument("--oracle-kp", action="store_true",
                        help="Diagnostic: feed VLA oracle (GT) keypoints instead of learned head predictions.")
    parser.add_argument("--perturb-xy-mm", type=float, default=None,
                        help="Override XY perturb range ±mm (default 25).")
    parser.add_argument("--perturb-y-min-mm", type=float, default=None)
    parser.add_argument("--perturb-y-max-mm", type=float, default=None)
    parser.add_argument("--perturb-z-min-mm", type=float, default=None)
    parser.add_argument("--perturb-z-max-mm", type=float, default=None)
    parser.add_argument("--perturb-angle-deg", type=float, default=None,
                        help="Override angle perturb range ±deg (default 25).")
    parser.add_argument("--kp-track-iters", type=int, default=1,
                        help="Iterative KP visual-servo iters BEFORE sensor grid (1=seed only, 3 recommended).")
    parser.add_argument("--handoff-trigger-mm", type=float, default=15.0,
                        help="Only run handoff if min_dist reached during VLA is ≤ this threshold")
    parser.add_argument("--num-inference-timesteps", type=int, default=None,
                        help="Override model.diffusion_steps at eval (e.g. 25/50/100).")
    parser.add_argument("--num-steps-execute", type=int, default=None,
                        help="Override eval.num_steps_execute (action chunk slice length).")
    parser.add_argument("--overlay-source", type=str, default="off",
                        choices=["gt", "predicted", "off"],
                        help="SutureBot goal-pixel overlay source. gt=oracle, predicted=kp head, off=no overlay")
    parser.add_argument("--overlay-color", type=int, nargs=3, default=[255, 0, 0],
                        metavar=("R", "G", "B"))
    parser.add_argument("--overlay-radius-px", type=int, default=3)
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
    cfg.use_sensor_success = args.sensor_success
    cfg.use_sensor_stop = args.sensor_stop
    cfg.retreat_mm = args.retreat_mm
    cfg.max_steps = args.max_steps
    cfg.eval_seed = args.eval_seed
    cfg.perturb_mode = args.perturb_mode
    cfg.xy_steps = args.xy_steps
    cfg.x_steps = args.x_steps
    cfg.y_steps = args.y_steps
    cfg.z_steps = args.z_steps
    cfg.angle_steps = args.angle_steps
    cfg.repeats = args.repeats
    cfg.use_handoff = args.handoff
    cfg.handoff_trigger_mm = args.handoff_trigger_mm
    cfg.uv_ckpt = args.uv_ckpt
    cfg.dist_ckpt = args.dist_ckpt
    cfg.use_kp_handoff = args.use_kp_handoff
    cfg.kp_lateral_thresh = args.kp_lateral_thresh
    cfg.kp_domain = args.kp_domain
    cfg.kp_seed_handoff = args.kp_seed_handoff
    cfg.kp_inline_trigger = args.kp_inline_trigger
    cfg.use_oracle_kp = args.oracle_kp
    cfg.kp_track_iters = args.kp_track_iters
    if args.perturb_xy_mm is not None: cfg.perturb_xy_mm = args.perturb_xy_mm
    if args.perturb_y_min_mm is not None: cfg.perturb_y_min_mm = args.perturb_y_min_mm
    if args.perturb_y_max_mm is not None: cfg.perturb_y_max_mm = args.perturb_y_max_mm
    if args.perturb_z_min_mm is not None: cfg.perturb_z_min_mm = args.perturb_z_min_mm
    if args.perturb_z_max_mm is not None: cfg.perturb_z_max_mm = args.perturb_z_max_mm
    if args.perturb_angle_deg is not None: cfg.perturb_angle_deg = args.perturb_angle_deg
    cfg.dump_vqa_out = args.dump_vqa_out
    cfg.vqa_band_lo = args.vqa_band_lo
    cfg.vqa_band_hi = args.vqa_band_hi
    if args.num_inference_timesteps is not None:
        cfg.model.diffusion_steps = args.num_inference_timesteps
    if args.num_steps_execute is not None:
        cfg.eval.num_steps_execute = args.num_steps_execute
    cfg.overlay_source = args.overlay_source
    cfg.overlay_color = tuple(args.overlay_color)
    cfg.overlay_radius_px = args.overlay_radius_px

    run_eval(cfg)
