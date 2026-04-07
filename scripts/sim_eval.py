"""
sim_eval.py

Evaluates a fine-tuned VLANeXt model in the MuJoCo needle-insertion simulation.

Usage:
python scripts/sim_eval.py --config config/sim_eval_config.yaml \
    --checkpoint /data/public/NAS/VLANeXt/output_step6500.pt

The script:
  1. Loads the fine-tuned model checkpoint
  2. Initializes the MuJoCo needle-insertion environment
  3. Runs closed-loop evaluation: render → model.predict_action → IK → mj_step
  4. Reports success rate and optionally saves rollout videos
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

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.VLANeXt import VLANeXt, LlamaProcessorWrapper
from src.datasets.sim_act import action_min_sim, action_max_sim


# ═══════════════════════════════════════════════════════════════════════════════
# Config helper
# ═══════════════════════════════════════════════════════════════════════════════
class DictConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# Model / Processor Loading  (adapted from VLANeXt_utils.py)
# ═══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, diffusion_steps: int = 10, scheduler_type: str = "flow_match",
               train_config_path: str = None):
    """Load a VLANeXt checkpoint for evaluation.

    Supports two formats:
      1. Full checkpoint with 'config' and 'model_state_dict' keys
      2. State-dict-only file from zero_to_fp32.py (requires train_config_path)
    """
    print(f"Loading model from {checkpoint_path}")

    def _default_train_config_path():
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "sim_train_config.yaml")

    if os.path.isdir(checkpoint_path):
        # Sharded checkpoint directory (from zero_to_fp32.py)
        from safetensors import safe_open
        import json
        index_path = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard in shard_files:
                shard_path = os.path.join(checkpoint_path, shard)
                state_dict.update(torch.load(shard_path, map_location="cpu"))
            print(f"Loaded {len(state_dict)} keys from {len(shard_files)} shards")
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            # Single-file checkpoint (small models like 0.8B)
            single_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            state_dict = torch.load(single_path, map_location="cpu")
            print(f"Loaded {len(state_dict)} keys from single pytorch_model.bin")
        else:
            raise FileNotFoundError(f"No pytorch_model.bin.index.json or pytorch_model.bin in {checkpoint_path}")
        # Check for config saved alongside checkpoint first
        embedded_config = os.path.join(checkpoint_path, "train_config.yaml")
        if os.path.exists(embedded_config):
            train_config_path = embedded_config
            print(f"Using embedded config: {embedded_config}")
        elif train_config_path is None:
            train_config_path = _default_train_config_path()
        print(f"Loading config from {train_config_path}")
        train_config = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Detect format: full checkpoint vs state-dict-only
        if isinstance(checkpoint, dict) and "config" in checkpoint and "model_state_dict" in checkpoint:
            train_config = checkpoint["config"]
            state_dict = checkpoint["model_state_dict"]
        else:
            if train_config_path is None:
                train_config_path = _default_train_config_path()
            print(f"State-dict-only checkpoint detected. Loading config from {train_config_path}")
            train_config = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
            state_dict = checkpoint

    mc = train_config["model"]
    dc = train_config["data"]

    num_train_timesteps = mc.get("num_train_timesteps", mc.get("diffusion_steps", 1000))
    ckpt_inf_steps = mc.get("num_inference_timesteps", mc.get("diffusion_steps", 10))
    num_inference_timesteps = diffusion_steps if diffusion_steps > 0 else ckpt_inf_steps
    sched = scheduler_type or mc.get("scheduler_type", "flow_match")

    model = VLANeXt(
        lmm_path=mc["lmm_path"],
        vision_encoder_path=mc.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
        action_dim=mc["action_dim"],
        num_actions=dc["future_len"],
        num_queries=mc["num_queries"],
        num_history=dc["history_len"],
        loss_type=mc.get("loss_type", "diffusion"),
        future_image_loss_weight=float(mc.get("future_image_loss_weight", 1.0)),
        num_train_timesteps=num_train_timesteps,
        num_inference_timesteps=num_inference_timesteps,
        scheduler_type=sched,
        condition_type=mc.get("condition_type", "loose"),
        policy_hidden_size=mc.get("policy_hidden_size", 1024),
        policy_depth=mc.get("policy_depth", 29),
        policy_num_heads=mc.get("policy_num_heads", 12),
        policy_mlp_ratio=mc.get("policy_mlp_ratio", 4.0),
        use_proprio_input_vlm=mc.get("use_proprio_input_vlm", True),
        use_action_input_policy=mc.get("use_action_input_policy", False),
        use_transformer_proprio_projector=mc.get("use_transformer_proprio_projector", False),
        projector_depth=mc["projector_depth"],
        projector_num_heads=mc["projector_num_heads"],
        use_transformer_connector=mc["use_transformer_connector"],
        connector_depth=mc["connector_depth"],
        connector_num_heads=mc["connector_num_heads"],
        backbone_mode=mc.get("backbone_mode", "finetune"),
        lora_config=mc.get("lora", None),
        gradient_checkpointing=False,
        num_bins=mc.get("num_bins", 256),
        action_vqvae=mc.get("action_vqvae", None),
        generator_hidden_size=mc.get("generator_hidden_size", 768),
        generator_depth=mc.get("generator_depth", 12),
        generator_num_heads=mc.get("generator_num_heads", 12),
        generator_mlp_ratio=mc.get("generator_mlp_ratio", 4.0),
        attn_implementation="flash_attention_2",
        dct_loss_weight=mc.get("dct_loss_weight", 0.1),
        dct_low_freq_weight=mc.get("dct_low_freq_weight", 1.0),
        dct_high_freq_weight=mc.get("dct_high_freq_weight", 1.0),
        dct_freq_split=mc.get("dct_freq_split", 0.125),
        dct_similarity_type=mc.get("dct_similarity_type", "mae"),
        spatial_loss_weight=mc.get("spatial_loss_weight", 0.0),
        proprio_dim=mc.get("proprio_dim", None),
    )

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"State dict loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    model.train_config = train_config
    return model


def load_processor(checkpoint_path: str, train_config_path: str = None):
    """Load the vision-language processor that matches the checkpoint."""
    lmm_path = None
    if not os.path.isdir(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            lmm_path = checkpoint["config"]["model"]["lmm_path"]
    if lmm_path is None:
        if train_config_path is None:
            train_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "sim_train_config.yaml")
        cfg = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
        lmm_path = cfg["model"]["lmm_path"]

    if "llama" in lmm_path.lower():
        ve_path = checkpoint["config"]["model"].get("vision_encoder_path", "google/siglip2-base-patch16-256")
        tokenizer = AutoTokenizer.from_pretrained(lmm_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(ve_path)
        return LlamaProcessorWrapper(tokenizer, image_processor)

    return AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference  (adapted from VLANeXt_utils.py / get_vla_action)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_action(model, processor, obs, task_label):
    """
    Run one forward pass and return an action chunk as numpy (num_actions, 7).

    obs dict keys:
        full_image        : (H, W, 3) uint8 RGB  — current exterior view
        full_image_wrist  : (H, W, 3) uint8 RGB  — current wrist view
        image_history     : list of (H, W, 3)     — past exterior images
        image_history_wrist: list of (H, W, 3)    — past wrist images
        state_history     : list of (7,) float     — past EE poses
        action_history    : list of (7,) float     — past actions
    """
    data_cfg = model.train_config["data"]
    input_modality = data_cfg.get("input_modality", "image")
    view_mode = data_cfg.get("view_mode", "single")
    history_len = getattr(model, "num_history", 0)
    device = next(model.parameters()).device

    effective_processor = getattr(model, "processor", processor)
    is_qwen = "Qwen" in effective_processor.__class__.__name__
    is_llama = "Llama" in effective_processor.__class__.__name__

    # ── helper: pad history to length n ───────────────────────────────────────
    def _take_last(history_list, fallback, n):
        if not history_list:
            history_list = [fallback]
        if n > 0:
            xs = history_list[-n:]
            if len(xs) < n:
                xs = ([xs[0]] * (n - len(xs))) + xs
            return xs
        return [fallback]

    # ── images ────────────────────────────────────────────────────────────────
    all_images = obs.get("image_history", [obs["full_image"]])
    images_np = _take_last(all_images, obs["full_image"], history_len)
    pil_images = [Image.fromarray(img) for img in images_np]

    if view_mode == "multi":
        all_wrist = obs.get("image_history_wrist", [obs["full_image_wrist"]])
        wrist_np = _take_last(all_wrist, obs["full_image_wrist"], history_len)
        pil_wrist = [Image.fromarray(img) for img in wrist_np]
        if "full_image_top" in obs:
            all_top = obs.get("image_history_top", [obs["full_image_top"]])
            top_np = _take_last(all_top, obs["full_image_top"], history_len)
            pil_top = [Image.fromarray(img) for img in top_np]

    # ── proprioception ────────────────────────────────────────────────────────
    proprioception = None
    if model.use_proprio_input_vlm:
        all_states = obs.get("state_history", [])
        if history_len > 0:
            states = all_states[-history_len:]
            if len(states) < history_len:
                states = ([states[0]] * (history_len - len(states))) + states if states else [np.zeros(model.action_dim)] * history_len
        else:
            states = []
        if len(states) > 0:
            proprioception = torch.tensor(np.stack(states), dtype=torch.bfloat16).unsqueeze(0).to(device)

    # ── history actions ───────────────────────────────────────────────────────
    history_actions = None
    if getattr(model, "use_action_input_policy", False):
        all_actions = obs.get("action_history", [])
        if history_len > 0:
            actions = all_actions[-history_len:]
            if len(actions) < history_len:
                actions = ([np.zeros(model.action_dim)] * (history_len - len(actions))) + actions
        else:
            actions = []
        if len(actions) > 0:
            history_actions = torch.tensor(np.stack(actions), dtype=torch.bfloat16).unsqueeze(0).to(device)

    # ── build processor inputs ────────────────────────────────────────────────
    if view_mode == "multi":
        images = [pil_images[-1], pil_wrist[-1]]
        if "full_image_top" in obs:
            images.append(pil_top[-1])
    else:
        images = [pil_images[-1]]

    if is_llama:
        text = [task_label]
        inputs = effective_processor.tokenizer(text, padding=True, return_tensors="pt")
        image_inputs = effective_processor.image_processor(images, return_tensors="pt")
        inputs["pixel_values"] = image_inputs["pixel_values"]
    elif is_qwen:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": task_label})
        messages = [{"role": "user", "content": content}]
        text = effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = effective_processor(text=[text], images=images, padding=True, return_tensors="pt")
    else:
        # PaliGemma / fallback
        text = "<image>" * len(images) + task_label
        inputs = effective_processor(text=[text], images=images, padding=True, return_tensors="pt")

    valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    inputs = {k: v.to(device) for k, v in inputs.items() if k in valid_keys}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    has_spatial = getattr(model, "spatial_head", None) is not None
    if has_spatial:
        action_pred, spatial_pred = model.predict_action(
            proprioception=proprioception,
            history_actions=history_actions,
            return_spatial=True,
            **inputs,
        )
        return action_pred[0].float().cpu().numpy(), spatial_pred
    else:
        action_pred = model.predict_action(
            proprioception=proprioception,
            history_actions=history_actions,
            **inputs,
        )
        return action_pred[0].float().cpu().numpy(), None


# ═══════════════════════════════════════════════════════════════════════════════
# MuJoCo Environment Helpers  (adapted from Save_dataset.py)
# ═══════════════════════════════════════════════════════════════════════════════
SIM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Sim", "meca_add.xml")
TARGET_INSERTION_DEPTH = 0.0275
TASK_INSTRUCTION = "Align the needle with the hollow cylindrical opening and insert it"
IMG_WIDTH, IMG_HEIGHT = 640, 480


def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def randomize_phantom_pos(model_mj, data_mj, phantom_id, rot_id):
    offset_x = np.random.uniform(-0.05, 0.05)
    offset_y = np.random.uniform(-0.4, 0.0)
    offset_z = 0.0
    model_mj.body_pos[phantom_id] = np.array([offset_x, offset_y, offset_z])

    # Position-dependent rotation (matches Save_dataset.py)
    if offset_y >= -0.25:
        random_angle_deg = np.random.uniform(-15, 15)
    else:
        random_angle_deg = np.random.uniform(-15 - 90, 15 - 90)

    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model_mj.body_quat[rot_id] = new_quat

    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model_mj, data_mj)
    return np.array([offset_x, offset_y, offset_z], dtype=np.float32), new_quat.astype(np.float32)


def project_to_2d(point_3d, model_mj, data_mj, cam_name, img_w, img_h):
    """Project a 3D world point to normalized 2D image coordinates [0,1]."""
    cam_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data_mj.cam_xpos[cam_id]
    cam_mat = data_mj.cam_xmat[cam_id].reshape(3, 3)
    fovy = model_mj.cam_fovy[cam_id]
    p_cam = cam_mat.T @ (point_3d - cam_pos)
    f = img_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    u = -f * (p_cam[0] / p_cam[2]) + (img_w - 1) / 2.0
    v = f * (p_cam[1] / p_cam[2]) + (img_h - 1) / 2.0
    return np.array([u / img_w, v / img_h], dtype=np.float32)


class SimEnv:
    """Wraps the MuJoCo needle-insertion environment for closed-loop evaluation."""

    def __init__(self, model_xml_path: str, randomize: bool = True):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMG_HEIGHT, width=IMG_WIDTH)

        # site / body IDs
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        self.back_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        self.target_entry_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        self.target_depth_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        self.phantom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        self.rotating_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        self.link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        self.n_motors = self.model.nu
        self.dof = self.model.nv

        self.randomize = randomize

    # ── EE pose (matches Save_dataset.py) ─────────────────────────────────────
    def get_ee_pose(self):
        """Return (6,) array: [x, y, z] in mm, [rx, ry, rz] in rad."""
        pos = self.data.xpos[self.link6_id].copy() * 1000.0  # m → mm
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

    # ── Rendering ─────────────────────────────────────────────────────────────
    def render_cameras(self):
        """Return dict of RGB uint8 images {cam_name: (H, W, 3)}."""
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            self.renderer.update_scene(self.data, camera=cam_name)
            frames[cam_name] = self.renderer.render().copy()  # already RGB
        return frames

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        home_pose = np.array([
            np.random.uniform(-0.45, 0.55),
            np.random.uniform(-0.3, -0.4),
            np.random.uniform(0.3, 0.4),
            0.0,
            np.random.uniform(0.45, 0.55),
            np.random.uniform(0.95, 1.05),
        ])
        self.data.qpos[:6] = home_pose
        if self.randomize:
            randomize_phantom_pos(self.model, self.data, self.phantom_id, self.rotating_id)
        mujoco.mj_forward(self.model, self.data)

    # ── IK-based EE tracking controller ───────────────────────────────────────
    def apply_delta_ee(self, delta_ee_6d, n_sim_steps: int = 67, gain: float = 0.5):
        """
        Apply a predicted delta-EE (mm/rad) by tracking the target pose
        over *n_sim_steps* MuJoCo physics steps using resolved-rate IK.

        delta_ee_6d : (6,) — [dx, dy, dz, drx, dry, drz]  (mm / rad)
        """
        # Current EE → target EE (in metres / rad for MuJoCo)
        current_ee = self.get_ee_pose()                     # mm / rad
        target_ee = current_ee + delta_ee_6d                # mm / rad
        target_pos_m = target_ee[:3] / 1000.0               # → metres
        target_rpy = target_ee[3:]                          # rad

        for _ in range(n_sim_steps):
            # Current world-frame position (metres) & orientation
            cur_pos = self.data.xpos[self.link6_id].copy()
            cur_mat = self.data.xmat[self.link6_id].reshape(3, 3)

            # Position error (metres)
            err_pos = target_pos_m - cur_pos

            # Orientation error via rotation-matrix difference → axis-angle
            target_mat = self._rpy_to_rotmat(target_rpy)
            err_rot_mat = target_mat @ cur_mat.T
            err_rot = self._rotmat_to_axisangle(err_rot_mat)

            err = np.concatenate([err_pos * 50.0, err_rot * 10.0])  # P-gains

            # Jacobian of link6 body
            jac_pos = np.zeros((3, self.dof))
            jac_rot = np.zeros((3, self.dof))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.link6_id)
            J = np.vstack([jac_pos[:, :self.n_motors], jac_rot[:, :self.n_motors]])

            dq = np.linalg.pinv(J, rcond=1e-4) @ err
            self.data.ctrl[:self.n_motors] = self.data.qpos[:self.n_motors] + dq * gain
            mujoco.mj_step(self.model, self.data)

    # ── Success check ─────────────────────────────────────────────────────────
    def check_success(self, lateral_thresh=0.003, angle_thresh_deg=30.0):
        """
        Check insertion success with three conditions:
          1. Insertion depth along trocar axis >= TARGET_INSERTION_DEPTH
          2. Lateral distance from trocar axis < lateral_thresh (metres)
          3. Needle axis aligned with trocar axis within angle_thresh_deg
        """
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()

        # Trocar insertion axis
        axis = depth_pos - entry_pos
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            return False
        axis_dir = axis / axis_len

        # 1) Depth: projection of tip onto insertion axis
        tip_offset = tip_pos - entry_pos
        depth = np.dot(tip_offset, axis_dir)
        if depth < TARGET_INSERTION_DEPTH:
            return False

        # 2) Lateral distance: how far tip is from the insertion axis line
        projection = tip_offset - depth * axis_dir
        lateral_dist = np.linalg.norm(projection)
        if lateral_dist > lateral_thresh:
            return False

        # 3) Angle alignment: needle axis vs trocar axis
        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        if needle_len < 1e-8:
            return False
        needle_dir = needle_dir / needle_len

        cos_angle = abs(np.dot(needle_dir, axis_dir))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
        if angle_deg > angle_thresh_deg:
            return False

        return True

    def get_sensor_dist(self):
        """Get needle-tip sensor distance (mm), same as Save_dataset.py."""
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

    # ── Spatial metrics ────────────────────────────────────────────────────────
    def get_spatial_metrics(self):
        """Return a dict of spatial metrics for the current state."""
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()

        # 3D distance (mm)
        dist_mm = np.linalg.norm((entry_pos - tip_pos) * 1000.0)

        # Insertion axis
        axis = depth_pos - entry_pos
        axis_len = np.linalg.norm(axis)
        axis_dir = axis / (axis_len + 1e-10)

        # Depth along insertion axis (mm)
        tip_offset = tip_pos - entry_pos
        insertion_depth_mm = np.dot(tip_offset, axis_dir) * 1000.0

        # Lateral distance from axis (mm)
        projection = tip_offset - np.dot(tip_offset, axis_dir) * axis_dir
        lateral_mm = np.linalg.norm(projection) * 1000.0

        # Needle-trocar angle (degrees)
        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        if needle_len > 1e-8:
            needle_dir /= needle_len
            cos_angle = abs(np.dot(needle_dir, axis_dir))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
        else:
            angle_deg = 90.0

        # 2D projections on wrist camera
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

    # ── rotation helpers ──────────────────────────────────────────────────────
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
        """Convert rotation matrix to axis-angle (3-vector)."""
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
# Image preprocessing (match training distribution)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_image(img_rgb, target_size=(256, 256)):
    """
    JPEG encode→decode + resize, matching the RLDS/DROID training pipeline.
    img_rgb: (H, W, 3) uint8 RGB
    """
    # JPEG round-trip (same quality as data collection)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)

    # Resize with Lanczos
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
    return img_resized


# ═══════════════════════════════════════════════════════════════════════════════
# Video saving
# ═══════════════════════════════════════════════════════════════════════════════
def save_rollout_video(images, episode_idx, success, save_dir, fps=15):
    os.makedirs(save_dir, exist_ok=True)
    tag = "success" if success else "fail"
    path = os.path.join(save_dir, f"episode_{episode_idx:04d}_{tag}.mp4")
    writer = imageio.get_writer(path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()
    print(f"  Saved video: {path}")


def save_episode_plot(metrics_history, episode_idx, success, save_dir):
    """Save a multi-panel plot of spatial metrics over the episode."""
    os.makedirs(save_dir, exist_ok=True)
    tag = "SUCCESS" if success else "FAIL"
    steps = list(range(len(metrics_history)))

    dist = [m["dist_mm"] for m in metrics_history]
    depth = [m["insertion_depth_mm"] for m in metrics_history]
    lateral = [m["lateral_mm"] for m in metrics_history]
    angle = [m["angle_deg"] for m in metrics_history]

    has_spatial = any("spatial_pred" in m for m in metrics_history)
    nrows = 3 if has_spatial else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
    fig.suptitle(f"Episode {episode_idx} [{tag}]", fontsize=14, fontweight="bold")

    axes[0, 0].plot(steps, dist, "b-", linewidth=1.2, label="GT dist")
    if has_spatial:
        pred_steps = [i for i, m in enumerate(metrics_history) if "spatial_pred" in m]
        pred_dist = [m["spatial_pred"]["dist_norm"] * 100 for m in metrics_history if "spatial_pred" in m]
        axes[0, 0].plot(pred_steps, pred_dist, "r--", linewidth=1.0, alpha=0.7, label="Pred dist")
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("Distance (mm)")
    axes[0, 0].set_title("Needle Tip <-> Trocar Entry")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, depth, "g-", linewidth=1.2)
    axes[0, 1].axhline(y=TARGET_INSERTION_DEPTH * 1000, color="r", linestyle="--", label=f"target={TARGET_INSERTION_DEPTH*1000:.1f}mm")
    axes[0, 1].set_ylabel("Depth (mm)")
    axes[0, 1].set_title("Insertion Depth (along axis)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, lateral, "m-", linewidth=1.2)
    axes[1, 0].axhline(y=3.0, color="r", linestyle="--", label="thresh=3mm")
    axes[1, 0].set_ylabel("Lateral (mm)")
    axes[1, 0].set_title("Lateral Distance from Axis")
    axes[1, 0].set_xlabel("Control Step")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, angle, "c-", linewidth=1.2)
    axes[1, 1].axhline(y=30.0, color="r", linestyle="--", label="thresh=30deg")
    axes[1, 1].set_ylabel("Angle (deg)")
    axes[1, 1].set_title("Needle-Trocar Axis Angle")
    axes[1, 1].set_xlabel("Control Step")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    if has_spatial:
        pred_steps = [i for i, m in enumerate(metrics_history) if "spatial_pred" in m]
        # Visibility predictions
        tip_vis = [m["spatial_pred"]["tip_visible"] for m in metrics_history if "spatial_pred" in m]
        tro_vis = [m["spatial_pred"]["trocar_visible"] for m in metrics_history if "spatial_pred" in m]
        gt_tip_vis = []
        gt_tro_vis = []
        for i in pred_steps:
            m = metrics_history[i]
            t_uv = m["tip_uv"]
            r_uv = m["trocar_uv"]
            gt_tip_vis.append(1.0 if (0 <= t_uv[0] <= 1 and 0 <= t_uv[1] <= 1) else 0.0)
            gt_tro_vis.append(1.0 if (0 <= r_uv[0] <= 1 and 0 <= r_uv[1] <= 1) else 0.0)

        axes[2, 0].plot(pred_steps, tip_vis, "r-", linewidth=1.2, label="Pred tip_vis")
        axes[2, 0].plot(pred_steps, gt_tip_vis, "r--", linewidth=0.8, alpha=0.5, label="GT tip_vis")
        axes[2, 0].plot(pred_steps, tro_vis, "g-", linewidth=1.2, label="Pred tro_vis")
        axes[2, 0].plot(pred_steps, gt_tro_vis, "g--", linewidth=0.8, alpha=0.5, label="GT tro_vis")
        axes[2, 0].set_ylabel("Visibility (prob)")
        axes[2, 0].set_title("Visibility Predictions vs GT")
        axes[2, 0].set_xlabel("Control Step")
        axes[2, 0].set_ylim(-0.1, 1.1)
        axes[2, 0].legend(fontsize=7)
        axes[2, 0].grid(True, alpha=0.3)

        # Phase predictions
        phase_pred = [m["spatial_pred"]["phase"] for m in metrics_history if "spatial_pred" in m]
        axes[2, 1].plot(pred_steps, phase_pred, "b-", linewidth=1.2, label="Pred phase")
        axes[2, 1].set_ylabel("Phase (prob)")
        axes[2, 1].set_title("Phase Prediction (0=align, 1=insert)")
        axes[2, 1].set_xlabel("Control Step")
        axes[2, 1].set_ylim(-0.1, 1.1)
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"episode_{episode_idx:04d}_{tag.lower()}_metrics.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def draw_overlay(frame, metrics, ctrl_step):
    """Draw spatial metrics as text overlay on a video frame (in-place)."""
    h, w = frame.shape[:2]
    lines = [
        f"step={ctrl_step}",
        f"dist={metrics['dist_mm']:.1f}mm",
        f"depth={metrics['insertion_depth_mm']:.1f}mm",
        f"lateral={metrics['lateral_mm']:.1f}mm",
        f"angle={metrics['angle_deg']:.1f}deg",
    ]
    sp = metrics.get("spatial_pred")
    if sp is not None:
        lines.append(f"pred: tip_vis={sp['tip_visible']:.2f} tro_vis={sp['trocar_visible']:.2f}")
        lines.append(f"pred: dist={sp['dist_norm']*100:.1f}mm phase={sp['phase']:.2f}")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (5, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (5, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # (keypoint visualization removed — spatial head no longer predicts u,v coordinates)

    # Draw GT keypoints on wrist panel (cyan/yellow circles)
    tip_uv = metrics.get("tip_uv")
    trocar_uv = metrics.get("trocar_uv")
    if tip_uv is not None:
        panel_w = w // 3
        wrist_offset_x = panel_w
        if 0.0 <= tip_uv[0] <= 1.0 and 0.0 <= tip_uv[1] <= 1.0:
            px = int(tip_uv[0] * panel_w) + wrist_offset_x
            py = int(tip_uv[1] * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 255), 1)  # cyan = GT tip
        if 0.0 <= trocar_uv[0] <= 1.0 and 0.0 <= trocar_uv[1] <= 1.0:
            px = int(trocar_uv[0] * panel_w) + wrist_offset_x
            py = int(trocar_uv[1] * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 255), 1)  # cyan = GT trocar

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_eval(cfg):
    checkpoint_path = cfg.eval.finetuned_checkpoint
    assert checkpoint_path, "eval.finetuned_checkpoint must be set!"

    set_seed(cfg.eval.seed)

    # ── Load model & processor ────────────────────────────────────────────────
    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)
    model = load_model(checkpoint_path, diffusion_steps=diff_steps, scheduler_type=sched_type, train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    image_size = getattr(cfg.eval, "image_size", 256)
    num_episodes = getattr(cfg.eval, "num_episodes", 50)
    max_steps = getattr(cfg.eval, "max_steps_per_episode", 600)
    num_steps_execute = getattr(cfg.eval, "num_steps_execute", 1)
    sim_steps_per_ctrl = getattr(cfg.eval, "sim_steps_per_control", 67)
    do_randomize = getattr(cfg.eval, "randomize_phantom_pos", True)
    save_video = getattr(cfg.eval, "save_video", True)
    video_fps = getattr(cfg.eval, "video_fps", 15)

    # ── Output directory ──────────────────────────────────────────────────────
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1]
    except (ValueError, IndexError):
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"sim_eval_step{step_str}_exec{num_steps_execute}_diff{diff_steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    # ── Environment ───────────────────────────────────────────────────────────
    model_xml = os.path.abspath(SIM_MODEL_PATH)
    env = SimEnv(model_xml, randomize=do_randomize)

    total_successes = 0

    # CSV for all episodes summary
    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "success", "steps", "final_dist_mm", "final_depth_mm",
                         "final_lateral_mm", "final_angle_deg", "min_dist_mm"])

    for ep in range(1, num_episodes + 1):
        env.reset()
        last_ee_pose = env.get_ee_pose()

        image_history = []
        image_history_wrist = []
        image_history_top = []
        state_history = []
        action_history = []
        action_buffer = []
        replay_images = []
        metrics_history = []

        success = False
        last_phase_pred = 0.0  # track phase prediction for gripper state

        for ctrl_step in range(max_steps):
            # ── 1. Observe ────────────────────────────────────────────────────
            frames = env.render_cameras()
            img_ext = preprocess_image(frames["side_camera"], (image_size, image_size))
            img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))
            img_top = preprocess_image(frames["top_camera"], (image_size, image_size))

            image_history.append(img_ext)
            image_history_wrist.append(img_wrist)
            image_history_top.append(img_top)

            # Collect spatial metrics
            metrics = env.get_spatial_metrics()
            metrics_history.append(metrics)

            # Concat all 3 views side-by-side for replay video (overlay drawn after prediction)
            replay_frame = np.concatenate([img_ext, img_wrist, img_top], axis=1)  # (H, W*3, 3)

            # Proprioception: ee_pose(6) + gripper_proxy(1)
            # gripper: matches training data (phase >= 2 → 1.0, else 0.0)
            # At eval time, use spatial head's phase prediction as proxy
            ee_pose = env.get_ee_pose()  # (6,)
            sensor_dist = env.get_sensor_dist()
            gripper_state = 1.0 if last_phase_pred > 0.5 else 0.0
            proprio = np.concatenate([ee_pose, [gripper_state]])  # (7,)
            state_history.append(proprio)

            # Proximity info → instruction text (matches training)
            proximity = (0 <= sensor_dist < 10)
            if proximity:
                task_instruction = TASK_INSTRUCTION + ". Object detected nearby"
            else:
                task_instruction = TASK_INSTRUCTION

            observation = {
                "full_image": img_ext,
                "full_image_wrist": img_wrist,
                "full_image_top": img_top,
                "image_history": image_history,
                "image_history_wrist": image_history_wrist,
                "image_history_top": image_history_top,
                "state_history": state_history,
                "action_history": action_history,
            }

            # ── 2. Get action from model (or from buffer) ────────────────────
            spatial_pred = None
            if len(action_buffer) == 0:
                raw_chunk, spatial_pred = predict_action(model, processor, observation, task_instruction)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk[None, :]
                steps_exec = min(num_steps_execute, len(raw_chunk))
                action_buffer = list(raw_chunk[:steps_exec])
            if spatial_pred is not None:
                last_phase_pred = spatial_pred["phase"]
                metrics["spatial_pred"] = spatial_pred
                if ctrl_step % 100 == 0:
                    print(f"  [step {ctrl_step}] spatial_pred: "
                          f"tip_vis={spatial_pred['tip_visible']:.3f} tro_vis={spatial_pred['trocar_visible']:.3f} "
                          f"dist={spatial_pred['dist_norm']:.3f} phase={spatial_pred['phase']:.3f}")

            # Draw overlay AFTER spatial prediction is available
            draw_overlay(replay_frame, metrics, ctrl_step)
            replay_images.append(replay_frame)

            raw_action = action_buffer.pop(0)  # (7,) normalized [-1, 1]
            action_history.append(raw_action)

            # ── 3. Denormalize & apply action ─────────────────────────────────
            # Model outputs [-1, 1] → convert back to mm/rad
            a_min = np.array(action_min_sim, dtype=np.float32)
            a_max = np.array(action_max_sim, dtype=np.float32)
            denorm_action = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
            delta_ee = denorm_action[:6]   # (dx, dy, dz, drx, dry, drz) in mm/rad
            # 7th dim (gripper) is ignored — no physical gripper

            env.apply_delta_ee(delta_ee, n_sim_steps=sim_steps_per_ctrl)

            # Update last_ee_pose for logging
            last_ee_pose = env.get_ee_pose()

            # ── 4. Check success ──────────────────────────────────────────────
            if env.check_success():
                success = True
                # Collect final metrics after success
                metrics_history.append(env.get_spatial_metrics())
                break

        if success:
            total_successes += 1

        sr = total_successes / ep * 100
        final_m = metrics_history[-1]
        min_dist = min(m["dist_mm"] for m in metrics_history)
        msg = (f"Episode {ep}/{num_episodes} | {'SUCCESS' if success else 'FAIL'} | "
               f"Steps: {ctrl_step + 1} | SR: {sr:.1f}% ({total_successes}/{ep}) | "
               f"dist={final_m['dist_mm']:.1f}mm lateral={final_m['lateral_mm']:.1f}mm "
               f"angle={final_m['angle_deg']:.1f}deg min_dist={min_dist:.1f}mm")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        # Write CSV row
        csv_writer.writerow([
            ep, int(success), ctrl_step + 1,
            f"{final_m['dist_mm']:.2f}", f"{final_m['insertion_depth_mm']:.2f}",
            f"{final_m['lateral_mm']:.2f}", f"{final_m['angle_deg']:.2f}",
            f"{min_dist:.2f}",
        ])
        csv_file.flush()

        # Save per-episode metrics plot
        save_episode_plot(metrics_history, ep, success, str(eval_dir))

        if save_video:
            save_rollout_video(replay_images, ep, success, str(eval_dir), fps=video_fps)

    csv_file.close()
    print(f"  Metrics CSV: {csv_path}")

    # ── Final report ──────────────────────────────────────────────────────────
    final_sr = total_successes / num_episodes * 100
    summary = f"\n{'='*60}\nFinal Success Rate: {final_sr:.2f}% ({total_successes}/{num_episodes})\n{'='*60}"
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()

    # Rename directory to include success rate
    new_dir = eval_dir.parent / f"{eval_dir.name}_SR{final_sr:.2f}"
    try:
        eval_dir.rename(new_dir)
        print(f"Results saved to: {new_dir}")
    except Exception as e:
        print(f"Could not rename directory: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLANeXt on needle-insertion sim")
    parser.add_argument("--config", type=str, default="config/sim_eval_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Override eval.finetuned_checkpoint")
    parser.add_argument("--train-config", type=str, default=None, help="Path to train config (for model architecture)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if args.checkpoint:
        config_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint

    cfg = DictConfig(config_dict)
    cfg.train_config_path = args.train_config
    run_eval(cfg)
