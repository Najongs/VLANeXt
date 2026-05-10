"""
sim_eval_lerobot.py

Evaluate a lerobot-trained policy (ACT / Diffusion / VQ-BeT) on the fine-align
sim used by sim_eval_align_only.py. Reuses AlignSimEnv + perturb grid + success
criterion verbatim — only the policy inference call differs.

Usage:
    python -m scripts.sim_eval_lerobot \
        --policy act \
        --checkpoint /path/to/lerobot/output_dir/checkpoints/last/pretrained_model \
        --max-steps 250 --eval-seed 2026 \
        --perturb-mode grid --xy-steps 3 --z-steps 2 --angle-steps 3 --repeats 1

Notes:
- `--checkpoint` should point at the lerobot saved policy directory containing
  `config.json`, `model.safetensors`, plus `policy_preprocessor.json` and
  `policy_postprocessor.json`. Lerobot's `lerobot-train` writes this layout.
- Observation key convention (lerobot v0.5):
      observation.state          : (B, 6)           float32
      observation.images.tool_camera : (B, 3, H, W) float32 in [0, 1]
- Action returned by `postprocessor(policy.select_action(...))` is in the same
  units/convention as the dataset action (Mecademic intrinsic XYZ delta, mm + rad).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import pathlib
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.sim_eval import (
    SIM_MODEL_PATH,
    save_rollout_video, draw_overlay, set_seed,
)
from scripts.sim_eval_align_only import (
    AlignSimEnv, build_perturb_grid, _save_trajectory_plot,
    TASK_INSTRUCTION,
    SENSOR_STOP_CLOSE_MM, SENSOR_STOP_HOLE_MM, SENSOR_STOP_HOLD_STEPS,
)
from src.datasets.euler_convention import (
    mujoco_to_mecademic_euler, mecademic_to_mujoco_euler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Lerobot policy + processors loading (v0.5)
# ─────────────────────────────────────────────────────────────────────────────
def _import_policy_class(policy_name: str):
    name = policy_name.lower()
    if name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    if name in ("dp", "diffusion"):
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    if name in ("vqbet", "vq_bet"):
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        return VQBeTPolicy
    raise ValueError(f"Unknown lerobot policy: {policy_name}")


def load_lerobot_policy(policy_name: str, checkpoint: str, device: str = "cuda"):
    """Load policy + pre/post processor pipelines from a lerobot checkpoint dir."""
    from lerobot.policies import make_pre_post_processors

    Policy = _import_policy_class(policy_name)
    policy = Policy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
    )
    return policy, preprocessor, postprocessor


def _build_observation(img_rgb_uint8: np.ndarray, state6: np.ndarray, device: str):
    """Build a one-batch lerobot observation dict (pre-processor input).

    Image must be uint8 (H, W, 3); preprocessor converts to float and normalizes.
    State is float32 (6,).
    """
    img_t = torch.from_numpy(img_rgb_uint8).to(device)               # (H, W, 3) uint8
    img_t = img_t.permute(2, 0, 1).contiguous().float() / 255.0      # (3, H, W) [0, 1]
    state_t = torch.from_numpy(state6.astype(np.float32)).to(device) # (6,)
    return {
        "observation.images.tool_camera": img_t.unsqueeze(0),
        "observation.state": state_t.unsqueeze(0),
        "task": [TASK_INSTRUCTION],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(args):
    seed = args.eval_seed if args.eval_seed is not None else 2026
    if args.shard_id is not None:
        seed = seed + args.shard_id * 1000
    set_seed(seed); np.random.seed(seed); random.seed(seed)
    print(f"[seed] eval seed = {seed} (shard_id={args.shard_id})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_lerobot_policy(
        args.policy, args.checkpoint, device=device
    )

    image_size = args.image_size
    img_h = args.image_h
    img_w = args.image_w
    max_steps = args.max_steps
    sim_steps_per_ctrl = args.sim_steps_per_control
    save_video = not args.no_video

    if args.perturb_mode == "grid":
        grid_cells_all = build_perturb_grid(
            xy_steps=args.xy_steps, z_steps=args.z_steps,
            angle_steps=args.angle_steps, repeats=args.repeats,
        )
        num_episodes = len(grid_cells_all)
        print(f"[grid] {num_episodes} cells")
    else:
        grid_cells_all = None
        num_episodes = args.num_episodes

    all_episodes = list(range(1, num_episodes + 1))
    if args.shard_id is not None and args.num_shards is not None:
        all_episodes = [ep for ep in all_episodes if (ep - 1) % args.num_shards == args.shard_id]
        shard_suffix = f"_shard{args.shard_id}"
    else:
        shard_suffix = ""
    ep_to_cell = (
        {ep: grid_cells_all[ep - 1] for ep in all_episodes}
        if grid_cells_all is not None else {ep: None for ep in all_episodes}
    )

    ckpt_path = pathlib.Path(args.checkpoint)
    eval_dir = ckpt_path.parent / f"lerobot_{args.policy}_eval_{ckpt_path.name}{shard_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "success", "success_reason", "steps", "final_dist_mm",
        "final_lateral_mm", "final_angle_deg", "min_dist_mm", "final_sensor_dist_mm",
        "perturb_x_mm", "perturb_y_mm", "perturb_z_mm",
        "perturb_angle_deg", "perturb_dist_mm", "initial_dist_mm",
        "cell_idx", "repeat_id",
    ])

    env = AlignSimEnv(
        os.path.abspath(SIM_MODEL_PATH),
        randomize_phantom=False, use_sensor_success=False,
        phantom_pos=None, retreat_mm=args.retreat_mm,
    )

    total_successes = 0
    use_sensor_stop = args.sensor_stop

    for ep in all_episodes:
        env.reset(grid_cell=ep_to_cell[ep])
        if hasattr(policy, "reset"):
            policy.reset()

        replay_images = []
        metrics_history = []
        success = False
        success_reason = ""
        sensor_was_close = False
        sensor_spike_count = 0
        ctrl_step = 0

        for ctrl_step in range(max_steps):
            frames = env.render_cameras()
            tw = frames["tool_camera"]
            if tw.shape[0] != img_h or tw.shape[1] != img_w:
                tw = cv2.resize(tw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
            # Match training-side preprocessing (dataset/convert_to_lerobot.py): JPEG roundtrip
            # so train/eval image distributions align.
            _bgr = cv2.cvtColor(tw, cv2.COLOR_RGB2BGR)
            _ok, _enc = cv2.imencode(".jpg", _bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _bgr2 = cv2.imdecode(_enc, cv2.IMREAD_COLOR)
            img_wrist = cv2.cvtColor(_bgr2, cv2.COLOR_BGR2RGB)
            metrics = env.get_spatial_metrics()
            metrics_history.append(metrics)

            ee_pose = env.get_ee_pose()
            ee_pose_mec = ee_pose.copy()
            ee_pose_mec[3:6] = mujoco_to_mecademic_euler(ee_pose[3:6])
            state6 = ee_pose_mec[:6].astype(np.float32)

            # Replay video: tile at image_size×image_size each (wrist resized down for viz only).
            wrist_thumb = cv2.resize(img_wrist, (image_size, image_size), interpolation=cv2.INTER_AREA)
            replay_frame = np.concatenate(
                [
                    cv2.resize(frames["side_camera"], (image_size, image_size)),
                    wrist_thumb,
                    cv2.resize(frames["top_camera"], (image_size, image_size)),
                ],
                axis=1,
            )

            with torch.inference_mode():
                obs = _build_observation(img_wrist, state6, device)
                obs = preprocessor(obs)
                action = policy.select_action(obs)
                action = postprocessor(action)
            action_np = action.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
            assert action_np.shape[0] == 6, f"expected 6-DoF action, got {action_np.shape}"

            target_mec_rpy = ee_pose_mec[3:6] + action_np[3:6]
            target_mujoco_rpy = mecademic_to_mujoco_euler(target_mec_rpy)
            delta_rpy_mujoco = target_mujoco_rpy - ee_pose[3:6]
            delta_rpy_mujoco = np.arctan2(np.sin(delta_rpy_mujoco), np.cos(delta_rpy_mujoco))
            delta_ee = np.concatenate([action_np[:3], delta_rpy_mujoco]).astype(np.float32)

            env.apply_delta_ee(delta_ee, n_sim_steps=sim_steps_per_ctrl)
            draw_overlay(replay_frame, metrics, ctrl_step)
            replay_images.append(replay_frame)

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

        if success:
            total_successes += 1
        sr = total_successes / (all_episodes.index(ep) + 1) * 100
        final_m = metrics_history[-1]
        cell = ep_to_cell.get(ep)
        suffix = "S" if success else "F"
        if save_video and replay_images:
            save_rollout_video(
                replay_images,
                episode_idx=ep,
                success=success,
                save_dir=str(eval_dir),
                fps=15,
            )

        csv_writer.writerow([
            ep, int(success), success_reason, ctrl_step + 1,
            final_m.get("dist_mm", 0.0),
            final_m.get("lateral_mm", 0.0),
            final_m.get("angle_deg", 0.0),
            min(m.get("dist_mm", 1e9) for m in metrics_history),
            final_m.get("sensor_dist_mm", -1.0),
            cell["x_mm"] if cell else 0.0,
            cell["y_mm"] if cell else 0.0,
            cell["z_mm"] if cell else 0.0,
            cell["angle_deg"] if cell else 0.0,
            float(np.linalg.norm([cell["x_mm"], cell["y_mm"], cell["z_mm"]])) if cell else 0.0,
            metrics_history[0].get("dist_mm", 0.0),
            cell["cell_idx"] if cell else -1,
            cell["repeat_id"] if cell else -1,
        ])
        csv_file.flush()
        print(f"[ep {ep:03d} {suffix}] dist={final_m.get('dist_mm', 0):.2f}mm "
              f"angle={final_m.get('angle_deg', 0):.2f}°  SR={sr:.1f}%")

    csv_file.close()
    _save_trajectory_plot(eval_dir)
    print(f"\nDone. {total_successes}/{len(all_episodes)} success "
          f"({100 * total_successes / max(1, len(all_episodes)):.1f}%)")
    print(f"Output: {eval_dir}")


def main():
    p = argparse.ArgumentParser(description="Eval lerobot policy on fine-align sim")
    p.add_argument("--policy", type=str, required=True, choices=["act", "dp", "diffusion", "vqbet"])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Lerobot saved policy dir (config.json + model.safetensors + processor jsons)")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256, help="Side-view tile size for replay video only.")
    p.add_argument("--image-h", type=int, default=480, help="Policy input image height (must match training).")
    p.add_argument("--image-w", type=int, default=640, help="Policy input image width (must match training).")
    p.add_argument("--sim-steps-per-control", type=int, default=67)
    p.add_argument("--retreat-mm", type=float, default=10.0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--sensor-stop", action="store_true")
    p.add_argument("--perturb-mode", type=str, default="grid", choices=["grid", "random"])
    p.add_argument("--xy-steps", type=int, default=3)
    p.add_argument("--z-steps", type=int, default=2)
    p.add_argument("--angle-steps", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
