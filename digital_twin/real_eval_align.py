"""
real_eval_align.py

Real-world deployment of a fine-alignment phase VLANeXt policy.
Mirrors scripts/sim_eval_align_only.py but applies inferred actions to a real
Mecademic Meca500 robot with OAK cameras instead of MuJoCo.

Differences from real_eval_approach.py:
  - Loads action_min/max from sim_act_align (tighter ±0.6mm / ±0.003rad ranges)
  - Primary view = tool_camera (cam_exterior in sim_train_align_config.yaml)
  - Different task instruction
  - Optional sensor_dist proprio (substituted with 20.0 since real has no
    MuJoCo ray-cast — wire OCT/FPI here later if you have a real proxy)
  - Default max_steps = 200 (align is shorter than approach)
  - Recommends `--skip-home` so you can manually pre-position the needle near
    the trocar before inference starts (sim's pre-align + perturbation pipeline
    has no real-world counterpart)

Usage:
    python -m digital_twin.real_eval_align \
        --config config/sim_eval_align_config.yaml \
        --checkpoint /path/to/checkpoint \
        --train-config config/sim_train_align_config.yaml \
        --max-steps 200 [--skip-home] [--use-sensor] [--dry-run] [--swap-cameras]
"""

import os
import sys
import time
import argparse
import logging
import pathlib

import yaml
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, set_seed,
)
from src.datasets.sim_act_align import (
    action_min_sim_align as action_min_sim,
    action_max_sim_align as action_max_sim,
)

# Reuse robot + camera wrapper from approach script
from digital_twin.real_eval_approach import (
    OAKCameraManager, ApproachRealEnv as RealRobotEnv,
    ROBOT_ADDRESS_DEFAULT, HOME_JOINTS,
    IMG_WIDTH, IMG_HEIGHT,
)

logger = logging.getLogger(__name__)

# Different task + bigger constants for align
TASK_INSTRUCTION = (
    "Align the needle tip to the small grey circular trocar port on the eye model, "
    "next to the larger lens opening"
)

# sensor_dist substitute when use_sensor=True but real sensor not wired (saturated/no-contact)
SENSOR_DIST_FALLBACK_MM = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop (mirrors run_eval in sim_eval_align_only.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_real_eval(cfg):
    checkpoint_path = cfg.eval.finetuned_checkpoint
    assert checkpoint_path, "eval.finetuned_checkpoint must be set!"
    set_seed(cfg.eval.seed)

    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)

    model = load_model(checkpoint_path, diffusion_steps=diff_steps,
                       scheduler_type=sched_type, train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    image_size = getattr(cfg.eval, "image_size", 256)
    num_steps_execute = getattr(cfg.eval, "num_steps_execute", 1)
    video_fps = getattr(cfg.eval, "video_fps", 15)
    save_video = getattr(cfg.eval, "save_video", True)

    # use_sensor can come from eval-config model section OR CLI override
    use_sensor = bool(getattr(cfg.model, "use_sensor", False) or getattr(cfg, "use_sensor", False))
    if use_sensor:
        logger.warning(f"⚠️ use_sensor=True: substituting sensor_dist={SENSOR_DIST_FALLBACK_MM}mm "
                       f"(real OCT/FPI not wired — model proprio dim must be 8)")

    num_episodes = int(getattr(cfg, "num_episodes", 1))
    max_steps = int(getattr(cfg, "max_steps", 200))
    control_dt = 1.0 / float(video_fps)  # ~67ms at 15 Hz

    # Output dir alongside checkpoint
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1] if ckpt_path.stem else "unknown"
    except Exception:
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"real_align_step{step_str}_max{max_steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    logger.info(f"📁 Output dir: {eval_dir}")

    env = RealRobotEnv(
        robot_address=getattr(cfg, "robot_address", ROBOT_ADDRESS_DEFAULT),
        swap_cameras=getattr(cfg, "swap_cameras", False),
        skip_home=getattr(cfg, "skip_home", False),
        joint_vel_limit=getattr(cfg, "joint_vel_limit", None),
    )
    dry_run = getattr(cfg, "dry_run", False)
    if dry_run:
        logger.warning("⚠️ DRY-RUN: MovePose calls disabled; targets only printed")

    a_min = np.array(action_min_sim, dtype=np.float32)
    a_max = np.array(action_max_sim, dtype=np.float32)

    try:
        for ep in range(1, num_episodes + 1):
            logger.info("\n" + "=" * 60)
            logger.info(f"▶ Align Episode {ep}/{num_episodes}")
            logger.info("=" * 60)
            log_file.write(f"\n=== Episode {ep}/{num_episodes} ===\n")

            env.reset()

            image_history = []
            image_history_wrist = []
            state_history = []
            action_history = []
            action_buffer = []
            replay_images = []
            ee_traj = []
            user_quit = False
            ctrl_step = 0

            for ctrl_step in range(max_steps):
                # 1. Render
                frames = env.render_cameras()
                img_top = preprocess_image(frames["top_camera"], (image_size, image_size))
                img_tool = preprocess_image(frames["tool_camera"], (image_size, image_size))

                # 2. Map: align uses tool_camera as primary (cam_exterior in train config)
                #    matches sim_eval_align_only.py:701 (`img_primary = img_wrist`)
                img_primary = img_tool
                img_secondary = img_top  # not consumed (view_mode=single), kept for replay
                image_history.append(img_primary)
                image_history_wrist.append(img_secondary)

                # 3. Proprio (7D or 8D depending on use_sensor)
                ee_pose = env.get_ee_pose()
                ee_traj.append(ee_pose[:3].copy())
                if use_sensor:
                    proprio = np.concatenate([ee_pose, [0.0], [SENSOR_DIST_FALLBACK_MM]])  # (8,)
                else:
                    proprio = np.concatenate([ee_pose, [0.0]])  # (7,)
                state_history.append(proprio)

                observation = {
                    "full_image": img_primary,
                    "full_image_wrist": img_secondary,
                    "image_history": image_history,
                    "image_history_wrist": image_history_wrist,
                    "state_history": state_history,
                    "action_history": action_history,
                }

                # 4. Inference
                if len(action_buffer) == 0:
                    raw_chunk, _ = predict_action(model, processor, observation, TASK_INSTRUCTION)
                    if raw_chunk.ndim == 1:
                        raw_chunk = raw_chunk[None, :]
                    steps_exec = min(num_steps_execute, len(raw_chunk))
                    action_buffer = list(raw_chunk[:steps_exec])

                raw_action = action_buffer.pop(0)
                action_history.append(raw_action)

                # 5. Denormalize
                denorm = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
                delta_ee = denorm[:6]

                # 6. Apply
                target = env.apply_delta_ee(delta_ee, control_dt=control_dt, dry_run=dry_run)

                # 7. Replay overlay
                # Note: tool first (primary), then top, to highlight what model sees
                replay_frame = np.concatenate([img_tool, img_top], axis=1)
                cv2.putText(replay_frame, f"step {ctrl_step+1}/{max_steps} [ALIGN]", (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(replay_frame,
                            f"dpos_mm={delta_ee[:3].round(2).tolist()}",
                            (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(replay_frame,
                            f"drot_deg={np.rad2deg(delta_ee[3:]).round(3).tolist()}",
                            (5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                if dry_run:
                    cv2.putText(replay_frame, "DRY-RUN", (5, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
                replay_images.append(replay_frame.copy())

                try:
                    cv2.imshow("Real Align Eval (tool | top) — press 'q' to abort",
                               cv2.cvtColor(replay_frame, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.warning("🛑 'q' pressed — aborting episode")
                        user_quit = True
                        break
                except cv2.error:
                    pass

                if (ctrl_step + 1) % 10 == 0:
                    msg = (f"  step {ctrl_step+1:3d}: "
                           f"dpos={delta_ee[:3].round(2).tolist()} "
                           f"drot_deg={np.rad2deg(delta_ee[3:]).round(3).tolist()} "
                           f"target={[f'{v:.1f}' for v in (target or [])]}")
                    logger.info(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

            # End of episode
            done_msg = f"  Episode {ep} ended after {ctrl_step + 1} steps (user_quit={user_quit})"
            logger.info(done_msg)
            log_file.write(done_msg + "\n")
            log_file.flush()

            ee_arr = np.stack(ee_traj) if ee_traj else np.zeros((0, 3), dtype=np.float32)
            np.savez_compressed(eval_dir / f"traj_ep{ep:03d}.npz", ee_pose=ee_arr)

            if save_video and replay_images:
                save_rollout_video(replay_images, ep, success=False,
                                   save_dir=str(eval_dir), fps=video_fps)

            if user_quit:
                break

    except KeyboardInterrupt:
        logger.warning("\n🛑 Interrupted (Ctrl-C)")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
        log_file.close()
        logger.info(f"📁 Results: {eval_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-robot eval for VLANeXt fine-alignment policy")
    parser.add_argument("--config", type=str, default="config/sim_eval_align_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-config", type=str, default="config/sim_train_align_config.yaml")
    parser.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--swap-cameras", action="store_true",
                        help="Swap camera1↔camera2 mapping")
    parser.add_argument("--skip-home", action="store_true",
                        help="Don't MoveJoints to HOME — use whatever pose the robot is in. "
                             "Recommended for align: manually pre-position needle near trocar first.")
    parser.add_argument("--joint-vel-limit", type=float, default=None,
                        help="Mecademic SetJointVelLimit (deg/s). Lower = slower + smoother. "
                             "Combine with larger --max-steps for slower trajectory. e.g. 5")
    parser.add_argument("--use-sensor", action="store_true",
                        help="Add 8th proprio dim with constant 20.0 (real sensor_dist proxy). "
                             "Required if checkpoint was trained with use_sensor=True.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip MovePose; just print targets (safe smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint

    cfg = DictConfig(cfg_dict)
    cfg.train_config_path = args.train_config
    cfg.robot_address = args.robot_address
    cfg.max_steps = args.max_steps
    cfg.num_episodes = args.num_episodes
    cfg.swap_cameras = args.swap_cameras
    cfg.skip_home = args.skip_home
    cfg.use_sensor = args.use_sensor
    cfg.joint_vel_limit = args.joint_vel_limit
    cfg.dry_run = args.dry_run

    run_real_eval(cfg)
