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

2026-05-18 status:
  - **VLA-only**. handoff/KP head 미구현 (sensor_handoff.py가 sim env API에 의존).
    sim의 close_5mm 66.7% 같은 수치는 handoff 포함 결과 — real에선 그 보정 없음.
  - 추천 ckpt:
    A) frozen baseline (안정): HOLD_focus_v2/checkpoint_3000.pt
       + config/sim_train_align_siglip2_b24_ft10mm_HOLD_focus_v2_config.yaml
    B) unfreeze last4 (sim 최고, seed lottery): unfreeze_last4/checkpoint_1000.pt
       + config/sim_train_align_siglip2_b24_ft10mm_unfreeze_last4_config.yaml
  - ckpt와 train-config는 반드시 짝이 맞아야 함 (architecture mismatch 방지).

Usage:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m digital_twin.real_eval_align \
    --config config/sim_eval_align_config.yaml \
    --checkpoint checkpoints/checkpoint_1500.pt \
    --train-config config/sim_train_align_qwen_reach_recover_v2_aggressive_config.yaml \
    --max-steps 200 \
    --start-joints "43, -28, 28, 0, 34, 57" \
    --joint-vel-limit 5 \
    --converge-thresh-mm 0.3 \
    --num-steps-execute 4
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

# sensor_dist substitute when use_sensor=True but real sensor not wired.
# Two-channel format matches dataset/sim eval (src/utils/sensor_proc.py):
#   [dist_clipped (mm, ≤5), valid (1.0 if informative else 0.0)]
# No real sensor → valid=0, dist=SENSOR_MAX_MM (5.0) so the model sees a clean "no info" signal.
from src.utils.sensor_proc import SENSOR_MAX_MM, process_sensor_dist_scalar
SENSOR_FALLBACK_DIST_MM = SENSOR_MAX_MM
SENSOR_FALLBACK_VALID = 0.0


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
        logger.warning(f"⚠️ use_sensor=True: substituting [dist={SENSOR_FALLBACK_DIST_MM}mm, valid={SENSOR_FALLBACK_VALID}] "
                       f"(real sensor not wired — model proprio dim must be 9)")

    num_episodes = int(getattr(cfg, "num_episodes", 1))
    max_steps = int(getattr(cfg, "max_steps", 200))
    control_dt = 1.0 / float(video_fps)  # ~67ms at 15 Hz

    # Convergence-based early stop (action magnitude self-report).
    # When mean(||dpos[:3]||) over last K steps drops below threshold, model declares "done".
    converge_thresh_mm = float(getattr(cfg, "converge_thresh_mm", 0.0))
    converge_window = int(getattr(cfg, "converge_window", 5))
    converge_min_step = int(getattr(cfg, "converge_min_step", 30))
    if converge_thresh_mm > 0:
        logger.info(f"🛑 Convergence stop: mean(|dpos|) < {converge_thresh_mm:.2f}mm "
                    f"over last {converge_window} steps (after step {converge_min_step})")

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
        start_joints=getattr(cfg, "start_joints", None),
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
            dpos_norms_mm = []
            converged = False
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

                # 3. Proprio: 6-DoF EE pose only (gripper dropped, sensor detection-only)
                ee_pose = env.get_ee_pose()
                ee_traj.append(ee_pose[:3].copy())
                proprio = ee_pose[:6].astype(np.float32)  # (6,)
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

                # 6b. Convergence check (action magnitude self-report)
                dpos_mm = float(np.linalg.norm(delta_ee[:3]) * 1000.0)
                dpos_norms_mm.append(dpos_mm)
                if (converge_thresh_mm > 0
                        and ctrl_step + 1 >= converge_min_step
                        and len(dpos_norms_mm) >= converge_window):
                    recent_mean = float(np.mean(dpos_norms_mm[-converge_window:]))
                    if recent_mean < converge_thresh_mm:
                        msg = (f"  ✓ converged at step {ctrl_step+1}: "
                               f"mean|dpos| over last {converge_window} = {recent_mean:.3f}mm "
                               f"< {converge_thresh_mm:.2f}mm")
                        logger.info(msg)
                        log_file.write(msg + "\n"); log_file.flush()
                        converged = True
                        break

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
            done_msg = (f"  Episode {ep} ended after {ctrl_step + 1} steps "
                        f"(user_quit={user_quit}, converged={converged})")
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
    parser.add_argument("--train-config", type=str,
                        default="config/sim_train_align_siglip2_b24_ft10mm_HOLD_focus_v2_config.yaml",
                        help="ckpt 학습 시 사용한 train config. 반드시 ckpt와 짝이 맞아야 함.")
    parser.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--swap-cameras", action="store_true",
                        help="Swap camera1↔camera2 mapping")
    parser.add_argument("--skip-home", action="store_true",
                        help="Don't MoveJoints to HOME — use whatever pose the robot is in. "
                             "Recommended for align: manually pre-position needle near trocar first.")
    parser.add_argument("--start-joints", type=str, default=None,
                        help="6 joint angles (deg) 'j1,j2,j3,j4,j5,j6' to MoveJoints to before "
                             "inference — the trocar-adjacent align start pose from data collection "
                             "(HOME_JOINTS is only a far transit pose). Overrides HOME and forces "
                             "homing on (ignores --skip-home).")
    parser.add_argument("--joint-vel-limit", type=float, default=None,
                        help="Mecademic SetJointVelLimit (deg/s). Lower = slower + smoother. "
                             "Combine with larger --max-steps for slower trajectory. e.g. 5")
    parser.add_argument("--num-steps-execute", type=int, default=None,
                        help="How many actions from each predicted chunk to execute before "
                             "re-inferring (overrides eval.num_steps_execute in config). "
                             "Higher = more open-loop/smoother/fewer model calls, but less "
                             "closed-loop correction. SoTA measured at 2.")
    parser.add_argument("--use-sensor", action="store_true",
                        help="Append 2-channel sensor proprio [dist=5.0mm, valid=0.0] (saturated/no-info). "
                             "Required if checkpoint was trained with use_sensor=True (proprio_dim=9).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip MovePose; just print targets (safe smoke test)")
    parser.add_argument("--converge-thresh-mm", type=float, default=0.0,
                        help="Early-stop when mean(|dpos|) over last K steps < this (mm). "
                             "0 = disabled. Recommended: 0.3 (action range ±0.6mm/step).")
    parser.add_argument("--converge-window", type=int, default=5,
                        help="K: window of recent steps for convergence mean.")
    parser.add_argument("--converge-min-step", type=int, default=30,
                        help="Don't check convergence before this step (avoid premature stop at init).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint
    if args.num_steps_execute is not None:
        cfg_dict["eval"]["num_steps_execute"] = args.num_steps_execute

    # Parse fixed start joint config (deg). If given, it overrides --skip-home:
    # we always want to MoveJoints to the specified pose before inference.
    start_joints = None
    if args.start_joints:
        parts = [float(x) for x in args.start_joints.replace(" ", "").split(",")]
        assert len(parts) == 6, f"--start-joints needs 6 comma-separated values, got {len(parts)}"
        start_joints = tuple(parts)
        if args.skip_home:
            logging.getLogger(__name__).warning(
                "--start-joints given → overriding --skip-home (will MoveJoints to start pose)")
        args.skip_home = False

    cfg = DictConfig(cfg_dict)
    cfg.train_config_path = args.train_config
    cfg.robot_address = args.robot_address
    cfg.max_steps = args.max_steps
    cfg.num_episodes = args.num_episodes
    cfg.swap_cameras = args.swap_cameras
    cfg.skip_home = args.skip_home
    cfg.start_joints = start_joints
    cfg.use_sensor = args.use_sensor
    cfg.joint_vel_limit = args.joint_vel_limit
    cfg.dry_run = args.dry_run
    cfg.converge_thresh_mm = args.converge_thresh_mm
    cfg.converge_window = args.converge_window
    cfg.converge_min_step = args.converge_min_step

    run_real_eval(cfg)
