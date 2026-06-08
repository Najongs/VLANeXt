"""
real_eval_approach.py

Real-world deployment of an approach-phase VLANeXt policy.
Mirrors scripts/sim_eval_approach_only.py but applies the same predicted actions
to a real Mecademic Meca500 robot with OAK cameras instead of MuJoCo simulation.

Reuses load_model / load_processor / predict_action / preprocess_image / save_rollout_video
from scripts.sim_eval, and action_min/max from src.datasets.sim_act_approach.

Usage:
    python -m digital_twin.real_eval_approach \
        --config config/sim_eval_approach_config.yaml \
        --checkpoint /path/to/checkpoint \
        --train-config config/sim_train_approach_config.yaml \
        --max-steps 100 [--dry-run] [--swap-cameras]
"""

import os
import sys
import time
import argparse
import logging
import contextlib
import pathlib

import yaml
import numpy as np
import cv2

import depthai as dai
import mecademicpy.robot as mdr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, set_seed,
)
from src.datasets.sim_act_approach import (
    action_min_sim_approach as action_min_sim,
    action_max_sim_approach as action_max_sim,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
ROBOT_ADDRESS_DEFAULT = "192.168.0.100"
HOME_JOINTS = (30, -20, 20, 0, 30, 60)  # matches Robot_action.py — within sim training home distribution

TASK_INSTRUCTION = (
    "Approach the needle tip to the small grey circular trocar port on the eye model, "
    "next to the larger lens opening"
)

IMG_WIDTH, IMG_HEIGHT = 640, 480

# Safety clamps: model normalization is [-1, +1]mm / [-0.01, +0.01]rad.
# Clip 5x the trained range to catch wild outputs without strangling normal motion.
SAFETY_CLAMP_POS_MM = 5.0
SAFETY_CLAMP_ROT_RAD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# OAK Camera Manager (mirrors digital_twin/Robot_action.py:586-622)
# ─────────────────────────────────────────────────────────────────────────────
class OAKCameraManager:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.stack = contextlib.ExitStack()
        self.queues = []
        self.camera_ids = []

    def initialize_cameras(self):
        infos = dai.Device.getAllAvailableDevices()
        if not infos:
            raise RuntimeError("No OAK devices found")
        for info in infos:
            p = dai.Pipeline()
            c = p.create(dai.node.ColorCamera)
            c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            c.setFps(30)
            c.setPreviewSize(self.width, self.height)
            c.setInterleaved(False)

            cid = info.getMxId()
            self.camera_ids.append(cid)
            if cid.startswith("19"):
                c.initialControl.setManualFocus(101)
                logger.info(f"📷 Camera {cid}: manual focus 101")
            else:
                logger.info(f"📷 Camera {cid}: auto focus")

            out = p.create(dai.node.XLinkOut)
            out.setStreamName("rgb")
            c.preview.link(out.input)
            d = self.stack.enter_context(dai.Device(p, info, dai.UsbSpeed.SUPER))
            self.queues.append(d.getOutputQueue("rgb", 4, False))
        return len(self.queues)

    def get_frames(self, blocking_timeout=0.5):
        """Return dict {camera1, camera2, ...} with BGR uint8 frames (whatever's freshest)."""
        frames = {}
        deadline = time.time() + blocking_timeout
        for i, q in enumerate(self.queues):
            f = None
            while time.time() < deadline:
                f = q.tryGet()
                if f is not None:
                    break
                time.sleep(0.005)
            if f is not None:
                frames[f"camera{i+1}"] = f.getCvFrame()
        return frames

    def close(self):
        self.stack.close()


# ─────────────────────────────────────────────────────────────────────────────
# Real Robot Env (mirror of ApproachSimEnv)
# ─────────────────────────────────────────────────────────────────────────────
class ApproachRealEnv:
    """Real Meca500 + OAK cameras. Drop-in replacement for ApproachSimEnv during eval."""

    def __init__(self, robot_address=ROBOT_ADDRESS_DEFAULT, swap_cameras=False, skip_home=False,
                 joint_vel_limit=None, start_joints=None):
        self.swap_cameras = swap_cameras
        self.skip_home = skip_home
        # Joint config (deg) to MoveJoints to before inference. Defaults to
        # HOME_JOINTS (an inter-episode *transit* pose, far from the trocar).
        # For align, pass the trocar-adjacent start pose used during data
        # collection so eval begins in-distribution.
        self.start_joints = tuple(start_joints) if start_joints is not None else HOME_JOINTS

        # Robot
        logger.info(f"🔌 Connecting to robot at {robot_address}...")
        self.robot = mdr.Robot()
        self.robot.Connect(address=robot_address)
        if not self.robot.IsConnected():
            raise RuntimeError(f"Failed to connect to robot at {robot_address}")
        logger.info("✅ Robot connected. Activating + homing...")
        # A prior uncleanly-disconnected/crashed run can leave a latched error that
        # makes ActivateAndHome silently fail → later commands hit "1005 NOT_ACTIVATED".
        # Clear it first (no-op if no error), then activate and BLOCK until homed
        # (ActivateAndHome is async — sending MoveJoints before it completes fails).
        try:
            self.robot.ResetError()
            self.robot.ResumeMotion()
        except Exception as e:
            logger.warning(f"ResetError/ResumeMotion (non-fatal): {e}")
        self.robot.ActivateAndHome()
        try:
            self.robot.WaitHomed(timeout=60)
            logger.info("✅ Robot activated + homed")
        except Exception as e:
            logger.warning(f"WaitHomed failed/timeout (continuing): {e}")
        self.robot.SetRealTimeMonitoring(1)
        if joint_vel_limit is not None and joint_vel_limit > 0:
            try:
                self.robot.SetJointVelLimit(float(joint_vel_limit))
                logger.info(f"⚙️  SetJointVelLimit = {joint_vel_limit} deg/s (slower = smoother)")
            except Exception as e:
                logger.warning(f"SetJointVelLimit failed (non-fatal): {e}")
        if not skip_home:
            self.robot.MoveJoints(*self.start_joints)
            self.robot.WaitIdle()
            logger.info(f"🏠 Start joints {self.start_joints} reached")
        else:
            logger.info("⏭️ skip_home: starting from current robot pose (manual pre-positioning)")

        # Cameras
        self.cam_mgr = OAKCameraManager(width=IMG_WIDTH, height=IMG_HEIGHT)
        n_cams = self.cam_mgr.initialize_cameras()
        logger.info(f"📷 {n_cams} OAK camera(s) initialized: {self.cam_mgr.camera_ids}")
        if n_cams < 1:
            raise RuntimeError("Need at least 1 OAK camera (top_camera role)")
        if n_cams < 2:
            logger.warning("⚠️ Only 1 camera — tool_camera will be blank (model uses single view anyway)")

        # Warm cameras up
        for _ in range(15):
            self.cam_mgr.get_frames(blocking_timeout=0.1)
            time.sleep(0.05)

    # ── matches ApproachSimEnv.reset ──────────────────────────────────────────
    def reset(self):
        if not self.skip_home:
            self.robot.MoveJoints(*self.start_joints)
            self.robot.WaitIdle()
            time.sleep(0.5)
        try:
            initial_pose = self.robot.GetPose()
            logger.info(f"  Start pose (mm/deg): "
                        f"[{', '.join(f'{v:.1f}' for v in initial_pose[:6])}]")
        except Exception as e:
            logger.warning(f"GetPose failed at reset: {e}")

    # ── matches ApproachSimEnv.get_ee_pose ────────────────────────────────────
    def get_ee_pose(self) -> np.ndarray:
        """Return [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad] in NEEDLE-TIP frame.

        Mecademic GetPose() returns flange pose; we shift it by R_flange @ TIP_OFFSET
        so the model sees the same TCP as the sim/training data.
        """
        from src.utils.tip_frame import flange_to_tip_euler6
        for _ in range(5):
            try:
                pose = list(self.robot.GetPose())
                if pose and len(pose) >= 6:
                    arr = np.array(pose[:6], dtype=np.float64)
                    flange_pose = np.concatenate([arr[:3], np.deg2rad(arr[3:])])
                    tip_pose = flange_to_tip_euler6(flange_pose, mm=True)
                    return tip_pose.astype(np.float32)
            except Exception:
                time.sleep(0.01)
        logger.warning("GetPose failed; returning zeros")
        return np.zeros(6, dtype=np.float32)

    # ── matches ApproachSimEnv.render_cameras ─────────────────────────────────
    def render_cameras(self) -> dict:
        """Return RGB uint8 frames keyed by sim role names."""
        raw = self.cam_mgr.get_frames()
        cam1 = raw.get("camera1")
        cam2 = raw.get("camera2")
        if self.swap_cameras:
            cam1, cam2 = cam2, cam1

        blank = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        top = cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB) if cam1 is not None else blank
        tool = cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB) if cam2 is not None else blank
        return {
            "top_camera": top,
            "tool_camera": tool,
            "side_camera": blank,  # not consumed by approach model (view_mode=single)
        }

    # ── matches ApproachSimEnv.apply_delta_ee — sim units → robot command ──────
    def apply_delta_ee(self, delta_ee_6d, control_dt=0.067, dry_run=False):
        """
        Apply a single-step delta-EE command on the real robot.

        delta_ee_6d : (6,) — [dx_mm, dy_mm, dz_mm, drx_rad, dry_rad, drz_rad]
                              — same convention as sim's get_ee_pose.

        Strategy: read GetPose (mm + deg), add delta (mm + deg), MovePose absolute.
        This mirrors sim's `target = current + delta` in base frame, since training
        data was collected with delta = next_ee - current_ee in base/world frame.
        """
        # delta_ee_6d is in TIP frame (matches training data). Steps:
        #   1) read flange GetPose (mm + deg) → convert to tip (mm + rad)
        #   2) tip_target = tip + delta
        #   3) invert tip_target → flange_target (mm + rad)
        #   4) MovePose(flange_target with rot in deg)
        from src.utils.tip_frame import flange_to_tip_euler6, tip_to_flange_euler6

        delta_clamped = delta_ee_6d.copy().astype(np.float32)
        delta_clamped[:3] = np.clip(delta_clamped[:3], -SAFETY_CLAMP_POS_MM, SAFETY_CLAMP_POS_MM)
        delta_clamped[3:6] = np.clip(delta_clamped[3:6], -SAFETY_CLAMP_ROT_RAD, SAFETY_CLAMP_ROT_RAD)

        try:
            current = list(self.robot.GetPose())[:6]
        except Exception as e:
            logger.warning(f"GetPose failed in apply_delta_ee: {e}")
            time.sleep(control_dt)
            return None

        # current is flange (mm + deg). Convert to tip (mm + rad), apply delta, invert.
        flange_pose = np.array(
            [current[0], current[1], current[2],
             np.deg2rad(current[3]), np.deg2rad(current[4]), np.deg2rad(current[5])],
            dtype=np.float64,
        )
        tip_pose = flange_to_tip_euler6(flange_pose, mm=True)
        tip_target = tip_pose.copy()
        tip_target[:3] += delta_clamped[:3]
        tip_target[3:6] += delta_clamped[3:6].astype(np.float64)
        flange_target = tip_to_flange_euler6(tip_target, mm=True)
        target = [
            float(flange_target[0]),
            float(flange_target[1]),
            float(flange_target[2]),
            float(np.rad2deg(flange_target[3])),
            float(np.rad2deg(flange_target[4])),
            float(np.rad2deg(flange_target[5])),
        ]

        if dry_run:
            logger.info(f"[DRY] target_mm_deg={[f'{v:.2f}' for v in target]} "
                        f"delta={[f'{v:.3f}' for v in delta_clamped]}")
        else:
            try:
                self.robot.MovePose(*target)
            except Exception as e:
                logger.warning(f"MovePose failed: {e}; ResetError + ResumeMotion")
                try:
                    self.robot.ResetError()
                    self.robot.ResumeMotion()
                except Exception as e2:
                    logger.error(f"Recovery failed: {e2}")

        time.sleep(control_dt)
        return target

    def close(self):
        try:
            self.cam_mgr.close()
        except Exception as e:
            logger.error(f"Camera close failed: {e}")
        try:
            if self.robot.IsConnected():
                self.robot.DeactivateRobot()
                self.robot.Disconnect()
                logger.info("🔌 Robot disconnected")
        except Exception as e:
            logger.error(f"Robot close failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop (mirrors run_eval in sim_eval_approach_only.py)
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

    num_episodes = int(getattr(cfg, "num_episodes", 1))
    max_steps = int(getattr(cfg, "max_steps", 100))
    control_dt = 1.0 / float(video_fps)  # ~67ms at 15 Hz

    # Output dir alongside checkpoint
    ckpt_path = pathlib.Path(checkpoint_path)
    try:
        step_str = ckpt_path.stem.split("_")[-1] if ckpt_path.stem else "unknown"
    except Exception:
        step_str = "unknown"
    eval_dir = ckpt_path.parent / f"real_eval_step{step_str}_max{max_steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    logger.info(f"📁 Output dir: {eval_dir}")

    env = ApproachRealEnv(
        robot_address=getattr(cfg, "robot_address", ROBOT_ADDRESS_DEFAULT),
        swap_cameras=getattr(cfg, "swap_cameras", False),
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
            logger.info(f"▶ Episode {ep}/{num_episodes}")
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
                img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))

                # 2. Map cameras: cam_exterior=top_camera → primary; tool→wrist (held but not used in single-view)
                img_primary = img_top
                img_secondary = img_wrist
                image_history.append(img_primary)
                image_history_wrist.append(img_secondary)

                # 3. Proprio: 6-DoF EE pose only (gripper dropped, sensor detection-only)
                ee_pose = env.get_ee_pose()
                ee_traj.append(ee_pose[:3].copy())  # log mm-pos for trajectory plot
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

                # 4. Inference (replan when buffer empty)
                if len(action_buffer) == 0:
                    raw_chunk, _ = predict_action(model, processor, observation, TASK_INSTRUCTION)
                    if raw_chunk.ndim == 1:
                        raw_chunk = raw_chunk[None, :]
                    steps_exec = min(num_steps_execute, len(raw_chunk))
                    action_buffer = list(raw_chunk[:steps_exec])

                raw_action = action_buffer.pop(0)
                action_history.append(raw_action)

                # 5. Denormalize → delta_ee (mm + rad)
                denorm = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
                delta_ee = denorm[:6]

                # 6. Apply on real robot
                target = env.apply_delta_ee(delta_ee, control_dt=control_dt, dry_run=dry_run)

                # 7. Replay overlay + display
                replay_frame = np.concatenate([img_top, img_wrist], axis=1)
                cv2.putText(replay_frame, f"step {ctrl_step+1}/{max_steps}", (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(replay_frame,
                            f"dpos_mm={delta_ee[:3].round(2).tolist()}",
                            (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(replay_frame,
                            f"drot_deg={np.rad2deg(delta_ee[3:]).round(2).tolist()}",
                            (5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                if dry_run:
                    cv2.putText(replay_frame, "DRY-RUN", (5, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
                replay_images.append(replay_frame.copy())

                try:
                    cv2.imshow("Real Eval (top | tool) — press 'q' to abort",
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
                           f"drot_deg={np.rad2deg(delta_ee[3:]).round(2).tolist()} "
                           f"target={[f'{v:.1f}' for v in (target or [])]}")
                    logger.info(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

            # End of episode
            done_msg = f"  Episode {ep} ended after {ctrl_step + 1} steps (user_quit={user_quit})"
            logger.info(done_msg)
            log_file.write(done_msg + "\n")
            log_file.flush()

            # Trajectory NPZ
            ee_arr = np.stack(ee_traj) if ee_traj else np.zeros((0, 3), dtype=np.float32)
            np.savez_compressed(eval_dir / f"traj_ep{ep:03d}.npz", ee_pose=ee_arr)

            # Video
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
    parser = argparse.ArgumentParser(description="Real-robot eval for VLANeXt approach policy")
    parser.add_argument("--config", type=str, default="config/sim_eval_approach_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-config", type=str, default="config/sim_train_approach_config.yaml")
    parser.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--swap-cameras", action="store_true",
                        help="Swap camera1↔camera2 mapping")
    parser.add_argument("--joint-vel-limit", type=float, default=None,
                        help="Mecademic SetJointVelLimit (deg/s). Lower = slower + smoother. "
                             "Combine with larger --max-steps for slower trajectory. e.g. 5")
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
    cfg.joint_vel_limit = args.joint_vel_limit
    cfg.dry_run = args.dry_run

    run_real_eval(cfg)
