"""
Sim_eval_withReal_action_align.py

Hybrid diagnostic: **sim images (model input) + real robot action (shadow)**.

Goal: decouple the *visual domain gap* (real camera) from the *action/control
pipeline gap* (frame conversion, Euler convention, scaling). The model only ever
sees clean sim renders/proprio; the real Meca500 mirrors every commanded delta as
a passive shadow. By comparing the sim tip trajectory against the real achieved
tip trajectory we can answer:

  - real reproduces sim motion  → action pipeline faithful; the only remaining
    real-world problem is the camera / visual domain gap.
  - real diverges from sim       → action pipeline broken (frame conv, Euler
    convention, axis/scale). Per-axis Δtip logging exposes which axis.

Design = "Sim master + real shadow" (AlignSimEnv drives state & renders; the same
predicted delta is applied to the real robot each step). Real never feeds back, so
sim is the source of truth for the visual observation.

⚠️ EULER CONVENTION (the historical sim↔real mismatch):
  MuJoCo proprio/action = extrinsic XYZ; Mecademic robot = intrinsic XYZ.
  Model output `denorm_action[3:6]` is in Mecademic intrinsic XYZ (training conv).
    - REAL  ← denorm_action[:6]                      (as-is; real_eval_align path)
    - SIM   ← rotation delta converted to MuJoCo      (sim_eval_align_only path)
  Position delta [:3] is applied identically to both (no conversion).

Usage:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m digital_twin.Sim_eval_withReal_action_align \
    --config config/sim_eval_align_config.yaml \
    --checkpoint checkpoints/checkpoint_1500.pt \
    --train-config config/sim_train_align_qwen_reach_recover_v2_aggressive_config.yaml \
    --start-joints "43,-28,28,0,34,57" \
    --max-steps 200 --joint-vel-limit 5 \
    --phantom-pos 0 0 0 \
    --phantom-angle 0 
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
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.sim_eval import (
    DictConfig, load_model, load_processor, predict_action,
    preprocess_image, save_rollout_video, set_seed, SIM_MODEL_PATH,
)
from scripts.sim_eval_align_only import AlignSimEnv
from src.datasets.sim_act_align import (
    action_min_sim_align as action_min_sim,
    action_max_sim_align as action_max_sim,
)
from src.datasets.euler_convention import (
    mujoco_to_mecademic_euler, mecademic_to_mujoco_euler,
)
from digital_twin.real_eval_approach import (
    ApproachRealEnv as RealRobotEnv, ROBOT_ADDRESS_DEFAULT,
)

logger = logging.getLogger(__name__)

TASK_INSTRUCTION = (
    "Align the needle tip to the small grey circular trocar port on the eye model, "
    "next to the larger lens opening"
)


def reset_sim_to_joints(env, joints_deg, phantom_pos, phantom_angle_deg, retreat_mm):
    """Put the sim robot at an explicit joint config (deg) with a fixed phantom,
    so sim starts in the SAME configuration as the real shadow (start_joints).
    Bypasses the pre-align+perturb reset. Returns initial needle→trocar dist (mm)."""
    mujoco.mj_resetData(env.model, env.data)
    if phantom_pos is not None:
        px, py = float(phantom_pos[0]), float(phantom_pos[1])
        pz = float(phantom_pos[2]) if len(phantom_pos) > 2 else 0.0
        ang = float(phantom_angle_deg) if phantom_angle_deg is not None else 0.0
        env._apply_phantom(px, py, pz, ang)
    q = np.deg2rad(np.asarray(joints_deg, dtype=np.float64))
    n = env.n_motors
    env.data.qpos[:6] = q
    env.data.qvel[:n] = 0.0
    env.data.ctrl[:n] = q[:n]
    mujoco.mj_forward(env.model, env.data)
    # Cache trocar sites + goal_tip (for metrics / distance readout)
    env._p_entry = env.data.site_xpos[env.target_entry_id].copy()
    env._p_depth = env.data.site_xpos[env.target_depth_id].copy()
    axis = env._p_depth - env._p_entry
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    env._goal_tip = env._p_entry - axis * (retreat_mm / 1000.0)
    env.align_hold_counter = 0
    tip = env.data.site_xpos[env.tip_id].copy()
    return float(np.linalg.norm(tip - env._p_entry) * 1000.0)


def run(cfg):
    checkpoint_path = cfg.eval.finetuned_checkpoint
    assert checkpoint_path, "eval.finetuned_checkpoint must be set!"
    set_seed(getattr(cfg.eval, "seed", 0))

    diff_steps = getattr(cfg.model, "diffusion_steps", 10)
    sched_type = getattr(cfg.model, "scheduler_type", "flow_match")
    train_config_path = getattr(cfg, "train_config_path", None)

    model = load_model(checkpoint_path, diffusion_steps=diff_steps,
                       scheduler_type=sched_type, train_config_path=train_config_path)
    processor = load_processor(checkpoint_path, train_config_path=train_config_path)

    pdim = getattr(model, "proprio_dim", 6)
    if pdim != 6:
        logger.warning(f"⚠️ model.proprio_dim={pdim} != 6. This diagnostic feeds 6-DoF EE "
                       f"proprio only (no sensor/KP). Use a proprio_dim=6 checkpoint.")

    image_size = getattr(cfg.eval, "image_size", 256)
    num_steps_execute = int(getattr(cfg.eval, "num_steps_execute", 1))
    sim_steps_per_ctrl = int(getattr(cfg.eval, "sim_steps_per_control", 67))
    video_fps = getattr(cfg.eval, "video_fps", 15)
    save_video = getattr(cfg.eval, "save_video", True)

    max_steps = int(getattr(cfg, "max_steps", 200))
    control_dt = 1.0 / float(video_fps)
    use_real = not getattr(cfg, "no_real", False)
    dry_run = getattr(cfg, "dry_run", False)
    real_wait_idle = getattr(cfg, "real_wait_idle", True)
    real_mode = getattr(cfg, "real_mode", "delta")  # "joints" (demo) | "delta" (diagnostic)
    if real_mode == "joints":
        logger.info("🎬 real_mode=joints: real robot mirrors sim joint trajectory (demo mode)")

    converge_thresh_mm = float(getattr(cfg, "converge_thresh_mm", 0.0))
    converge_window = int(getattr(cfg, "converge_window", 5))
    converge_min_step = int(getattr(cfg, "converge_min_step", 30))

    # Output dir
    ckpt_path = pathlib.Path(checkpoint_path)
    step_str = ckpt_path.stem.split("_")[-1] if ckpt_path.stem else "unknown"
    eval_dir = ckpt_path.parent / f"simimg_realact_step{step_str}_max{max_steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(eval_dir / "log.txt", "w")
    logger.info(f"📁 Output dir: {eval_dir}")

    # ── Sim env (master) ──────────────────────────────────────────────────────
    model_xml = os.path.abspath(SIM_MODEL_PATH)
    phantom_pos = getattr(cfg, "phantom_pos", None)
    retreat_mm = getattr(cfg, "retreat_mm", 10.0)
    sim_env = AlignSimEnv(model_xml, randomize_phantom=False,
                          phantom_pos=phantom_pos, retreat_mm=retreat_mm)
    # Force a fixed phantom rotation WITHOUT depending on the AlignSimEnv signature
    # (older copies of sim_eval_align_only.py lack the phantom_angle_deg kwarg).
    # Wrap _set_fixed_phantom so its angle_deg is pinned. Only effective with --phantom-pos.
    _angle = getattr(cfg, "phantom_angle_deg", None)
    if _angle is not None:
        if phantom_pos is None:
            logger.warning("⚠️ --phantom-angle has no effect without --phantom-pos (phantom not fixed)")
        else:
            import types
            _orig_set = sim_env._set_fixed_phantom
            def _set_fixed_angle(self, pos, pz=0.0, angle_deg=None, _a=_angle, _o=_orig_set):
                return _o(pos, pz=pz, angle_deg=_a)
            sim_env._set_fixed_phantom = types.MethodType(_set_fixed_angle, sim_env)
            logger.info(f"🔧 Phantom rotation pinned to {_angle}°")

    # ── Real env (shadow) ─────────────────────────────────────────────────────
    real_env = None
    if use_real:
        real_env = RealRobotEnv(
            robot_address=getattr(cfg, "robot_address", ROBOT_ADDRESS_DEFAULT),
            swap_cameras=getattr(cfg, "swap_cameras", False),
            skip_home=getattr(cfg, "skip_home", False),
            joint_vel_limit=getattr(cfg, "joint_vel_limit", None),
            start_joints=getattr(cfg, "start_joints", None),
        )
    else:
        logger.warning("⏭️ --no-real: pure sim sanity check (no robot)")

    a_min = np.array(action_min_sim, dtype=np.float32)
    a_max = np.array(action_max_sim, dtype=np.float32)

    try:
        # Sim start: match the real shadow's joint config by default (--sim-start-joints,
        # else --start-joints). Falls back to pre-align+perturb if neither is set.
        sim_start_joints = getattr(cfg, "sim_start_joints", None) or getattr(cfg, "start_joints", None)
        if sim_start_joints is not None:
            dist_mm = reset_sim_to_joints(sim_env, sim_start_joints, phantom_pos, _angle, retreat_mm)
            logger.info(f"🔧 Sim started at joints {tuple(round(float(j),1) for j in sim_start_joints)} "
                        f"(matches real); needle→trocar = {dist_mm:.1f}mm")
            if dist_mm > 30.0:
                logger.warning(f"⚠️ Sim needle is {dist_mm:.0f}mm from trocar — likely OUT of "
                               f"fine-align distribution. Move the sim phantom to match the real "
                               f"setup via --phantom-pos (x y z, meters) until this drops to a few mm.")
        else:
            sim_env.reset()
        if real_env is not None:
            real_env.reset()

        image_history, image_history_wrist = [], []
        state_history, action_history, action_buffer = [], [], []
        replay_images = []          # combined: sim tool | sim top | real tool (256px)
        real_replay_images = []     # real cameras only, native res (tool | top)
        log_rows = []            # per-step: sim_tip(mec6), real_tip(mec6), cmd_delta(mec6)
        dpos_norms_mm = []
        prev_sim_tip = None
        prev_real_tip = None

        for ctrl_step in range(max_steps):
            # 1. SIM render (model input)
            frames = sim_env.render_cameras()
            img_tool = preprocess_image(frames["tool_camera"], (image_size, image_size))
            img_top = preprocess_image(frames["top_camera"], (image_size, image_size))
            img_primary = img_tool      # train cfg cam_exterior = tool_camera
            image_history.append(img_primary)
            image_history_wrist.append(img_top)

            # 2. SIM proprio → Mecademic intrinsic XYZ (training convention)
            ee_pose = sim_env.get_ee_pose()                  # MuJoCo extrinsic XYZ
            ee_pose_mec = ee_pose.copy()
            ee_pose_mec[3:6] = mujoco_to_mecademic_euler(ee_pose[3:6])
            proprio = ee_pose_mec[:6].astype(np.float32)
            state_history.append(proprio)

            observation = {
                "full_image": img_primary,
                "full_image_wrist": img_top,
                "image_history": image_history,
                "image_history_wrist": image_history_wrist,
                "state_history": state_history,
                "action_history": action_history,
            }

            # 3. Inference
            if len(action_buffer) == 0:
                raw_chunk, _ = predict_action(model, processor, observation, TASK_INSTRUCTION)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk[None, :]
                steps_exec = min(num_steps_execute, len(raw_chunk))
                action_buffer = list(raw_chunk[:steps_exec])
            raw_action = action_buffer.pop(0)
            action_history.append(raw_action)

            # 4. Denormalize → Mecademic intrinsic XYZ tip-frame delta
            denorm = (raw_action + 1.0) / 2.0 * (a_max - a_min) + a_min
            delta_mec = denorm[:6].astype(np.float32)        # [dx,dy,dz mm, drx,dry,drz rad (mecademic)]

            # 5a. SIM apply — convert rotation delta to MuJoCo extrinsic XYZ
            target_mec_rpy = ee_pose_mec[3:6] + delta_mec[3:6]
            target_muj_rpy = mecademic_to_mujoco_euler(target_mec_rpy)
            d_rpy_muj = target_muj_rpy - ee_pose[3:6]
            d_rpy_muj = np.arctan2(np.sin(d_rpy_muj), np.cos(d_rpy_muj))
            delta_sim = np.concatenate([delta_mec[:3], d_rpy_muj]).astype(np.float32)
            sim_env.apply_delta_ee(delta_sim, n_sim_steps=sim_steps_per_ctrl)

            # 5b. REAL apply. Two modes:
            #   "joints" (DEMO): mirror sim's resulting joint config exactly →
            #       real reproduces sim's converging trajectory (looks aligned).
            #       Bypasses tip-frame/Euler/WRF entirely (joints map 1:1).
            #   "delta"  (DIAGNOSTIC): replay tip-frame delta open-loop (drifts).
            if real_env is not None:
                if real_mode == "joints":
                    sim_qpos_deg = np.rad2deg(np.asarray(sim_env.data.qpos[:6], dtype=np.float64))
                    if not dry_run:
                        try:
                            real_env.robot.MoveJoints(*[float(x) for x in sim_qpos_deg])
                            if real_wait_idle:
                                real_env.robot.WaitIdle()
                        except Exception as e:
                            logger.warning(f"MoveJoints(mirror) failed: {e}")
                    else:
                        logger.info(f"[DRY] mirror joints = {sim_qpos_deg.round(2).tolist()}")
                    time.sleep(control_dt)
                else:  # "delta"
                    real_env.apply_delta_ee(delta_mec, control_dt=control_dt, dry_run=dry_run)
                    if not dry_run and real_wait_idle:
                        try:
                            real_env.robot.WaitIdle()
                        except Exception as e:
                            logger.warning(f"WaitIdle failed: {e}")

            # 6. Read achieved tip poses (both in Mecademic mm+rad tip frame) → compare
            sim_tip = ee_pose_mec[:6].copy()
            real_tip = real_env.get_ee_pose() if real_env is not None else np.full(6, np.nan, np.float32)
            log_rows.append(np.concatenate([sim_tip, real_tip, delta_mec]).astype(np.float32))

            sim_dtip = (sim_tip[:3] - prev_sim_tip[:3]) if prev_sim_tip is not None else np.zeros(3)
            real_dtip = (real_tip[:3] - prev_real_tip[:3]) if (prev_real_tip is not None and real_env is not None) else np.full(3, np.nan)
            prev_sim_tip, prev_real_tip = sim_tip.copy(), real_tip.copy()

            # 7. Video. Combined panel (sim tool | sim top | real tool, 256px) +
            #    a separate native-res real-camera video (tool | top).
            row = [img_tool, img_top]
            if real_env is not None:
                real_frames = real_env.render_cameras()
                real_panel = np.concatenate([real_frames["tool_camera"],
                                             real_frames["top_camera"]], axis=1)
                cv2.putText(real_panel, f"step {ctrl_step+1}/{max_steps}  tool | top",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                real_replay_images.append(real_panel.copy())
                row.append(preprocess_image(real_frames["tool_camera"], (image_size, image_size)))
            replay_frame = np.concatenate(row, axis=1)
            cv2.putText(replay_frame, f"step {ctrl_step+1}/{max_steps} [SIMimg|REALact]",
                        (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(replay_frame, f"cmd dpos_mm={delta_mec[:3].round(2).tolist()}",
                        (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            replay_images.append(replay_frame.copy())

            # 8. Convergence (commanded action magnitude self-report)
            dpos_mm = float(np.linalg.norm(delta_mec[:3]))
            dpos_norms_mm.append(dpos_mm)
            if (converge_thresh_mm > 0 and ctrl_step + 1 >= converge_min_step
                    and len(dpos_norms_mm) >= converge_window
                    and float(np.mean(dpos_norms_mm[-converge_window:])) < converge_thresh_mm):
                logger.info(f"  ✓ converged at step {ctrl_step+1}")
                break

            if (ctrl_step + 1) % 5 == 0:
                msg = (f"  step {ctrl_step+1:3d}: cmd_dpos={delta_mec[:3].round(2).tolist()} "
                       f"sim_dtip={np.round(sim_dtip,2).tolist()} "
                       f"real_dtip={np.round(real_dtip,2).tolist()}")
                logger.info(msg)
                log_file.write(msg + "\n"); log_file.flush()

        # Save trajectory comparison: columns = sim_tip[6], real_tip[6], cmd_delta[6]
        arr = np.stack(log_rows) if log_rows else np.zeros((0, 18), np.float32)
        np.savez_compressed(eval_dir / "traj_compare.npz",
                            data=arr,
                            columns=np.array(
                                [f"sim_{a}" for a in "xyzABC"] +
                                [f"real_{a}" for a in "xyzABC"] +
                                [f"cmd_{a}" for a in "xyzABC"]))
        logger.info(f"💾 traj_compare.npz saved ({arr.shape[0]} steps). "
                    f"Compare sim_x/y/z vs real_x/y/z step-deltas to spot axis/scale/sign mismatch.")
        if save_video and replay_images:
            save_rollout_video(replay_images, 1, success=False,
                               save_dir=str(eval_dir), fps=video_fps)
        if save_video and real_replay_images:
            import imageio
            rpath = str(eval_dir / "real_camera.mp4")
            w = imageio.get_writer(rpath, fps=video_fps)
            for im in real_replay_images:
                w.append_data(im)
            w.close()
            logger.info(f"  Saved real-camera video: {rpath}")
    except KeyboardInterrupt:
        logger.warning("\n🛑 Interrupted (Ctrl-C)")
    finally:
        if real_env is not None:
            real_env.close()
        log_file.close()
        logger.info(f"📁 Results: {eval_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sim-image + real-action shadow diagnostic (align)")
    p.add_argument("--config", type=str, default="config/sim_eval_align_config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str,
                   default="config/sim_train_align_qwen_reach_recover_v2_aggressive_config.yaml")
    p.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--start-joints", type=str, default=None,
                   help="6 joint angles (deg) 'j1,..,j6' the real shadow homes to before start.")
    p.add_argument("--sim-start-joints", type=str, default=None,
                   help="6 joint angles (deg) for the SIM start. Default = same as --start-joints "
                        "(sim & real begin in identical config). Omit both to use pre-align+perturb.")
    p.add_argument("--joint-vel-limit", type=float, default=None)
    p.add_argument("--num-steps-execute", type=int, default=None,
                   help="Override eval.num_steps_execute (action chunk slice length).")
    p.add_argument("--swap-cameras", action="store_true")
    p.add_argument("--phantom-pos", type=float, nargs=3, default=None,
                   help="Fixed sim phantom (x y z) offset in meters from XML base. "
                        "'0 0 0' = default centered placement.")
    p.add_argument("--phantom-angle", type=float, default=None,
                   help="Fixed sim phantom rotation (deg) about Z. Default None = random ±25°. "
                        "Pass 0 for no rotation.")
    p.add_argument("--retreat-mm", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Disable real MovePose (robot only homes to start; sim still runs).")
    p.add_argument("--no-real", action="store_true",
                   help="Skip real robot entirely — pure sim sanity check.")
    p.add_argument("--real-mode", choices=["delta", "joints"], default="delta",
                   help="How the real robot follows sim. 'joints' (DEMO): mirror sim's joint "
                        "trajectory each step → real reproduces sim's converging motion (looks "
                        "aligned). 'delta' (DIAGNOSTIC): open-loop tip-delta replay (drifts).")
    p.add_argument("--no-real-wait-idle", dest="real_wait_idle", action="store_false",
                   help="Don't WaitIdle after each real move (faster but achieved pose lags).")
    p.add_argument("--converge-thresh-mm", type=float, default=0.0)
    p.add_argument("--converge-window", type=int, default=5)
    p.add_argument("--converge-min-step", type=int, default=30)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("eval", {})["finetuned_checkpoint"] = args.checkpoint
    if args.num_steps_execute is not None:
        cfg_dict["eval"]["num_steps_execute"] = args.num_steps_execute

    def _parse_joints(s, flag):
        if not s:
            return None
        parts = [float(x) for x in s.replace(" ", "").split(",")]
        assert len(parts) == 6, f"{flag} needs 6 comma-separated values, got {len(parts)}"
        return tuple(parts)

    start_joints = _parse_joints(args.start_joints, "--start-joints")
    sim_start_joints = _parse_joints(args.sim_start_joints, "--sim-start-joints")

    cfg = DictConfig(cfg_dict)
    cfg.train_config_path = args.train_config
    cfg.robot_address = args.robot_address
    cfg.max_steps = args.max_steps
    cfg.start_joints = start_joints
    cfg.sim_start_joints = sim_start_joints
    cfg.skip_home = False                       # always home the real shadow to start_joints
    cfg.joint_vel_limit = args.joint_vel_limit
    cfg.swap_cameras = args.swap_cameras
    cfg.phantom_pos = tuple(args.phantom_pos) if args.phantom_pos is not None else None
    cfg.phantom_angle_deg = args.phantom_angle
    cfg.retreat_mm = args.retreat_mm
    cfg.dry_run = args.dry_run
    cfg.no_real = args.no_real
    cfg.real_mode = args.real_mode
    cfg.real_wait_idle = args.real_wait_idle
    cfg.converge_thresh_mm = args.converge_thresh_mm
    cfg.converge_window = args.converge_window
    cfg.converge_min_step = args.converge_min_step

    run(cfg)
