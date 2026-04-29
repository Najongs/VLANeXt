"""Real-robot dataset collection (fine-alignment) — sim-plan + real-replay variant.

Core difference vs `real_collect_align.py`:
  - Sim does NOT drive the real robot tick-by-tick. Instead:
      1. Sim computes `aligned_qpos` once (pre-align) and `perturbed_qpos` once
         (perturbation sampling + safety dry-run).
      2. Real robot is sent ONE blocking MoveJoints to perturbed.
      3. Real robot is sent ONE non-blocking MoveJoints toward aligned. The
         Mecademic controller plans its own smooth trajectory (S-curve in joint
         space). No streaming loop, no EMA, no jitter from sim IK noise.
      4. Recording loop runs at wall-clock 7.46 Hz: read real GetJoints/GetPose,
         set sim qpos = real qpos, mj_forward → derive aux keys (needle_tip_pos,
         keypoints, sensor_dist) from sim FK at the real pose. This guarantees
         aux labels are perfectly synced with real images/state.
      5. Hold phase: identical structure (single MoveJoints already at aligned,
         just record at wall-clock for HOLD_RECORD_STEPS frames).

Tradeoff:
  - Trajectory shape on real is Mecademic's joint-linear / S-curve interpolation,
    not sim's smooth_step IK trajectory. End-pose identical, intermediate path
    differs slightly.
  - Same HDF5 schema as real_collect_align.py — drop-in replacement for training.

Usage: same flags as Run_Collect_Real_Align.sh
"""

import os
import sys
import time
import argparse
import logging
import pathlib

import numpy as np
import cv2
import mujoco

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Sim"))

from Save_dataset_align_only import (  # noqa: E402
    SimRecorder,
    project_to_2d,
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

from digital_twin.real_collect_approach import (  # noqa: E402
    RealCollectEnv,
    SimHandles,
    _set_phantom,
    _wrap_pi,
    _build_display_frame,
)

from digital_twin.real_collect_align import (  # noqa: E402
    _check_tip_occluded,
    _check_qpos_path_safe,
    _clone_state,
    _dryrun_validate_episode,
    _pre_align_sim,
    _solve_retreat_qpos,
    _sample_perturbation,
)

from digital_twin.real_eval_approach import ROBOT_ADDRESS_DEFAULT, HOME_JOINTS  # noqa: E402

logger = logging.getLogger(__name__)


def _sync_sim_to_real(model, data, h: SimHandles, real_qpos_deg):
    """Set sim qpos = real qpos (rad), zero qvel, then mj_forward.
    Sim becomes a pure FK calculator at the real robot's actual pose."""
    data.qpos[: h.n_motors] = np.deg2rad(np.asarray(real_qpos_deg, dtype=np.float64))
    data.qvel[: h.n_motors] = 0.0
    mujoco.mj_forward(model, data)


def run_collection_episode_replay(model, data, env: RealCollectEnv,
                                  recorder: SimRecorder, cfg, h: SimHandles,
                                  ep_idx: int, rng: np.random.Generator,
                                  aligned_qpos, aligned_qvel, p_entry, p_depth,
                                  needle_len, path_anchor_qpos=None) -> bool:
    """One sim-planned, real-replay fine-alignment episode."""

    # ── ① Sim setup at aligned (used for safety check + perturb solving) ──
    mujoco.mj_resetData(model, data)
    data.qpos[: h.n_motors] = aligned_qpos
    data.qvel[: h.n_motors] = aligned_qvel
    _set_phantom(model, data, h, cfg.phantom_pos, cfg.phantom_rot)
    mujoco.mj_forward(model, data)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    retreat_m = RETREAT_MM / 1000.0
    goal_tip = p_entry - axis_dir * retreat_m
    goal_back = p_entry - axis_dir * (retreat_m + needle_len)

    # ── ② Sample + validate perturbation (sim-only dry-run) ──
    perturbed_qpos = None
    if cfg.skip_safety_validation:
        # Need a perturbed_qpos still — quick sim solve via dryrun_validate
        # with a single attempt and zero tolerance (we'll trust it).
        xyz, ang, ax = _sample_perturbation(rng)
        ok, reason, perturbed_qpos = _dryrun_validate_episode(
            model, data, h, goal_tip, goal_back, xyz, ang, ax,
            max_ctrl_steps=cfg.max_steps,
            tolerance_m=1.0,  # large → always ok (we just want the perturbed_qpos)
            qpos_path_steps=cfg.qpos_path_steps,
            start_qpos=path_anchor_qpos if path_anchor_qpos is not None else aligned_qpos,
        )
        logger.info(f"  Episode {ep_idx}: perturb [SAFETY OFF] solved={ok}")
    else:
        n_attempts = max(1, int(cfg.max_perturb_retries))
        last_reason = "no_attempt"
        for attempt in range(1, n_attempts + 1):
            xyz, ang, ax = _sample_perturbation(rng)
            anchor = path_anchor_qpos if path_anchor_qpos is not None else aligned_qpos
            ok, reason, candidate_qpos = _dryrun_validate_episode(
                model, data, h, goal_tip, goal_back, xyz, ang, ax,
                max_ctrl_steps=cfg.max_steps,
                tolerance_m=cfg.occlusion_tol_m,
                qpos_path_steps=cfg.qpos_path_steps,
                start_qpos=anchor,
            )
            if ok:
                perturbed_qpos = candidate_qpos
                logger.info(
                    f"  Episode {ep_idx}: perturb attempt {attempt} ✓ "
                    f"xyz_mm={(xyz*1000).round(1).tolist()} ang_deg={np.rad2deg(ang):.1f}"
                )
                break
            last_reason = reason
            logger.debug(f"    perturb attempt {attempt} rejected: {reason}")
        if perturbed_qpos is None:
            logger.warning(f"  Episode {ep_idx}: skipped (all candidates failed: {last_reason})")
            return False

    perturbed_deg = np.rad2deg(perturbed_qpos)
    aligned_deg = np.rad2deg(aligned_qpos)

    # ── ③ Move real to perturbed (blocking) ──
    logger.info(f"  Episode {ep_idx}: → perturbed pose (blocking MoveJoints)")
    env.reset_to_joints(perturbed_deg)

    # ── ④ Send aligned target as ONE non-blocking MoveJoints ──
    logger.info(f"  Episode {ep_idx}: ▶ MoveJoints(aligned) — Mecademic plans trajectory")
    if env.robot is not None and not env.dry_run:
        try:
            env.robot.MoveJoints(*[float(j) for j in aligned_deg])
        except Exception as e:
            logger.warning(f"MoveJoints(aligned) failed: {e}; recover + retry")
            env._recover()
            try:
                env.robot.MoveJoints(*[float(j) for j in aligned_deg])
            except Exception as e2:
                logger.error(f"MoveJoints retry failed: {e2}")
                return False

    # ── ⑤ Recording loop (wall-clock paced) ──
    record_dt = float(cfg.record_dt)  # seconds, default 1/7.46 ≈ 0.134
    user_quit = False
    success = False
    align_timer = 0
    ctrl_step = 0

    # Initialize last_real_ee from current state
    real_state0 = env.read_state()
    last_real_ee = real_state0["ee_pose"].copy()

    record_start = time.time()
    next_t = record_start

    while True:
        # Wall-clock pacing
        now = time.time()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += record_dt

        # Read real → sync sim
        real_state = env.read_state()
        real_qpos_deg = real_state["qpos_deg"]
        real_ee = real_state["ee_pose"]
        _sync_sim_to_real(model, data, h, real_qpos_deg)

        # ee delta
        real_delta_ee = real_ee - last_real_ee
        real_delta_ee[3:6] = _wrap_pi(real_delta_ee[3:6])
        real_clip_mm = ACTION_CLIP_MM * 5.0
        mag = np.linalg.norm(real_delta_ee[:3])
        if mag > real_clip_mm:
            real_delta_ee[:3] *= real_clip_mm / mag

        # sim aux at real pose
        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()
        nlen = np.linalg.norm(curr_tip - curr_back) + 1e-10
        needle_dir = (curr_tip - curr_back) / nlen
        dist = mujoco.mj_ray(
            model, data, curr_tip, needle_dir, None, 1, h.link6_id,
            np.zeros(1, dtype=np.int32),
        )
        current_sensor_dist = dist * 1000.0 if dist >= 0 else -1.0

        needle_tip_mm = curr_tip * 1000.0
        trocar_entry_mm = data.site_xpos[h.target_entry_id].copy() * 1000.0
        tip_uv = project_to_2d(curr_tip, model, data, "tool_camera",
                               IMG_WIDTH, IMG_HEIGHT)
        trocar_uv = project_to_2d(data.site_xpos[h.target_entry_id], model, data,
                                  "tool_camera", IMG_WIDTH, IMG_HEIGHT)
        keypoints_wrist = np.concatenate([tip_uv, trocar_uv]).astype(np.float32)
        tip_visible = float(0.0 <= tip_uv[0] <= 1.0 and 0.0 <= tip_uv[1] <= 1.0)
        trocar_visible = float(0.0 <= trocar_uv[0] <= 1.0 and 0.0 <= trocar_uv[1] <= 1.0)
        keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

        # cameras
        real_frames = env.render_frames()
        frames_for_recorder = {
            cam: real_frames[cam] for cam in cfg.record_cameras if cam in real_frames
        }

        recorder.add(
            frames_for_recorder,
            real_qpos_deg,
            real_ee.astype(np.float32),
            real_delta_ee.astype(np.float32),
            float(time.time() - record_start),
            1,  # phase=1
            float(current_sensor_dist),
            needle_tip_mm=needle_tip_mm.astype(np.float32),
            trocar_entry_mm=trocar_entry_mm.astype(np.float32),
            keypoints_wrist=keypoints_wrist,
            keypoints_visibility=keypoints_visibility,
            instruction=TASK_INSTRUCTION,
        )
        last_real_ee = real_ee.copy()
        ctrl_step += 1

        # Display + 'q' abort
        if cfg.show_preview:
            try:
                disp = _build_display_frame(
                    real_frames, ctrl_step, cfg.max_steps, 1,
                    real_delta_ee[:3], np.rad2deg(real_delta_ee[3:6]),
                    env.dry_run,
                )
                cv2.imshow("Real Collect Align (replay) — 'q' abort", disp)
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
                f"real_dpos_mm={real_delta_ee[:3].round(2).tolist()}  "
                f"tip→goal_mm={np.linalg.norm(curr_tip - goal_tip)*1000:.2f}"
            )

        # Success check (real ee close to aligned target via sim mirror)
        if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
            align_timer += 1
        else:
            align_timer = 0
        if align_timer > ALIGN_HOLD_STEPS:
            success = True
            break

        if ctrl_step >= cfg.max_steps:
            logger.warning(f"  Episode {ep_idx}: hit max_steps={cfg.max_steps}")
            break
        if (time.time() - record_start) > TIMEOUT_SEC * 2.0:
            logger.warning(f"  Episode {ep_idx}: wall-clock timeout")
            break

    if user_quit or not success:
        recorder.discard()
        reason = "user_quit" if user_quit else "no_success"
        logger.warning(f"  ❌ Episode {ep_idx} discarded ({reason})")
        return False

    # ── ⑥ Hold phase (continue recording at aligned for HOLD_RECORD_STEPS) ──
    # Robot already at aligned (success condition met). No new MoveJoints needed.
    next_t = time.time()
    for hold_i in range(HOLD_RECORD_STEPS):
        now = time.time()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += record_dt

        real_state = env.read_state()
        real_qpos_deg = real_state["qpos_deg"]
        real_ee = real_state["ee_pose"]
        _sync_sim_to_real(model, data, h, real_qpos_deg)

        real_delta_ee = real_ee - last_real_ee
        real_delta_ee[3:6] = _wrap_pi(real_delta_ee[3:6])
        real_clip_mm = ACTION_CLIP_MM * 5.0
        mag = np.linalg.norm(real_delta_ee[:3])
        if mag > real_clip_mm:
            real_delta_ee[:3] *= real_clip_mm / mag

        curr_tip = data.site_xpos[h.tip_id].copy()
        curr_back = data.site_xpos[h.back_id].copy()
        nlen = np.linalg.norm(curr_tip - curr_back) + 1e-10
        needle_dir = (curr_tip - curr_back) / nlen
        dist = mujoco.mj_ray(
            model, data, curr_tip, needle_dir, None, 1, h.link6_id,
            np.zeros(1, dtype=np.int32),
        )
        current_sensor_dist = dist * 1000.0 if dist >= 0 else -1.0

        needle_tip_mm = curr_tip * 1000.0
        trocar_entry_mm = data.site_xpos[h.target_entry_id].copy() * 1000.0
        tip_uv = project_to_2d(curr_tip, model, data, "tool_camera",
                               IMG_WIDTH, IMG_HEIGHT)
        trocar_uv = project_to_2d(data.site_xpos[h.target_entry_id], model, data,
                                  "tool_camera", IMG_WIDTH, IMG_HEIGHT)
        keypoints_wrist = np.concatenate([tip_uv, trocar_uv]).astype(np.float32)
        tip_visible = float(0.0 <= tip_uv[0] <= 1.0 and 0.0 <= tip_uv[1] <= 1.0)
        trocar_visible = float(0.0 <= trocar_uv[0] <= 1.0 and 0.0 <= trocar_uv[1] <= 1.0)
        keypoints_visibility = np.array([tip_visible, trocar_visible], dtype=np.float32)

        real_frames = env.render_frames()
        frames_for_recorder = {
            cam: real_frames[cam] for cam in cfg.record_cameras if cam in real_frames
        }

        recorder.add(
            frames_for_recorder,
            real_qpos_deg,
            real_ee.astype(np.float32),
            real_delta_ee.astype(np.float32),
            float(time.time() - record_start),
            1,
            float(current_sensor_dist),
            needle_tip_mm=needle_tip_mm.astype(np.float32),
            trocar_entry_mm=trocar_entry_mm.astype(np.float32),
            keypoints_wrist=keypoints_wrist,
            keypoints_visibility=keypoints_visibility,
            instruction=TASK_INSTRUCTION,
        )
        last_real_ee = real_ee.copy()

    if len(recorder.buffer) > 0:
        recorder.save_async()
        logger.info(f"  ✅ Episode {ep_idx} saved ({ctrl_step + HOLD_RECORD_STEPS} frames)")
        return True

    recorder.discard()
    return False


def _parse_args():
    ap = argparse.ArgumentParser(description="Real-robot fine-alignment dataset collection — sim-plan + real-replay")
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--phantom-pos", type=float, nargs=2, required=True,
                    help="Real phantom XY in robot base frame (meters). CALIBRATE.")
    ap.add_argument("--phantom-rot", type=float, default=None,
                    help="Phantom rotation degrees (default: auto-pick from y-coord)")
    ap.add_argument("--mujoco-xml", type=str,
                    default=str(_PROJECT_ROOT / "Sim" / "meca_add.xml"))
    ap.add_argument("--save-dir", type=str,
                    default=str(_PROJECT_ROOT / "dataset" / "real_align" / "collected_data_real_replay"))
    ap.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    ap.add_argument("--swap-cameras", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_CTRL_STEPS,
                    help="Recording-frame upper bound per episode (wall-clock).")
    ap.add_argument("--record-rate-hz", type=float, default=7.46,
                    help="Wall-clock recording rate (Hz). Matches sim's 67-step rate.")
    ap.add_argument("--joint-vel-limit", type=float, default=25.0,
                    help="SetJointVelLimit (deg/s). Lower = slower + smoother motion.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cameras", nargs="+", default=["tool_camera"],
                    choices=["top_camera", "tool_camera", "side_camera"])
    ap.add_argument("--no-return-home", dest="return_home",
                    action="store_false", default=True,
                    help="Skip MoveJoints(HOME) at shutdown")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-perturb-retries", type=int, default=5)
    ap.add_argument("--no-safety-validation", action="store_true")
    ap.add_argument("--occlusion-tolerance-mm", type=float, default=1.0)
    ap.add_argument("--qpos-path-steps", type=int, default=20)
    ap.add_argument("--inter-episode-retreat-mm", type=float, default=50.0)
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Save dir: {save_dir}")

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

    logger.info(f"📦 Loading MuJoCo model: {args.mujoco_xml}")
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    data = mujoco.MjData(model)
    h = SimHandles(model)
    if h.phantom_body_id < 0 or h.tip_id < 0 or h.target_entry_id < 0:
        raise RuntimeError("Required sim sites/bodies missing. Check meca_add.xml")

    env = RealCollectEnv(
        robot_address=args.robot_address,
        swap_cameras=args.swap_cameras,
        dry_run=args.dry_run,
        joint_vel_limit_deg_s=float(args.joint_vel_limit),
        ema_alpha=0.0,  # replay mode does not stream — EMA unused
    )

    recorder = SimRecorder(str(save_dir))

    class _Cfg: pass
    cfg = _Cfg()
    cfg.phantom_pos = tuple(args.phantom_pos)
    cfg.phantom_rot = args.phantom_rot
    cfg.record_dt = 1.0 / max(1e-3, float(args.record_rate_hz))
    cfg.max_steps = int(args.max_steps)
    cfg.record_cameras = list(args.cameras)
    logger.info(f"📸 Recording cameras: {cfg.record_cameras}  rate={1.0/cfg.record_dt:.2f}Hz")
    cfg.show_preview = not (args.no_display or not os.environ.get("DISPLAY"))
    cfg.skip_safety_validation = bool(args.no_safety_validation)
    cfg.max_perturb_retries = int(args.max_perturb_retries)
    cfg.occlusion_tol_m = float(args.occlusion_tolerance_mm) / 1000.0
    cfg.qpos_path_steps = int(args.qpos_path_steps)
    cfg.inter_episode_retreat_mm = float(args.inter_episode_retreat_mm)
    if cfg.skip_safety_validation:
        logger.warning("⚠️  Safety validation DISABLED")
    else:
        logger.info(
            f"🛡  Safety: dry-run validation ON (retries≤{cfg.max_perturb_retries}, "
            f"occl_tol={cfg.occlusion_tol_m*1000:.1f}mm)"
        )

    n_ok = 0
    try:
        # Pre-alignment (one-time)
        logger.info("🎯 Pre-alignment (sim) — solving aligned pose…")
        mujoco.mj_resetData(model, data)
        _set_phantom(model, data, h, cfg.phantom_pos, cfg.phantom_rot)
        aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len = _pre_align_sim(model, data, h)
        logger.info(f"   sim aligned_qpos (deg) = {np.rad2deg(aligned_qpos).round(1).tolist()}")

        if not cfg.skip_safety_validation:
            if _check_tip_occluded(model, data, h, cfg.occlusion_tol_m):
                raise RuntimeError(
                    f"Pre-align tip is occluded — phantom_pos={cfg.phantom_pos} likely misplaced."
                )
            home_rad = np.deg2rad(np.asarray(HOME_JOINTS, dtype=np.float64))
            sim_check = _clone_state(model, data)
            ok, reason = _check_qpos_path_safe(
                model, sim_check, h, home_rad, aligned_qpos,
                n_steps=cfg.qpos_path_steps, tolerance_m=cfg.occlusion_tol_m,
            )
            if not ok:
                raise RuntimeError(
                    f"HOME→aligned path occluded ({reason}). Fix phantom placement."
                )
            logger.info("   ✓ Pre-align safety check passed")

        retreated_qpos = None
        retreated_deg = None
        if cfg.inter_episode_retreat_mm > 0:
            sim_clone = _clone_state(model, data)
            retreated_qpos = _solve_retreat_qpos(
                model, sim_clone, h, p_entry, p_depth, needle_len,
                extra_retreat_mm=cfg.inter_episode_retreat_mm,
            )
            retreated_deg = np.rad2deg(retreated_qpos)
            logger.info(
                f"   inter-episode retreat qpos (deg, +{cfg.inter_episode_retreat_mm:.0f}mm) = "
                f"{retreated_deg.round(2).tolist()}"
            )

        logger.info("🤖 Mirroring aligned_qpos to real robot (blocking)…")
        env.reset_to_joints(np.rad2deg(aligned_qpos))

        current_anchor_rad = aligned_qpos.copy()
        for ep in range(1, args.num_episodes + 1):
            logger.info("\n" + "=" * 60 + f"\n▶ Replay Episode {ep}/{args.num_episodes}\n" + "=" * 60)
            ok = run_collection_episode_replay(
                model, data, env, recorder, cfg, h, ep, rng,
                aligned_qpos, aligned_qvel, p_entry, p_depth, needle_len,
                path_anchor_qpos=current_anchor_rad,
            )
            if ok:
                n_ok += 1

            if retreated_deg is not None:
                if ep < args.num_episodes:
                    logger.info("↩️  Inter-episode retreat along axis")
                else:
                    logger.info("↩️  Final retreat along axis")
                env.reset_to_joints(retreated_deg)
                current_anchor_rad = retreated_qpos.copy()
            else:
                current_anchor_rad = aligned_qpos.copy()
    except KeyboardInterrupt:
        logger.warning("\n🛑 KeyboardInterrupt — flushing pending saves")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        recorder.wait_for_all()
        if args.return_home and env.robot is not None and not args.dry_run:
            try:
                logger.info(f"🏠 Returning to HOME {HOME_JOINTS}…")
                env.robot.MoveJoints(*[float(j) for j in HOME_JOINTS])
                env.robot.WaitIdle()
            except Exception as e:
                logger.warning(f"Return-home failed: {e}")
        env.close()
        logger.info(f"\n✅ Done. {n_ok}/{args.num_episodes} episodes saved to {save_dir}")


if __name__ == "__main__":
    main()
