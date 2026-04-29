"""
move_to_final_pose.py — Sim-derived final-pose verification utility.

Computes the sim's *align-final* (needle retreated above trocar entry, axis
aligned with trocar) and *insertion-final* (needle tip at trocar depth) qpos
for a given phantom_pos, then drives the real Meca500 sequentially to those
poses with a pause at each so you can visually verify that the real trocar /
surrounding objects are positioned consistently with the sim.

NO data recording. NO cameras. Robot motion only.

Usage
-----
    # Both align + insertion final poses (default), 5s hold each
    python -m digital_twin.move_to_final_pose --phantom-pos 0.0 0.0

    # Align only, 10s hold for careful inspection
    python -m digital_twin.move_to_final_pose --phantom-pos 0.0 -0.4 \
        --phase align --pause-sec 10

    # Sim only — print solved qpos, no robot motion
    python -m digital_twin.move_to_final_pose --phantom-pos 0.0 0.0 --dry-run

Workflow
--------
  1) Load sim, place phantom at requested (x, y) in robot base.
  2) `_pre_align_sim` (reused) → aligned_qpos.
  3) `_solve_insertion_qpos` (here) → insertion_qpos (smooth IK from aligned).
  4) Connect Meca500, ActivateAndHome, MoveJoints HOME.
  5) MoveJoints aligned_qpos → WaitIdle → sleep(pause_sec).
  6) MoveJoints insertion_qpos → WaitIdle → sleep(pause_sec).
  7) (optional) Return to HOME, deactivate, disconnect.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import argparse
import logging
import pathlib

import numpy as np
import cv2
import mujoco
import mecademicpy.robot as mdr

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Sim"))

from Save_dataset_align_only import (  # noqa: E402
    smooth_step,
    ALIGN_THRESHOLD_M,
    ALIGN_HOLD_STEPS,
    RETREAT_MM,
)
from digital_twin.real_collect_approach import (  # noqa: E402
    SimHandles,
    _solve_ik_step,
    _set_phantom,
)
from digital_twin.real_collect_align import _pre_align_sim  # noqa: E402
from digital_twin.real_eval_approach import (  # noqa: E402
    ROBOT_ADDRESS_DEFAULT,
    HOME_JOINTS,
    OAKCameraManager,
)


logger = logging.getLogger(__name__)

INSERTION_SPEED_MPS = 0.005  # slow, mirrors FINE_ALIGN_SPEED for stable IK


def _solve_insertion_qpos(model, data, h: SimHandles, p_entry, p_depth, needle_len,
                          insertion_depth_mm=None, ik_speed: float = 0.5):
    """From current sim state (assumed at aligned pose), drive needle tip along
    the trocar axis to a target depth, returning the converged qpos.

    insertion_depth_mm: distance from p_entry along the trocar axis (toward
    p_depth) to drive the tip. None → use full sim depth (p_depth).
    """
    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    start_tip = data.site_xpos[h.tip_id].copy()
    start_back = data.site_xpos[h.back_id].copy()
    if insertion_depth_mm is None:
        goal_tip = p_depth.copy()
    else:
        goal_tip = p_entry + axis_dir * (float(insertion_depth_mm) / 1000.0)
    goal_back = goal_tip - axis_dir * needle_len

    insert_dist = float(np.linalg.norm(goal_tip - start_tip))
    duration = max(insert_dist / INSERTION_SPEED_MPS, 1e-3)
    t0 = data.time
    align_timer = 0

    while True:
        progress = smooth_step((data.time - t0) / duration)
        target_tip = (1 - progress) * start_tip + progress * goal_tip
        target_back = (1 - progress) * start_back + progress * goal_back
        _solve_ik_step(model, data, h, target_tip, target_back, ik_speed)
        mujoco.mj_step(model, data)

        if progress >= 1.0:
            curr_tip = data.site_xpos[h.tip_id].copy()
            if np.linalg.norm(curr_tip - goal_tip) < ALIGN_THRESHOLD_M:
                align_timer += 1
            else:
                align_timer = 0
            if align_timer > ALIGN_HOLD_STEPS:
                break

        if data.time - t0 > 60.0:
            raise RuntimeError("insertion IK timeout (>60s sim wall-time)")

    return data.qpos[: h.n_motors].copy()


def _display_frames(cam_mgr, status: str = ""):
    """Render available OAK frames side-by-side into a single window."""
    if cam_mgr is None:
        return
    try:
        frames = cam_mgr.get_frames()
    except Exception:
        return
    if not frames:
        return
    keys = sorted(frames.keys())
    imgs = []
    for k in keys:
        img = frames[k]
        if img is None:
            continue
        img = img.copy()
        cv2.putText(img, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if status:
            cv2.putText(img, status, (10, img.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        imgs.append(img)
    if not imgs:
        return
    disp = np.hstack(imgs) if len(imgs) > 1 else imgs[0]
    cv2.imshow("OAK Live Preview ('q' to skip waits)", disp)
    cv2.waitKey(1)


def _wait_idle_with_display(robot, cam_mgr, status: str, poll_sec: float = 0.05):
    """WaitIdle while pumping camera frames. Mecademic's WaitIdle(timeout)
    raises on timeout — we use that as a polling beat."""
    while True:
        try:
            robot.WaitIdle(timeout=poll_sec)
            return
        except Exception:
            pass
        _display_frames(cam_mgr, status)


def _hold_with_display(cam_mgr, status: str, hold_sec: float, poll_sec: float = 0.03):
    """Sleep for `hold_sec` while continuously displaying camera frames."""
    t_end = time.time() + hold_sec
    while time.time() < t_end:
        _display_frames(cam_mgr, status)
        time.sleep(poll_sec)


def _move_robot_to(robot, cam_mgr, joints_deg, label: str, pause_sec: float):
    rounded = [round(float(j), 2) for j in joints_deg]
    logger.info(f"🤖 → {label}: MoveJoints {rounded}")
    robot.MoveJoints(*[float(j) for j in joints_deg])
    _wait_idle_with_display(robot, cam_mgr, status=f"Moving → {label}")
    logger.info(f"   ✅ {label} reached. Holding {pause_sec:.1f}s for visual check…")
    _hold_with_display(cam_mgr, status=f"HOLD: {label}", hold_sec=pause_sec)


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Move real Meca500 to sim-derived align/insertion final poses (verification only)"
    )
    ap.add_argument("--phantom-pos", type=float, nargs=2, required=True,
                    metavar=("X", "Y"), help="Phantom XY in robot base (m)")
    ap.add_argument("--phantom-rot", type=float, default=None,
                    help="Phantom yaw (deg). Default: auto by Y (>=-0.25 → 0, else -90)")
    ap.add_argument("--phase", choices=["align", "insertion", "both"], default="both",
                    help="Which final pose(s) to send to the real robot")
    ap.add_argument("--pause-sec", type=float, default=5.0,
                    help="Seconds to hold at each pose for visual verification")
    ap.add_argument("--insertion-depth-mm", type=float, default=None,
                    help="Insertion depth from trocar entry along axis (mm). "
                         "Default: None → full sim depth (p_depth site).")
    ap.add_argument("--mujoco-xml", type=str,
                    default=str(_PROJECT_ROOT / "Sim" / "meca_add.xml"))
    ap.add_argument("--robot-address", type=str, default=ROBOT_ADDRESS_DEFAULT)
    ap.add_argument("--joint-vel-limit", type=float, default=30.0,
                    help="Mecademic SetJointVelLimit (deg/s). Conservative for verification.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sim only — print solved qpos, do not move robot")
    ap.add_argument("--return-home", action="store_true",
                    help="MoveJoints HOME at end before disconnect")
    ap.add_argument("--no-camera", action="store_true",
                    help="Skip OAK camera init / live preview window")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    # ── Sim solve ───────────────────────────────────────────────────────────
    logger.info(f"📦 Loading MuJoCo: {args.mujoco_xml}")
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    data = mujoco.MjData(model)
    h = SimHandles(model)
    if h.phantom_body_id < 0 or h.tip_id < 0 or h.target_entry_id < 0:
        raise RuntimeError("Sim model missing required phantom/tip/trocar handles")

    phantom_pos = tuple(args.phantom_pos)
    logger.info(f"🎯 phantom_pos={phantom_pos}, phantom_rot={args.phantom_rot}")
    mujoco.mj_resetData(model, data)
    _set_phantom(model, data, h, phantom_pos, args.phantom_rot)

    logger.info("🧮 Solving sim ALIGN-final qpos (sim_home → trocar entry retreated)…")
    aligned_qpos, _qvel, p_entry, p_depth, needle_len = _pre_align_sim(model, data, h)
    align_deg = np.rad2deg(aligned_qpos)
    logger.info(f"   trocar entry (mm)  = {(p_entry * 1000).round(1).tolist()}")
    logger.info(f"   trocar depth (mm)  = {(p_depth * 1000).round(1).tolist()}")
    logger.info(f"   needle length (mm) = {needle_len * 1000:.1f}")
    logger.info(f"   ALIGN qpos (deg)   = {align_deg.round(2).tolist()}")

    insertion_deg = None
    if args.phase in ("insertion", "both"):
        depth_label = (f"{args.insertion_depth_mm:.1f}mm from entry"
                       if args.insertion_depth_mm is not None else "full sim depth")
        logger.info(f"🧮 Solving sim INSERTION-final qpos (tip → {depth_label})…")
        insertion_qpos = _solve_insertion_qpos(
            model, data, h, p_entry, p_depth, needle_len,
            insertion_depth_mm=args.insertion_depth_mm,
        )
        insertion_deg = np.rad2deg(insertion_qpos)
        logger.info(f"   INSERTION qpos (deg) = {insertion_deg.round(2).tolist()}")

    if args.dry_run:
        logger.info("🟡 DRY-RUN done — no robot motion.")
        return

    # ── Camera init (live preview during motion + holds) ────────────────────
    cam_mgr = None
    if not args.no_camera:
        try:
            cam_mgr = OAKCameraManager(width=640, height=480)
            n_cams = cam_mgr.initialize_cameras()
            logger.info(f"📷 OAK cameras initialized ({n_cams})")
            for _ in range(10):
                cam_mgr.get_frames()
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Camera init failed: {e}; continuing without preview")
            cam_mgr = None

    # ── Real robot motion ───────────────────────────────────────────────────
    robot = mdr.Robot()
    logger.info(f"🔌 Connecting Meca500 @ {args.robot_address}…")
    robot.Connect(address=args.robot_address)
    if not robot.IsConnected():
        raise RuntimeError(f"Failed to connect to robot at {args.robot_address}")

    try:
        logger.info("🔋 ActivateAndHome…")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)
        try:
            robot.SetJointVelLimit(float(args.joint_vel_limit))
        except Exception as e:
            logger.warning(f"SetJointVelLimit failed (non-fatal): {e}")

        logger.info(f"🏠 MoveJoints HOME = {HOME_JOINTS}")
        robot.MoveJoints(*HOME_JOINTS)
        _wait_idle_with_display(robot, cam_mgr, status="Moving → HOME")
        time.sleep(0.5)

        # ALIGN is always traversed — it's the safe gateway between HOME and the
        # phantom (going HOME → INSERTION directly would drive the needle into
        # the phantom from an unsafe angle). Pause only when user requested it.
        align_pause = args.pause_sec if args.phase in ("align", "both") else 0.0
        _move_robot_to(robot, cam_mgr, align_deg, "ALIGN final", align_pause)

        if args.phase in ("insertion", "both") and insertion_deg is not None:
            _move_robot_to(robot, cam_mgr, insertion_deg, "INSERTION final", args.pause_sec)
            # Retract along trocar axis: insertion → align (reverse the path the
            # needle just took). HOME from inside the phantom would yank the
            # needle out at an angle — must un-insert first.
            logger.info("↩️  Retracting needle along trocar axis (insertion → align)…")
            robot.MoveJoints(*[float(j) for j in align_deg])
            _wait_idle_with_display(robot, cam_mgr, status="Retracting → ALIGN")

        if args.return_home:
            logger.info("🏠 Returning to HOME…")
            robot.MoveJoints(*HOME_JOINTS)
            _wait_idle_with_display(robot, cam_mgr, status="Moving → HOME")

    except KeyboardInterrupt:
        logger.warning("\n🛑 KeyboardInterrupt — stopping")
    finally:
        try:
            robot.DeactivateRobot()
        except Exception:
            pass
        try:
            robot.Disconnect()
        except Exception:
            pass
        if cam_mgr is not None:
            try:
                cam_mgr.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("✅ Done.")


if __name__ == "__main__":
    main()
