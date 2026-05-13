"""Sensor-based handoff controller (v3: trocar-axis aware).

After VLA brings needle close (~5-10mm) to trocar, fine-search via sensor
ray reading (through-hole = ~30mm; off-axis = ~6-12mm). Once aligned,
push along TROCAR AXIS for insertion.

v3 changes vs v2:
  - Lateral sweep is now in the plane PERPENDICULAR to trocar axis (not world XY).
  - Insertion pushes along trocar axis direction (toward entry → depth).
  - Angle sweep still disabled.
  - Lateral_mm validation kept (avoid through-hole false-positives where
    the ray geometry coincidentally aligns but needle tip is far from entry).
"""
import numpy as np
import mujoco

THROUGH_HOLE_THRESHOLD_MM = 25.0
LATERAL_OK_MM = 1.5
INSERT_PUSH_MM = 8.0
INSERT_STEPS = 3
COARSE_WINDOW_MM = 3.0
COARSE_STEP_MM = 1.0
FINE_WINDOW_MM = 0.9
FINE_STEP_MM = 0.3


def snapshot(env):
    return {"qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
            "ctrl": env.data.ctrl.copy(), "time": env.data.time}


def restore(env, snap):
    env.data.qpos[:] = snap["qpos"]
    env.data.qvel[:] = snap["qvel"]
    env.data.ctrl[:] = snap["ctrl"]
    env.data.time = snap["time"]
    mujoco.mj_forward(env.model, env.data)


def get_trocar_axes(env):
    """Return (axis_dir, u, v) — unit vectors in world frame.
    axis_dir = entry → depth direction (insertion direction).
    u, v = orthonormal basis perpendicular to axis_dir.
    """
    p_entry = env.data.site_xpos[env.target_entry_id].copy()
    p_depth = env.data.site_xpos[env.target_depth_id].copy()
    axis = p_depth - p_entry
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    # Build u perpendicular to axis: pick world-X then orthogonalize. Fall back to Y if parallel.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, axis) * axis
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(axis, u)
    return axis, u, v


def _try_delta(env, delta_6d, sim_steps=40):
    snap = snapshot(env)
    env.apply_delta_ee(np.array(delta_6d, dtype=np.float32), n_sim_steps=sim_steps)
    s = env.get_sensor_dist()
    m = env.get_spatial_metrics()
    restore(env, snap)
    return s, m


def _score(sensor_mm, lateral_mm):
    s = 1000.0 if sensor_mm < 0 else min(sensor_mm, 100.0)
    if lateral_mm > LATERAL_OK_MM:
        s -= (lateral_mm - LATERAL_OK_MM) * 5.0
    return s


def grid_sweep_in_axis_plane(env, window_mm, step_mm, u, v, sim_steps=40, verbose=False):
    """Sweep in plane spanned by u, v (perpendicular to trocar axis). Returns best world-frame delta."""
    offsets = np.arange(-window_mm, window_mm + 1e-9, step_mm)
    best = {"score": -np.inf, "sensor": 0.0, "lateral": 99.0,
            "du": 0.0, "dv": 0.0, "world_delta": np.zeros(3), "metrics": None}
    for du in offsets:
        for dv in offsets:
            world_delta = du * u + dv * v  # 3D world translation
            delta6 = np.array([world_delta[0], world_delta[1], world_delta[2], 0, 0, 0], dtype=np.float32)
            s, m = _try_delta(env, delta6, sim_steps=sim_steps)
            score = _score(s, m["lateral_mm"])
            if verbose:
                print(f"    sweep du={du:+.2f} dv={dv:+.2f} → s={s:6.2f} lat={m['lateral_mm']:.2f}")
            if score > best["score"]:
                best = {"score": score, "sensor": s, "lateral": m["lateral_mm"],
                        "du": float(du), "dv": float(dv),
                        "world_delta": world_delta.copy(), "metrics": m}
    return best


def _render_replay_frame(env, label="", image_size=256):
    from scripts.sim_eval import preprocess_image, draw_overlay
    import cv2
    frames = env.render_cameras()
    img_ext = preprocess_image(frames["side_camera"], (image_size, image_size))
    img_wrist = preprocess_image(frames["tool_camera"], (image_size, image_size))
    img_top = preprocess_image(frames["top_camera"], (image_size, image_size))
    replay = np.concatenate([img_ext, img_wrist, img_top], axis=1)
    m = env.get_spatial_metrics()
    m["spatial_pred"] = None
    try:
        draw_overlay(replay, m, f"HANDOFF:{label}")
    except Exception as e:
        print(f"  [handoff] draw_overlay failed: {e}")
    # Banner so handoff frames are unmistakable in the saved mp4
    try:
        cv2.rectangle(replay, (0, 0), (replay.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(replay, f"HANDOFF: {label}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"  [handoff] banner draw failed: {e}")
    return replay


def run_sensor_handoff(env, verbose=True, frames_out=None, image_size=256, hold_frames=8):
    """Trocar-axis-aware sensor handoff. See module docstring.

    hold_frames: number of duplicate frames appended after each phase so the
    saved mp4 visibly lingers on each handoff stage instead of flashing past.
    """
    def _capture(label):
        if frames_out is None:
            return
        try:
            f = _render_replay_frame(env, label, image_size=image_size)
        except Exception as e:
            print(f"  [handoff] CAPTURE FAIL label={label}: {type(e).__name__}: {e}")
            return
        for _ in range(max(1, hold_frames)):
            frames_out.append(f)
        if verbose:
            print(f"  [handoff] captured frame: {label} (×{hold_frames}, total replay_frames now)")

    log = {"phases": []}
    _capture("handoff_pre")

    pre_sensor = env.get_sensor_dist()
    pre_m = env.get_spatial_metrics()
    log["pre"] = {"sensor": pre_sensor, "dist": pre_m["dist_mm"], "lateral": pre_m["lateral_mm"],
                  "angle": pre_m["angle_deg"]}
    if verbose:
        print(f"  [handoff] pre: sensor={pre_sensor:.2f}mm dist={pre_m['dist_mm']:.2f}mm "
              f"lateral={pre_m['lateral_mm']:.2f}mm angle={pre_m['angle_deg']:.2f}deg")

    axis_dir, u, v = get_trocar_axes(env)

    already_through = (pre_sensor < 0 or pre_sensor >= THROUGH_HOLE_THRESHOLD_MM) and pre_m["lateral_mm"] < LATERAL_OK_MM

    if not already_through:
        coarse = grid_sweep_in_axis_plane(env, COARSE_WINDOW_MM, COARSE_STEP_MM, u, v)
        if verbose:
            print(f"  [handoff] coarse: du={coarse['du']:+.2f} dv={coarse['dv']:+.2f} "
                  f"sensor={coarse['sensor']:.2f}mm lat={coarse['lateral']:.2f}mm")
        wd = coarse["world_delta"]
        env.apply_delta_ee(np.array([wd[0], wd[1], wd[2], 0, 0, 0]), n_sim_steps=40)
        log["phases"].append({"name": "coarse", **{k: v for k, v in coarse.items() if k != "world_delta"}})
        _capture("after_coarse")

        fine = grid_sweep_in_axis_plane(env, FINE_WINDOW_MM, FINE_STEP_MM, u, v)
        if verbose:
            print(f"  [handoff] fine: du={fine['du']:+.2f} dv={fine['dv']:+.2f} "
                  f"sensor={fine['sensor']:.2f}mm lat={fine['lateral']:.2f}mm")
        wd = fine["world_delta"]
        env.apply_delta_ee(np.array([wd[0], wd[1], wd[2], 0, 0, 0]), n_sim_steps=40)
        log["phases"].append({"name": "fine", **{k: v for k, v in fine.items() if k != "world_delta"}})
        _capture("after_fine")

    aligned_sensor = env.get_sensor_dist()
    aligned_m = env.get_spatial_metrics()
    did_align = ((aligned_sensor < 0 or aligned_sensor >= THROUGH_HOLE_THRESHOLD_MM)
                 and aligned_m["lateral_mm"] < LATERAL_OK_MM)
    log["aligned"] = {"sensor": aligned_sensor, "lateral": aligned_m["lateral_mm"],
                      "angle": aligned_m["angle_deg"], "achieved": did_align}
    if verbose:
        print(f"  [handoff] aligned: sensor={aligned_sensor:.2f}mm "
              f"lateral={aligned_m['lateral_mm']:.2f}mm "
              f"angle={aligned_m['angle_deg']:.2f}deg "
              f"→ {'ALIGNED ✓' if did_align else 'NOT ALIGNED ✗'}")

    if did_align:
        per_step = INSERT_PUSH_MM / INSERT_STEPS
        readings = []
        for i in range(INSERT_STEPS):
            push = per_step * axis_dir  # world frame push along trocar axis
            env.apply_delta_ee(np.array([push[0], push[1], push[2], 0, 0, 0], dtype=np.float32), n_sim_steps=40)
            readings.append({"sensor": env.get_sensor_dist(), "metrics": env.get_spatial_metrics()})
            _capture(f"insertion_{i+1}")
        log["insertion"] = readings
        final_m = readings[-1]["metrics"]
        final_sensor = readings[-1]["sensor"]
        if verbose:
            sensors_str = " → ".join(f"{r['sensor']:.1f}" for r in readings)
            depths_str = " → ".join(f"{r['metrics']['insertion_depth_mm']:.1f}" for r in readings)
            print(f"  [handoff] insertion sensors: {sensors_str}")
            print(f"  [handoff] insertion depth (mm): {depths_str}")
            print(f"  [handoff] final: dist={final_m['dist_mm']:.2f}mm "
                  f"lateral={final_m['lateral_mm']:.2f}mm "
                  f"depth={final_m['insertion_depth_mm']:.2f}mm")
        log["final"] = {**final_m, "sensor": final_sensor}
    else:
        log["insertion"] = None
        log["final"] = {**aligned_m, "sensor": aligned_sensor}

    return log
