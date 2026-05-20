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
import os as _os
# 2026-05-14: env var로 sweep 파라미터 튜닝 가능 (실행 시 export HANDOFF_COARSE_WINDOW=8 등으로 override).
# Default: window 6mm step 1mm + fine 0.9mm step 0.3mm (w6s1 setting).
COARSE_WINDOW_MM = float(_os.getenv("HANDOFF_COARSE_WINDOW", "6.0"))
COARSE_STEP_MM = float(_os.getenv("HANDOFF_COARSE_STEP", "1.0"))
FINE_WINDOW_MM = float(_os.getenv("HANDOFF_FINE_WINDOW", "0.9"))
FINE_STEP_MM = float(_os.getenv("HANDOFF_FINE_STEP", "0.3"))


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


def _try_delta(env, delta_6d, sim_steps=40, capture_fn=None):
    snap = snapshot(env)
    env.apply_delta_ee(np.array(delta_6d, dtype=np.float32), n_sim_steps=sim_steps)
    s = env.get_sensor_dist()
    m = env.get_spatial_metrics()
    if capture_fn is not None:
        try:
            capture_fn(s, m)
        except Exception as e:
            print(f"  [handoff] sweep capture failed: {e}")
    restore(env, snap)
    return s, m


def _score(sensor_mm, lateral_mm):
    s = 1000.0 if sensor_mm < 0 else min(sensor_mm, 100.0)
    if lateral_mm > LATERAL_OK_MM:
        s -= (lateral_mm - LATERAL_OK_MM) * 5.0
    return s


def angle_sweep_at_current(env, drx_range, dry_range, sim_steps=40, verbose=False, capture_fn=None):
    """At current position, try (drx, dry) corrections in world frame. Returns best delta_6d.
    Goal: find an angle that makes the sensor read through-hole (>=25mm).
    """
    best = {"score": -np.inf, "sensor": 0.0, "lateral": 99.0,
            "drx": 0.0, "dry": 0.0, "metrics": None}
    for drx in drx_range:
        for dry in dry_range:
            delta6 = np.array([0, 0, 0, drx, dry, 0], dtype=np.float32)
            s, m = _try_delta(env, delta6, sim_steps=sim_steps, capture_fn=capture_fn)
            score = _score(s, m["lateral_mm"])
            if verbose:
                print(f"    angle drx={drx:+.3f} dry={dry:+.3f} → s={s:6.2f} lat={m['lateral_mm']:.2f} ang={m['angle_deg']:.2f}")
            if score > best["score"]:
                best = {"score": score, "sensor": s, "lateral": m["lateral_mm"],
                        "drx": float(drx), "dry": float(dry), "metrics": m}
    return best


def grid_sweep_in_axis_plane(env, window_mm, step_mm, u, v, sim_steps=40, verbose=False, capture_fn=None):
    """Sweep in plane spanned by u, v (perpendicular to trocar axis). Returns best world-frame delta."""
    offsets = np.arange(-window_mm, window_mm + 1e-9, step_mm)
    best = {"score": -np.inf, "sensor": 0.0, "lateral": 99.0,
            "du": 0.0, "dv": 0.0, "world_delta": np.zeros(3), "metrics": None}
    for du in offsets:
        for dv in offsets:
            world_delta = du * u + dv * v  # 3D world translation
            delta6 = np.array([world_delta[0], world_delta[1], world_delta[2], 0, 0, 0], dtype=np.float32)
            s, m = _try_delta(env, delta6, sim_steps=sim_steps, capture_fn=capture_fn)
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


def run_sensor_handoff(env, verbose=True, frames_out=None, image_size=256, hold_frames=8,
                        keypoint_seed_world_mm=None, keypoint_track_fn=None,
                        kp_track_iters=1, kp_track_alpha=0.7, kp_track_stop_norm=0.015,
                        kp_query_fn=None):
    """Trocar-axis-aware sensor handoff. See module docstring.

    keypoint_seed_world_mm: legacy single-shot seed (3-vec mm).
    keypoint_track_fn: callable(env) -> (world_delta_mm 3-vec, lateral_norm). If provided
        and kp_track_iters > 1, runs iterative visual-servo BEFORE grid sweep:
        loop N times: predict delta, apply alpha*delta, stop if lateral_norm < stop_norm.
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

    # Optional per-trial sweep frame capture for video visualization.
    # HANDOFF_VIDEO_SWEEP=1 → record each candidate (49 coarse + 49 fine + angle grid).
    # HANDOFF_VIDEO_SWEEP_STRIDE=N → only every Nth trial (default 1).
    _sweep_video = _os.getenv("HANDOFF_VIDEO_SWEEP", "0") == "1"
    _sweep_stride = max(1, int(_os.getenv("HANDOFF_VIDEO_SWEEP_STRIDE", "1")))
    _sweep_hold = max(1, int(_os.getenv("HANDOFF_VIDEO_SWEEP_HOLD", "4")))
    _sweep_counter = [0]
    _best_sensor = [-1.0]  # track running best so banner shows search progress

    def _sweep_capture(s, m):
        if not _sweep_video or frames_out is None:
            return
        _sweep_counter[0] += 1
        if _sweep_counter[0] % _sweep_stride != 0:
            return
        if s > _best_sensor[0]:
            _best_sensor[0] = s
        # Bigger, multi-line label so the sweep progress reads clearly at 15fps
        # Optionally include KP-head prediction (extra inference cost ~40ms/trial).
        kp_str = ""
        if kp_query_fn is not None:
            try:
                kp_d, kp_latn = kp_query_fn(env)
                kp_str = f"  KP_dist={kp_d:5.2f}mm  KP_lat={kp_latn:5.3f}"
            except Exception:
                pass
        label = (f"SWEEP #{_sweep_counter[0]:03d}  sensor={s:6.2f}mm  lat={m['lateral_mm']:5.2f}mm  "
                 f"ang={m['angle_deg']:5.2f}deg  best_s={_best_sensor[0]:5.2f}mm{kp_str}")
        try:
            f = _render_replay_frame(env, label, image_size=image_size)
            for _ in range(_sweep_hold):
                frames_out.append(f)
        except Exception as e:
            print(f"  [handoff] sweep frame capture failed: {e}")

    sweep_cap = _sweep_capture if _sweep_video else None

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

    # === NEW: Iterative KP tracking (visual servo) ===
    kp_track_alpha = float(_os.getenv("KP_TRACK_ALPHA", str(kp_track_alpha)))
    if keypoint_track_fn is not None and kp_track_iters > 1:
        track_log = []
        for it in range(kp_track_iters):
            try:
                wdelta, lat_norm = keypoint_track_fn(env)
            except Exception as e:
                if verbose: print(f"  [kp_track] iter{it} failed: {e}; stop")
                break
            wdelta = np.asarray(wdelta, dtype=np.float32)
            # Project to (u,v) plane, scale by alpha, clip per-iter magnitude.
            du_p = float(np.dot(wdelta, u)) * kp_track_alpha
            dv_p = float(np.dot(wdelta, v)) * kp_track_alpha
            mag = float(np.hypot(du_p, dv_p))
            STEP_MAX_MM = 8.0
            if mag > STEP_MAX_MM:
                du_p *= STEP_MAX_MM / mag; dv_p *= STEP_MAX_MM / mag
            wstep = du_p * u + dv_p * v
            env.apply_delta_ee(np.array([wstep[0], wstep[1], wstep[2], 0, 0, 0], dtype=np.float32),
                               n_sim_steps=40)
            post_m = env.get_spatial_metrics()
            post_s = env.get_sensor_dist()
            if verbose:
                print(f"  [kp_track] it{it} lat_norm={lat_norm:.4f} step=({du_p:+.2f},{dv_p:+.2f})mm "
                      f"→ sensor={post_s:.2f}mm lat={post_m['lateral_mm']:.2f}mm")
            track_log.append({"it": it, "lat_norm": lat_norm, "du": du_p, "dv": dv_p,
                              "sensor": post_s, "lateral": post_m["lateral_mm"]})
            if lat_norm < kp_track_stop_norm:
                break
        log["kp_track"] = track_log
        _capture("handoff_kp_track")

    # === NEW: Keypoint-seeded lateral pre-correction ===
    # Project the requested world-frame delta onto the (u, v) plane perpendicular to trocar axis,
    # then apply. Removes most of lateral error before sensor grid search starts.
    skip_legacy_seed = (keypoint_track_fn is not None and kp_track_iters > 1)
    if keypoint_seed_world_mm is not None and not skip_legacy_seed:
        seed_vec = np.asarray(keypoint_seed_world_mm, dtype=np.float32)
        # Decompose onto trocar (u, v) basis to keep motion in handoff plane
        du_proj = float(np.dot(seed_vec, u))
        dv_proj = float(np.dot(seed_vec, v))
        seed_lateral_mm = float(np.hypot(du_proj, dv_proj))
        # Clip large seeds to safe magnitude (avoid huge jumps if keypoint badly off)
        SEED_MAX_MM = 20.0
        scale = min(1.0, SEED_MAX_MM / max(seed_lateral_mm, 1e-6))
        du_proj *= scale; dv_proj *= scale
        seed_world = du_proj * u + dv_proj * v
        # apply_delta_ee takes delta in MM (target = current + delta, both in mm units)
        delta6 = np.array([seed_world[0], seed_world[1], seed_world[2],
                            0.0, 0.0, 0.0], dtype=np.float32)
        env.apply_delta_ee(delta6, n_sim_steps=40)  # persistent apply
        seed_post_m = env.get_spatial_metrics()
        seed_post_s = env.get_sensor_dist()
        if verbose:
            print(f"  [kp_seed] applied (du,dv)=({du_proj:+.2f},{dv_proj:+.2f})mm → "
                  f"sensor={seed_post_s:.2f}mm lat={seed_post_m['lateral_mm']:.2f}mm")
        log["kp_seed"] = {"du": du_proj, "dv": dv_proj, "sensor_post": seed_post_s,
                          "lateral_post": seed_post_m["lateral_mm"]}
        _capture("handoff_kp_seed")

    already_through = (env.get_sensor_dist() < 0 or env.get_sensor_dist() >= THROUGH_HOLE_THRESHOLD_MM) and env.get_spatial_metrics()["lateral_mm"] < LATERAL_OK_MM

    if not already_through:
        # NEW: vision-aware pre-stage. Use oracle keypoints (in real this would be
        # replaced by a vision tracker). Compute world tip→entry vector, project
        # onto trocar-axis-perp plane, apply that delta directly (no sweep).
        # Enabled via HANDOFF_VISION_ORACLE=1. Cheap (1 trial) — sensor sweep follows.
        if _os.getenv("HANDOFF_VISION_ORACLE", "0") == "1":
            tip_pos = env.data.site_xpos[env.tip_id].copy()
            entry_pos = env.data.site_xpos[env.target_entry_id].copy()
            world_delta_3d = (entry_pos - tip_pos) * 1000.0  # m → mm
            # Project onto plane perpendicular to trocar axis (u, v) — don't move along axis here.
            du_v = float(np.dot(world_delta_3d, u))
            dv_v = float(np.dot(world_delta_3d, v))
            proj_delta = du_v * u + dv_v * v
            # Optional noise injection to simulate vision tracker error
            noise_mm = float(_os.getenv("HANDOFF_VISION_NOISE_MM", "0.0"))
            if noise_mm > 0:
                proj_delta = proj_delta + np.random.uniform(-noise_mm, noise_mm, 3)
            env.apply_delta_ee(np.array([proj_delta[0], proj_delta[1], proj_delta[2], 0, 0, 0],
                                        dtype=np.float32), n_sim_steps=40)
            vis_m = env.get_spatial_metrics()
            vis_s = env.get_sensor_dist()
            if verbose:
                print(f"  [handoff] vision: du={du_v:+.2f} dv={dv_v:+.2f} "
                      f"sensor={vis_s:.2f}mm lat={vis_m['lateral_mm']:.2f}mm")
            log["phases"].append({"name": "vision", "du": du_v, "dv": dv_v,
                                  "sensor": vis_s, "lateral": vis_m["lateral_mm"]})
            _capture("after_vision")

        # NEW: angle nudge pre-stage. If sensor doesn't currently see through,
        # try small drx/dry corrections so sensor ray can pierce the hole.
        # Enabled via HANDOFF_ANGLE_SWEEP=1 (default off).
        if _os.getenv("HANDOFF_ANGLE_SWEEP", "1") == "1":
            arange = float(_os.getenv("HANDOFF_ANGLE_RANGE", "0.26"))  # rad (~15°)
            asteps = int(_os.getenv("HANDOFF_ANGLE_STEPS", "7"))
            drx_arr = np.linspace(-arange, arange, asteps)
            dry_arr = np.linspace(-arange, arange, asteps)
            ang_best = angle_sweep_at_current(env, drx_arr, dry_arr, verbose=verbose, capture_fn=sweep_cap)
            if verbose:
                print(f"  [handoff] angle: drx={ang_best['drx']:+.3f} dry={ang_best['dry']:+.3f} "
                      f"sensor={ang_best['sensor']:.2f}mm lat={ang_best['lateral']:.2f}mm")
            if ang_best["sensor"] >= 5.0:  # apply only if improvement (sensor wasn't very small)
                env.apply_delta_ee(np.array([0, 0, 0, ang_best["drx"], ang_best["dry"], 0], dtype=np.float32),
                                   n_sim_steps=40)
                log["phases"].append({"name": "angle", **{k: v for k, v in ang_best.items() if k != "metrics"}})
                _capture("after_angle")

        coarse = grid_sweep_in_axis_plane(env, COARSE_WINDOW_MM, COARSE_STEP_MM, u, v, capture_fn=sweep_cap)
        if verbose:
            print(f"  [handoff] coarse: du={coarse['du']:+.2f} dv={coarse['dv']:+.2f} "
                  f"sensor={coarse['sensor']:.2f}mm lat={coarse['lateral']:.2f}mm")
        wd = coarse["world_delta"]
        env.apply_delta_ee(np.array([wd[0], wd[1], wd[2], 0, 0, 0]), n_sim_steps=40)
        log["phases"].append({"name": "coarse", **{k: v for k, v in coarse.items() if k != "world_delta"}})
        _capture("after_coarse")

        fine = grid_sweep_in_axis_plane(env, FINE_WINDOW_MM, FINE_STEP_MM, u, v, capture_fn=sweep_cap)
        if verbose:
            print(f"  [handoff] fine: du={fine['du']:+.2f} dv={fine['dv']:+.2f} "
                  f"sensor={fine['sensor']:.2f}mm lat={fine['lateral']:.2f}mm")
        wd = fine["world_delta"]
        env.apply_delta_ee(np.array([wd[0], wd[1], wd[2], 0, 0, 0]), n_sim_steps=40)
        log["phases"].append({"name": "fine", **{k: v for k, v in fine.items() if k != "world_delta"}})
        _capture("after_fine")

    # Polish pass: if not aligned after first pass, retry angle + fine XY once more.
    aligned_sensor = env.get_sensor_dist()
    aligned_m = env.get_spatial_metrics()
    not_aligned = not ((aligned_sensor < 0 or aligned_sensor >= THROUGH_HOLE_THRESHOLD_MM)
                       and aligned_m["lateral_mm"] < LATERAL_OK_MM)
    if (not_aligned and not already_through
            and _os.getenv("HANDOFF_POLISH", "1") == "1"
            and _os.getenv("HANDOFF_ANGLE_SWEEP", "1") == "1"):
        arange = float(_os.getenv("HANDOFF_ANGLE_RANGE", "0.26"))
        asteps = int(_os.getenv("HANDOFF_ANGLE_STEPS", "7"))
        drx_arr = np.linspace(-arange, arange, asteps)
        dry_arr = np.linspace(-arange, arange, asteps)
        ang2 = angle_sweep_at_current(env, drx_arr, dry_arr, verbose=verbose, capture_fn=sweep_cap)
        if verbose:
            print(f"  [handoff] polish-angle: drx={ang2['drx']:+.3f} dry={ang2['dry']:+.3f} "
                  f"sensor={ang2['sensor']:.2f}mm lat={ang2['lateral']:.2f}mm")
        if ang2["sensor"] >= 5.0:
            env.apply_delta_ee(np.array([0, 0, 0, ang2["drx"], ang2["dry"], 0], dtype=np.float32),
                               n_sim_steps=40)
            log["phases"].append({"name": "polish-angle", **{k: v for k, v in ang2.items() if k != "metrics"}})
            _capture("after_polish_angle")
        fine2 = grid_sweep_in_axis_plane(env, FINE_WINDOW_MM, FINE_STEP_MM, u, v, capture_fn=sweep_cap)
        if verbose:
            print(f"  [handoff] polish-fine: du={fine2['du']:+.2f} dv={fine2['dv']:+.2f} "
                  f"sensor={fine2['sensor']:.2f}mm lat={fine2['lateral']:.2f}mm")
        wd = fine2["world_delta"]
        env.apply_delta_ee(np.array([wd[0], wd[1], wd[2], 0, 0, 0]), n_sim_steps=40)
        log["phases"].append({"name": "polish-fine", **{k: v for k, v in fine2.items() if k != "world_delta"}})
        _capture("after_polish_fine")

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
