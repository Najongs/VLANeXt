"""Generate (image, tip_uv, trocar_uv) supervised dataset for the keypoint head.

Diversity recipe per sample:
  1. random phantom cell (x_mm in [-25,25], y_mm in [-25,75], z_mm in [0,25], angle in [-25,25])
  2. random tip target around goal_tip:
       lateral: 0-30mm in random direction perpendicular to trocar axis
       depth:   -10mm to +20mm along axis (so we get both far-approach and near-trocar views)
  3. random needle tilt: 0-15 degrees (axis-angle perturbation)
  4. run IK iterations to reach the target

Outputs a single HDF5 file with arrays:
  image      (N, H, W, 3) uint8  — H=W=image_size (default 512)
  tip_uv     (N, 2) float32, normalized [0,1]
  trocar_uv  (N, 2) float32, normalized [0,1]
  lateral_mm (N,) float32   — actual lateral after IK
  angle_deg  (N,) float32
  dist_mm    (N,) float32   — tip to entry

Usage:
    MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \\
    python -u -m scripts.keypoint_data_gen --n-samples 10000 \\
        --out dataset/keypoint/run01.h5
"""
import argparse
import os
import time

import h5py
import mujoco
import numpy as np
from PIL import Image

from scripts.sim_eval_align_only import AlignSimEnv, SIM_MODEL_PATH
from scripts.sim_eval import smooth_step


def sample_target(env, rng, lateral_max_mm=30.0, depth_range_mm=(-10.0, 20.0), tilt_max_deg=15.0):
    """Pick a target tip / back position around the trocar with controlled noise.
    Returns (target_tip, target_back) in world meters.
    """
    p_entry = env.data.site_xpos[env.target_entry_id].copy()
    p_depth = env.data.site_xpos[env.target_depth_id].copy()
    axis = p_depth - p_entry
    axis = axis / (np.linalg.norm(axis) + 1e-10)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    r = rng.uniform(0.0, lateral_max_mm) / 1000.0
    theta = rng.uniform(0.0, 2.0 * np.pi)
    lateral = e1 * (r * np.cos(theta)) + e2 * (r * np.sin(theta))

    depth_mm = rng.uniform(*depth_range_mm)
    goal_tip = p_entry - axis * (env.retreat_mm / 1000.0)
    target_tip = goal_tip + axis * (depth_mm / 1000.0) + lateral

    tilt_deg = rng.uniform(0.0, tilt_max_deg)
    tilt_axis = e1 * np.cos(rng.uniform(0.0, 2.0 * np.pi)) + e2 * np.sin(rng.uniform(0.0, 2.0 * np.pi))
    tilt_axis = tilt_axis / (np.linalg.norm(tilt_axis) + 1e-10)
    cos_t = np.cos(np.radians(tilt_deg)); sin_t = np.sin(np.radians(tilt_deg))
    # Rodrigues to rotate -axis vector (needle direction) by tilt around tilt_axis.
    needle_dir = -axis  # tip-to-back direction is opposite of axis (since tip toward hole)
    needle_dir_tilted = (needle_dir * cos_t
                         + np.cross(tilt_axis, needle_dir) * sin_t
                         + tilt_axis * np.dot(tilt_axis, needle_dir) * (1 - cos_t))

    # Determine needle length from current frame.
    curr_tip = env.data.site_xpos[env.tip_id].copy()
    curr_back = env.data.site_xpos[env.back_id].copy()
    needle_len = np.linalg.norm(curr_tip - curr_back)
    target_back = target_tip + needle_dir_tilted * needle_len
    return target_tip, target_back


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ik-iters", type=int, default=120)
    p.add_argument("--lateral-max-mm", type=float, default=30.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--image-size", type=int, default=512,
                   help="Output image resolution (square). 2026-05-18 upgrade: 256→512 to match SigLIP2 native + VLA eval pipeline.")
    args = p.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # _randomize_phantom uses np.random
    env = AlignSimEnv(os.path.abspath(SIM_MODEL_PATH), randomize_phantom=True, retreat_mm=10.0)

    H = W = int(args.image_size)
    f = h5py.File(args.out, "w")
    chunk_n = min(64, args.n_samples)
    img_ds = f.create_dataset("image", (args.n_samples, H, W, 3), dtype="u1", chunks=(chunk_n, H, W, 3))
    tip_ds = f.create_dataset("tip_uv", (args.n_samples, 2), dtype="f4")
    troc_ds = f.create_dataset("trocar_uv", (args.n_samples, 2), dtype="f4")
    lat_ds = f.create_dataset("lateral_mm", (args.n_samples,), dtype="f4")
    ang_ds = f.create_dataset("angle_deg", (args.n_samples,), dtype="f4")
    dist_ds = f.create_dataset("dist_mm", (args.n_samples,), dtype="f4")

    t0 = time.time()
    saved = 0
    samples_per_phantom = 10  # re-align this many times per fresh phantom

    failed_phantoms = 0
    while saved < args.n_samples:
        # Fresh phantom + pre-align needle to goal_tip (lateral=0, angle=0).
        env._aligned_qpos = None  # force re-align
        try:
            env._ensure_aligned_state()
        except RuntimeError as e:
            failed_phantoms += 1
            print(f"  pre-align failed ({failed_phantoms}); resampling phantom")
            continue

        for _ in range(samples_per_phantom):
            if saved >= args.n_samples:
                break
            # Start from the aligned state each iteration.
            env.data.qpos[: env.n_motors] = env._aligned_qpos
            env.data.qvel[: env.n_motors] = 0.0
            env.data.ctrl[: env.n_motors] = env._aligned_qpos[: env.n_motors]
            mujoco.mj_forward(env.model, env.data)

            target_tip, target_back = sample_target(env, rng)
            # Smooth IK from current state to target.
            start_tip = env.data.site_xpos[env.tip_id].copy()
            start_back = env.data.site_xpos[env.back_id].copy()
            for i in range(args.ik_iters):
                progress = smooth_step((i + 1) / args.ik_iters)
                t_tip = (1 - progress) * start_tip + progress * target_tip
                t_back = (1 - progress) * start_back + progress * target_back
                env._run_ik_step(t_tip, t_back, speed=0.6)
                mujoco.mj_step(env.model, env.data)
            # Extra convergence steps at the final target.
            for _ in range(20):
                env._run_ik_step(target_tip, target_back, speed=0.6)
                mujoco.mj_step(env.model, env.data)
        # Render + metrics — renderer outputs 640x480; resize to HxW (default 512).
        frames = env.render_cameras()
        img_raw = frames["tool_camera"]
        img = np.array(Image.fromarray(img_raw).resize((W, H), Image.LANCZOS))
        m = env.get_spatial_metrics()
        tip_uv = m["tip_uv"]; troc_uv = m["trocar_uv"]
        if tip_uv is None or troc_uv is None:
            continue
        # Reject samples where tip or trocar projection went off frame (well outside [0,1]).
        if not (-0.1 <= tip_uv[0] <= 1.1 and -0.1 <= tip_uv[1] <= 1.1):
            continue
        if not (-0.5 <= troc_uv[0] <= 1.5 and -0.5 <= troc_uv[1] <= 1.5):
            # allow trocar slightly off-frame (label still useful as direction)
            continue

        img_ds[saved] = img
        tip_ds[saved] = tip_uv
        troc_ds[saved] = troc_uv
        lat_ds[saved] = m["lateral_mm"]
        ang_ds[saved] = m["angle_deg"]
        dist_ds[saved] = m["dist_mm"]
        saved += 1
        if saved % args.save_every == 0:
            elapsed = time.time() - t0
            rate = saved / elapsed
            eta = (args.n_samples - saved) / rate
            print(f"  saved {saved}/{args.n_samples} | {rate:.1f} samples/s | ETA {eta:.0f}s | "
                  f"lat={m['lateral_mm']:.1f}mm ang={m['angle_deg']:.1f}deg")

    f.close()
    print(f"\nWrote {args.n_samples} samples to {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
