"""Smoke test: vision_axis_estimator vs ground-truth trocar axis.

Spawns AlignSimEnv, samples N random phantom configs, renders tool_camera,
fits ellipse → axis_world, compares to GT axis (entry→depth site direction).

Reports angular error (deg) distribution + ellipse-fit failure rate.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Headless rendering env vars must be set BEFORE mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                       "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
from sim_eval_align_only import AlignSimEnv, project_to_2d, IMG_WIDTH, IMG_HEIGHT
from vision_axis_estimator import estimate_trocar_axis_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="Sim/meca_add.xml")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--roi", type=int, default=64)
    ap.add_argument("--use-gt-uv", action="store_true",
                    help="Use GT trocar UV (skip KP head). Best-case axis estimator test.")
    ap.add_argument("--save-debug", default="",
                    help="If set, dump first 4 ROI frames with overlay to this dir.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = AlignSimEnv(args.xml, randomize_phantom=True)

    errs = []
    failures = 0
    debug_saved = 0
    for i in range(args.n):
        try:
            env.reset()
        except Exception as e:
            print(f"[ep{i}] reset fail: {e}"); continue

        frames = env.render_cameras()
        img = frames["tool_camera"]

        # GT axis (entry -> depth) in world.
        p_entry = env.data.site_xpos[env.target_entry_id].copy()
        p_depth = env.data.site_xpos[env.target_depth_id].copy()
        gt_axis = p_depth - p_entry
        gt_axis = gt_axis / (np.linalg.norm(gt_axis) + 1e-10)

        # GT trocar UV (skip KP for axis-only diagnostic).
        u_norm, v_norm = project_to_2d(p_entry, env.model, env.data, "tool_camera",
                                         IMG_WIDTH, IMG_HEIGHT)
        # project_to_2d returns normalized [0,1] coords already.
        kp_uv = (float(u_norm), float(v_norm))

        cam_mat = env.data.cam_xmat[env._tool_cam_id].reshape(3, 3)
        cam_pos = env.data.cam_xpos[env._tool_cam_id].copy()
        fovy_deg = float(env.model.cam_fovy[env._tool_cam_id])

        # Expected trocar radius in pixels at predicted depth.
        dist_to_entry = float(np.linalg.norm(p_entry - cam_pos))
        TROCAR_R_MM = 1.6  # approximate
        mm_per_px = (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0) * dist_to_entry) / 256.0
        exp_r_px = TROCAR_R_MM / max(mm_per_px, 1e-6)

        result = estimate_trocar_axis_world(img, kp_uv, cam_mat,
                                              roi_px=args.roi,
                                              prior_axis_world=gt_axis,
                                              expected_radius_px=exp_r_px)
        if not result["ok"]:
            failures += 1
            print(f"[ep{i}] ellipse fit FAILED  GT_axis={gt_axis}")
            continue

        est_axis = result["axis_world"]
        cos = float(np.clip(np.dot(est_axis, gt_axis), -1.0, 1.0))
        err_deg = float(np.rad2deg(np.arccos(cos)))
        errs.append(err_deg)
        ell = result["ellipse"]
        print(f"[ep{i}] tilt_est={result['tilt_deg']:5.1f}deg "
              f"axes=({ell['axes_px'][0]:.1f},{ell['axes_px'][1]:.1f})px "
              f"ang={ell['major_angle_deg']:.1f}deg  "
              f"axis_err={err_deg:5.2f}deg  conf={ell['confidence']:.2f}")

        if args.save_debug and debug_saved < 4:
            import cv2
            os.makedirs(args.save_debug, exist_ok=True)
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            from PIL import Image
            vis_pil = Image.fromarray(img).convert("RGB").resize((256, 256), Image.LANCZOS)
            vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
            cx, cy = ell["center_px"]
            smin, smaj = ell["axes_px"]
            cv2.ellipse(vis, (int(cx), int(cy)), (int(smaj), int(smin)),
                        ell["major_angle_deg"], 0, 360, (0, 255, 0), 1)
            cv2.circle(vis, (int(kp_uv[0]*256), int(kp_uv[1]*256)), 3, (0, 0, 255), -1)
            cv2.imwrite(f"{args.save_debug}/ep{i:03d}_err{err_deg:.1f}.png", vis)
            debug_saved += 1

    print("=" * 60)
    print(f"N={args.n}  ellipse_fit_failures={failures}  ok={len(errs)}")
    if errs:
        a = np.array(errs)
        print(f"axis_err_deg: median={np.median(a):5.2f}  mean={a.mean():5.2f}  "
              f"p90={np.percentile(a,90):5.2f}  max={a.max():5.2f}")
        print(f"<2deg: {(a<2).sum()}/{len(a)}   <5deg: {(a<5).sum()}/{len(a)}   "
              f"<10deg: {(a<10).sum()}/{len(a)}")


if __name__ == "__main__":
    main()
