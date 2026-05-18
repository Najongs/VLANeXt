"""Dump sim frames + ground-truth metrics for Qwen3.5-VL direction VQA sanity.

Spins up AlignSimEnv across a perturbation grid, takes the *initial* state
(no policy roll-out), renders tool_camera, and saves PNG + GT JSON sidecar.

Each sample's GT contains everything needed to grade VLM direction answers:
- tip_uv, trocar_uv (pixel coords in 256x256)
- lateral_mm, angle_deg, dist_mm
- delta_uv = trocar_uv - tip_uv (the direction we want VLM to recover)
- discrete_dir = 8-way label (up/down/.../upper-right) or "centered"

Usage:
    python -m scripts.dump_vqa_frames \\
        --train-config config/sim_train_align_siglip2_b24_ft10mm_NEW_finetune_v2_config.yaml \\
        --xy-steps 3 --z-steps 2 --angle-steps 3 --repeats 1 \\
        --out-dir vqa_samples/run01
"""
import argparse
import json
import os
from pathlib import Path

import os
import numpy as np
from PIL import Image
import yaml

from scripts.sim_eval_align_only import AlignSimEnv, build_perturb_grid, SIM_MODEL_PATH


def discretize_direction(du, dv, centered_px_thresh=8.0):
    """8-way + centered label from (du, dv) in pixel coords.
    Image convention: u right (+), v down (+).
    Returns one of: up, down, left, right, up-left, up-right, down-left, down-right, centered.
    """
    mag = float(np.hypot(du, dv))
    if mag < centered_px_thresh:
        return "centered"
    ang = np.degrees(np.arctan2(-dv, du))  # convert image-y-down to math-y-up
    ang = (ang + 360.0) % 360.0
    sectors = [
        ("right", 0.0), ("up-right", 45.0), ("up", 90.0), ("up-left", 135.0),
        ("left", 180.0), ("down-left", 225.0), ("down", 270.0), ("down-right", 315.0),
    ]
    best = min(sectors, key=lambda s: min(abs(ang - s[1]), 360.0 - abs(ang - s[1])))
    return best[0]


def discretize_magnitude_mm(lateral_mm):
    if lateral_mm < 2.0:
        return "tiny"
    if lateral_mm < 5.0:
        return "small"
    if lateral_mm < 10.0:
        return "medium"
    return "large"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-config", default=None, help="(unused, kept for CLI compat)")
    p.add_argument("--xy-steps", type=int, default=3)
    p.add_argument("--z-steps", type=int, default=2)
    p.add_argument("--angle-steps", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--perturb-mode", default="grid")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    model_xml = os.path.abspath(SIM_MODEL_PATH)
    env = AlignSimEnv(model_xml)
    grid_cells = build_perturb_grid(
        args.xy_steps, args.z_steps, args.angle_steps, args.repeats,
    )

    records = []
    for ep_idx, cell in enumerate(grid_cells):
        try:
            env.reset(grid_cell=cell)
        except Exception as e:
            print(f"[ep{ep_idx:03d}] reset failed: {e}")
            continue

        frames = env.render_cameras()
        img = frames["tool_camera"]
        m = env.get_spatial_metrics()

        # project_to_2d returns normalized [0,1]; convert to pixel coords (256x256).
        IMG = 256
        tip_uv_norm = m["tip_uv"]
        trocar_uv_norm = m["trocar_uv"]
        if tip_uv_norm is None or trocar_uv_norm is None:
            print(f"[ep{ep_idx:03d}] projection failed, skip")
            continue
        tip_uv = [float(tip_uv_norm[0]) * IMG, float(tip_uv_norm[1]) * IMG]
        trocar_uv = [float(trocar_uv_norm[0]) * IMG, float(trocar_uv_norm[1]) * IMG]

        du = trocar_uv[0] - tip_uv[0]
        dv = trocar_uv[1] - tip_uv[1]
        trocar_in_frame = (0 <= trocar_uv[0] < IMG) and (0 <= trocar_uv[1] < IMG)
        gt_dir = discretize_direction(du, dv)
        gt_mag = discretize_magnitude_mm(m["lateral_mm"])

        fname = f"ep{ep_idx:03d}.png"
        Image.fromarray(img).save(out_dir / "frames" / fname)
        records.append({
            "ep": ep_idx,
            "frame": str(Path("frames") / fname),
            "perturb_cell": list(cell) if cell is not None else None,
            "tip_uv": tip_uv,
            "trocar_uv": trocar_uv,
            "delta_uv": [du, dv],
            "trocar_in_frame": bool(trocar_in_frame),
            "lateral_mm": float(m["lateral_mm"]),
            "angle_deg": float(m["angle_deg"]),
            "dist_mm": float(m["dist_mm"]),
            "sensor_dist_mm": float(m["sensor_dist_mm"]) if m["sensor_dist_mm"] is not None else None,
            "gt_direction": gt_dir,
            "gt_magnitude": gt_mag,
        })
        print(f"[ep{ep_idx:03d}] lateral={m['lateral_mm']:.2f}mm angle={m['angle_deg']:.1f}deg "
              f"du={du:+.1f} dv={dv:+.1f} -> {gt_dir}/{gt_mag}")

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump({"samples": records, "image_size": [256, 256]}, f, indent=2)
    print(f"\nDumped {len(records)} samples to {out_dir}")


if __name__ == "__main__":
    main()
