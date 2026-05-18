"""Dump REALISTIC sim frames for VQA: roll out a VLA checkpoint and snapshot
frames whose lateral_mm falls in a target band (e.g. 1-15mm — the handoff regime).

Per episode we save at most one frame per lateral-band bucket (closest to bucket center).

Usage:
    python -m scripts.dump_vqa_frames_rollout \\
        --checkpoint .../checkpoint_7500.pt \\
        --train-config config/sim_train_align_siglip2_b24_ft10mm_NEW_finetune_config.yaml \\
        --xy-steps 2 --z-steps 1 --angle-steps 3 --repeats 1 \\
        --max-steps 250 --bands 2,5,10,15 \\
        --out-dir vqa_samples/run02
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# Reuse existing eval scaffolding.
from scripts.sim_eval_align_only import (
    AlignSimEnv, build_perturb_grid, SIM_MODEL_PATH,
    load_model_and_processor, predict_action, IMG_WIDTH, IMG_HEIGHT,
)


def discretize_direction(du, dv, centered_px_thresh=8.0):
    mag = float(np.hypot(du, dv))
    if mag < centered_px_thresh:
        return "centered"
    ang = np.degrees(np.arctan2(-dv, du))
    ang = (ang + 360.0) % 360.0
    sectors = [
        ("right", 0.0), ("up-right", 45.0), ("up", 90.0), ("up-left", 135.0),
        ("left", 180.0), ("down-left", 225.0), ("down", 270.0), ("down-right", 315.0),
    ]
    return min(sectors, key=lambda s: min(abs(ang - s[1]), 360.0 - abs(ang - s[1])))[0]


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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--xy-steps", type=int, default=2)
    p.add_argument("--z-steps", type=int, default=1)
    p.add_argument("--angle-steps", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bands", default="2,5,10,15",
                   help="Comma-separated upper bounds of lateral-mm buckets")
    args = p.parse_args()

    bands = [float(x) for x in args.bands.split(",")]
    bands = sorted(bands)
    print(f"Lateral bands (upper bounds, mm): {bands}")

    out_dir = Path(args.out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    with open(args.train_config) as f:
        cfg = yaml.safe_load(f)

    print("Loading model ...")
    model, processor, action_min, action_max = load_model_and_processor(
        args.checkpoint, cfg
    )
    model.eval()

    env = AlignSimEnv(os.path.abspath(SIM_MODEL_PATH))
    grid_cells = build_perturb_grid(args.xy_steps, args.z_steps, args.angle_steps, args.repeats)
    print(f"{len(grid_cells)} grid cells")

    records = []
    for ep_idx, cell in enumerate(grid_cells):
        try:
            env.reset(grid_cell=cell)
        except Exception as e:
            print(f"[ep{ep_idx:03d}] reset failed: {e}")
            continue

        # Per-episode best-frame-per-band tracker
        best = {b: None for b in bands}  # band_upper -> (lateral_mm, frame_data)

        history = []  # observation history if model uses it
        for step in range(args.max_steps):
            frames = env.render_cameras()
            img = frames["tool_camera"]
            m = env.get_spatial_metrics()
            lat = float(m["lateral_mm"])
            tip_uv_n = m["tip_uv"]; trocar_uv_n = m["trocar_uv"]
            if tip_uv_n is None or trocar_uv_n is None:
                continue
            tip_uv = [float(tip_uv_n[0]) * 256, float(tip_uv_n[1]) * 256]
            trocar_uv = [float(trocar_uv_n[0]) * 256, float(trocar_uv_n[1]) * 256]
            trocar_in_frame = (0 <= trocar_uv[0] < 256) and (0 <= trocar_uv[1] < 256)

            # Bucket assignment
            for b in bands:
                if lat < b and trocar_in_frame:
                    # closer to b/2 -> better (so dist = |lat - b/2|)
                    target = b * 0.6
                    score = abs(lat - target)
                    cur = best[b]
                    if cur is None or score < cur[0]:
                        best[b] = (score, {
                            "step": step,
                            "lateral_mm": lat,
                            "angle_deg": float(m["angle_deg"]),
                            "dist_mm": float(m["dist_mm"]),
                            "tip_uv": tip_uv, "trocar_uv": trocar_uv,
                            "delta_uv": [trocar_uv[0]-tip_uv[0], trocar_uv[1]-tip_uv[1]],
                            "sensor_dist_mm": float(m["sensor_dist_mm"]) if m["sensor_dist_mm"] is not None else None,
                            "img": img.copy(),
                        })
                    break  # smallest matching band only

            # Step policy
            try:
                obs = {"tool_camera": img, "ee_pose": env.get_ee_pose()}
                action = predict_action(model, processor, obs, history, cfg, action_min, action_max)
                env.apply_delta_ee(action, n_sim_steps=67)
                history.append(obs)
                if len(history) > 8:
                    history = history[-8:]
            except Exception as e:
                print(f"[ep{ep_idx:03d}] policy step {step} failed: {e}")
                break

            # Early exit if very close
            if lat < 0.5 and m["angle_deg"] < 3:
                break

        # Save best frames for this ep
        for b in bands:
            entry = best[b]
            if entry is None:
                continue
            _, d = entry
            fname = f"ep{ep_idx:03d}_band{int(b)}.png"
            Image.fromarray(d["img"]).save(out_dir / "frames" / fname)
            du, dv = d["delta_uv"]
            gt_dir = discretize_direction(du, dv)
            gt_mag = discretize_magnitude_mm(d["lateral_mm"])
            rec = {
                "ep": ep_idx, "band_upper_mm": b, "step": d["step"],
                "frame": str(Path("frames") / fname),
                "perturb_cell": dict(cell) if cell is not None else None,
                "tip_uv": d["tip_uv"], "trocar_uv": d["trocar_uv"],
                "delta_uv": d["delta_uv"], "trocar_in_frame": True,
                "lateral_mm": d["lateral_mm"], "angle_deg": d["angle_deg"],
                "dist_mm": d["dist_mm"], "sensor_dist_mm": d["sensor_dist_mm"],
                "gt_direction": gt_dir, "gt_magnitude": gt_mag,
            }
            records.append(rec)
            print(f"  saved [ep{ep_idx:03d} band≤{b:.0f}] step={d['step']:3d} "
                  f"lat={d['lateral_mm']:5.2f} ang={d['angle_deg']:5.1f} "
                  f"du={du:+5.1f} dv={dv:+5.1f} -> {gt_dir}/{gt_mag}")

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump({"samples": records, "image_size": [256, 256]}, f, indent=2)
    print(f"\nDumped {len(records)} samples to {out_dir}")


if __name__ == "__main__":
    main()
