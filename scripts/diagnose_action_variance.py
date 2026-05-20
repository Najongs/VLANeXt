"""Action chunk inference variance diagnostic.

For a fixed observation (one frame from a near-goal state), call predict_action
N times and measure per-dim std of the resulting action chunk. Repeats across
diff_steps to test whether diffusion noise floor is the precision bottleneck.

Hypothesis: if std of dx/dy at near-goal (3mm) is comparable to 5mm/2mm/1mm
target tolerances, then action precision is the real ceiling — not vision/BC.

Usage:
  python -m scripts.diagnose_action_variance \\
      --checkpoint <ckpt.pt> --train-config <cfg.yaml> \\
      --num-samples 50 --diff-steps 10 25 50 100
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

from scripts.sim_eval import load_model, load_processor, predict_action  # noqa: E402
from scripts.sim_eval_align_only import AlignSimEnv, TASK_INSTRUCTION  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def collect_one_obs(env, x_mm, y_mm, z_mm, angle_deg):
    """Reset env using grid_cell convention (matches sim_eval_align_only).

    x_mm/y_mm/z_mm: phantom offset in mm (perturbation from default)
    angle_deg: phantom rotation
    """
    env.reset(grid_cell={"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "angle_deg": angle_deg})
    img = env.render_cameras()
    pose = env.get_ee_pose()
    obs = {
        "full_image": img["tool_camera"],
        "full_image_wrist": img.get("wrist_camera", img["tool_camera"]),
        "image_history": [img["tool_camera"]],
        "image_history_wrist": [img.get("wrist_camera", img["tool_camera"])],
        "state_history": [pose],
        "action_history": [np.zeros(6, dtype=np.float32)],
    }
    dist_mm = env.get_alignment_dist_mm()
    return obs, dist_mm


def measure_variance(model, processor, obs, n_samples):
    """Sample N times, return action chunk stack (N, chunk_len, action_dim)."""
    chunks = []
    for i in range(n_samples):
        with torch.no_grad():
            result = predict_action(model, processor, obs, TASK_INSTRUCTION)
        if isinstance(result, tuple):
            chunk, _ = result
        else:
            chunk = result
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_samples} samples")
    return np.stack(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--config", default="config/sim_eval_align_config.yaml")
    ap.add_argument("--diff-steps", nargs="+", type=int, default=[10, 25, 50])
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--output", default="logs/action_variance_diagnostic.json")
    ap.add_argument("--scheduler-type", default="flow_match")
    args = ap.parse_args()

    # Load eval cfg (just for env construction)
    eval_cfg = OmegaConf.load(args.config)

    # 3 test states by phantom offset (perturbation from default).
    # The robot starts at fixed home pose; phantom shift = initial dist target.
    test_states = [
        dict(label="far_10mm", x_mm=10.0, y_mm=0.0,  z_mm=0.0, angle_deg=0.0),
        dict(label="mid_5mm",  x_mm=5.0,  y_mm=0.0,  z_mm=0.0, angle_deg=0.0),
        dict(label="near_3mm", x_mm=3.0,  y_mm=0.0,  z_mm=0.0, angle_deg=0.0),
    ]

    processor = load_processor(args.checkpoint, train_config_path=args.train_config)

    results = {}
    env = AlignSimEnv("Sim/meca_add.xml", retreat_mm=2.0)

    for diff_steps in args.diff_steps:
        print(f"\n=== diff_steps={diff_steps} ===")
        model = load_model(args.checkpoint, diffusion_steps=diff_steps,
                           scheduler_type=args.scheduler_type,
                           train_config_path=args.train_config)
        model.eval()
        results[diff_steps] = {}

        for state in test_states:
            print(f" State: {state['label']}")
            obs, actual_dist = collect_one_obs(env, state["x_mm"], state["y_mm"],
                                               state["z_mm"], state["angle_deg"])
            print(f"  actual dist={actual_dist:.2f}mm")

            t0 = time.time()
            chunks = measure_variance(model, processor, obs, args.num_samples)
            elapsed = time.time() - t0

            mean_chunk = chunks.mean(axis=0)        # (chunk_len, dim)
            std_chunk = chunks.std(axis=0)          # (chunk_len, dim)

            # Aggregate over chunk dim — first action (most relevant for next step)
            first_step_std = std_chunk[0]           # (dim,)
            mean_step_std = std_chunk.mean(axis=0)  # (dim,)

            # Action chunks are normalized to [-1, 1]. Denorm scale =
            # 10mm typically (action_max_sim ~0.01m per step).
            # Need actual scale from model config for mm conversion.
            results[diff_steps][state["label"]] = dict(
                actual_dist_mm=float(actual_dist),
                n_samples=int(args.num_samples),
                chunk_len=int(chunks.shape[1]),
                first_step_std=first_step_std.tolist(),
                mean_step_std=mean_step_std.tolist(),
                first_step_mean=mean_chunk[0].tolist(),
                elapsed_sec=float(elapsed),
            )

            print(f"  first action std (normalized): {np.array2string(first_step_std, precision=4)}")
            print(f"  first action mean (normalized): {np.array2string(mean_chunk[0], precision=4)}")
            print(f"  mean across chunk: {np.array2string(mean_step_std, precision=4)}")
            print(f"  elapsed: {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    # Write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n\n=== SUMMARY (first-step std, normalized [-1,1] action space) ===")
    print(f"{'diff_steps':>10} {'state':>10} {'dx':>8} {'dy':>8} {'dz':>8} {'rx':>8} {'ry':>8} {'rz':>8}")
    for ds, st_results in results.items():
        for state_label, r in st_results.items():
            s = r["first_step_std"]
            print(f"{ds:>10} {state_label:>10} " + " ".join(f"{v:8.4f}" for v in s[:6]))

    # Action chunk normalization: typical range ±10mm/step → 1 unit ≈ 5mm.
    # If std=0.1 → ~0.5mm uncertainty per step. For 5mm precision target, need std < 0.05
    # per dimension at near-goal.
    print("\nInterpretation: 0.1 std (normalized) ≈ 0.5mm uncertainty (if action range ~10mm).")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
