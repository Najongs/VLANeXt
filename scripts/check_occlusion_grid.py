#!/usr/bin/env python3
"""
Perturbation grid 위에서 occlusion(needle tip ↔ tool_camera) 발생률을 측정.

Real 수집 전에 sim_eval_align_only.py / Save_dataset_align_only.py와 동일한 IK/팬텀
세팅으로 어떤 (X, Y, Z) 영역이 occlusion 위험 영역인지 확인하기 위한 도구.
시각화는 vis_grid_coverage.py 스타일 (3D scatter + XY/XZ/YZ heatmap).

Usage:
    python -m scripts.check_occlusion_grid \
        --phantom-pos 0.0 -0.4 \
        --xy-mm 40 --z-min-mm -20 --z-max-mm 20 \
        --bins-xy 8 --bins-z 6 --samples-per-cell 5 \
        --angle-deg 10 \
        --out-dir dataset/occlusion_check
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.sim_eval_align_only import AlignSimEnv
from scripts.sim_eval import SIM_MODEL_PATH, smooth_step


def apply_perturbation(env, perturb_xyz, perturb_angle_rad, random_axis,
                       move_speed=0.05, max_steps=5000):
    """Reset to aligned state and IK-drive to (goal + perturb_xyz, rotated by angle).
    Returns (converged, ik_err_mm)."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:env.n_motors] = env._aligned_qpos
    env.data.qvel[:env.n_motors] = env._aligned_qvel
    mujoco.mj_forward(env.model, env.data)

    perturbed_tip = env._goal_tip + perturb_xyz
    rot_mat_perturb = np.eye(3)
    if abs(perturb_angle_rad) > 1e-6:
        K = np.array([
            [0, -random_axis[2], random_axis[1]],
            [random_axis[2], 0, -random_axis[0]],
            [-random_axis[1], random_axis[0], 0],
        ])
        rot_mat_perturb = (np.eye(3)
                           + np.sin(perturb_angle_rad) * K
                           + (1 - np.cos(perturb_angle_rad)) * (K @ K))
    perturbed_back_dir = rot_mat_perturb @ (env._goal_back - env._goal_tip)
    perturbed_back = perturbed_tip + perturbed_back_dir

    move_dist = np.linalg.norm(perturbed_tip - env._goal_tip)
    move_duration = max(move_dist / move_speed, 0.1)
    move_start_time = env.data.time

    converged = False
    for ps in range(max_steps):
        t = (env.data.time - move_start_time) / move_duration
        alpha = smooth_step(min(t, 1.0))
        interp_tip = (1 - alpha) * env._goal_tip + alpha * perturbed_tip
        interp_back = (1 - alpha) * env._goal_back + alpha * perturbed_back
        env._run_ik_step(interp_tip, interp_back)
        mujoco.mj_step(env.model, env.data)

        if t >= 1.0:
            if np.linalg.norm(env.data.site_xpos[env.tip_id] - perturbed_tip) < 0.001:
                for _ in range(50):
                    env._run_ik_step(perturbed_tip, perturbed_back)
                    mujoco.mj_step(env.model, env.data)
                converged = True
                break
            if ps > max_steps - 100:
                break

    ik_err_mm = np.linalg.norm(env.data.site_xpos[env.tip_id] - perturbed_tip) * 1000
    return converged, ik_err_mm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phantom-pos", nargs=2, type=float, default=[0.0, -0.4],
                   help="Fixed phantom (x, y) in robot base frame, meters.")
    p.add_argument("--xy-mm", type=float, default=40.0,
                   help="±XY perturbation range (mm). Match Save_dataset_align_only.PERTURB_POS_XY_MM.")
    p.add_argument("--z-min-mm", type=float, default=-20.0)
    p.add_argument("--z-max-mm", type=float, default=20.0)
    p.add_argument("--angle-deg", type=float, default=10.0,
                   help="±angle perturbation per random axis.")
    p.add_argument("--bins-xy", type=int, default=8)
    p.add_argument("--bins-z", type=int, default=6)
    p.add_argument("--samples-per-cell", type=int, default=5,
                   help="Random angle/axis samples per cell to estimate occlusion rate.")
    p.add_argument("--retreat-mm", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="dataset/occlusion_check")
    p.add_argument("--xml", type=str, default=os.path.abspath(SIM_MODEL_PATH))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = AlignSimEnv(
        args.xml,
        randomize_phantom=False,
        use_sensor_success=False,
        phantom_pos=tuple(args.phantom_pos),
        retreat_mm=args.retreat_mm,
    )
    print(f"[setup] phantom_pos={tuple(args.phantom_pos)}, retreat={args.retreat_mm}mm")
    env._ensure_aligned_state()
    print("[setup] pre-alignment done.")

    # --- Build perturbation grid (cell centers) ---
    x_centers = np.linspace(-args.xy_mm, args.xy_mm, args.bins_xy * 2 + 1)[1::2]
    y_centers = np.linspace(-args.xy_mm, args.xy_mm, args.bins_xy * 2 + 1)[1::2]
    z_centers = np.linspace(args.z_min_mm, args.z_max_mm, args.bins_z * 2 + 1)[1::2]
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    cells_mm = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n_cells = len(cells_mm)
    print(f"[grid] {args.bins_xy}x{args.bins_xy}x{args.bins_z} = {n_cells} cells, "
          f"{args.samples_per_cell} samples/cell, total trials={n_cells*args.samples_per_cell}")

    occ_count = np.zeros(n_cells, dtype=np.int32)
    ik_fail_count = np.zeros(n_cells, dtype=np.int32)
    total_count = np.zeros(n_cells, dtype=np.int32)

    for ci, center_mm in enumerate(cells_mm):
        for s in range(args.samples_per_cell):
            perturb_xyz = center_mm / 1000.0  # mm → m
            angle_rad = np.deg2rad(rng.uniform(-args.angle_deg, args.angle_deg))
            axis = rng.standard_normal(3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)

            converged, ik_err = apply_perturbation(env, perturb_xyz, angle_rad, axis)
            total_count[ci] += 1
            if not converged:
                ik_fail_count[ci] += 1
                continue
            if env._check_tip_occluded():
                occ_count[ci] += 1

        if (ci + 1) % max(1, n_cells // 20) == 0 or ci == n_cells - 1:
            print(f"  cell {ci+1}/{n_cells} ({(ci+1)/n_cells*100:.0f}%): "
                  f"this_cell occ={occ_count[ci]}/{total_count[ci]} ik_fail={ik_fail_count[ci]}")

    occ_rate = np.where(total_count > 0, occ_count / total_count, 0.0)
    ik_fail_rate = np.where(total_count > 0, ik_fail_count / total_count, 0.0)

    n_total = int(total_count.sum())
    n_occ = int(occ_count.sum())
    n_ikf = int(ik_fail_count.sum())
    print(f"\n[summary] trials={n_total}, occluded={n_occ} ({n_occ/n_total*100:.1f}%), "
          f"ik_fail={n_ikf} ({n_ikf/n_total*100:.1f}%)")

    # --- Save raw data ---
    np.savez(
        out_dir / "occlusion_grid.npz",
        cells_mm=cells_mm,
        occ_count=occ_count,
        ik_fail_count=ik_fail_count,
        total_count=total_count,
        occ_rate=occ_rate,
        ik_fail_rate=ik_fail_rate,
        bins_xy=args.bins_xy,
        bins_z=args.bins_z,
        xy_mm=args.xy_mm,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        angle_deg=args.angle_deg,
        samples_per_cell=args.samples_per_cell,
        phantom_pos=np.array(args.phantom_pos),
    )
    with open(out_dir / "occlusion_grid_summary.json", "w") as f:
        json.dump({
            "phantom_pos": args.phantom_pos,
            "xy_mm": args.xy_mm,
            "z_min_mm": args.z_min_mm,
            "z_max_mm": args.z_max_mm,
            "angle_deg": args.angle_deg,
            "bins_xy": args.bins_xy,
            "bins_z": args.bins_z,
            "samples_per_cell": args.samples_per_cell,
            "n_total": n_total,
            "n_occluded": n_occ,
            "n_ik_fail": n_ikf,
            "occ_pct": n_occ / n_total * 100 if n_total else 0,
            "ik_fail_pct": n_ikf / n_total * 100 if n_total else 0,
        }, f, indent=2)

    # --- Plot: 3D scatter + 2D projections ---
    fig = plt.figure(figsize=(16, 12))

    # 3D scatter colored by occlusion rate
    ax1 = fig.add_subplot(221, projection="3d")
    sc = ax1.scatter(cells_mm[:, 0], cells_mm[:, 1], cells_mm[:, 2],
                     c=occ_rate, cmap="RdYlGn_r", vmin=0, vmax=1,
                     s=40, alpha=0.85)
    if (ik_fail_rate > 0).any():
        m = ik_fail_rate > 0
        ax1.scatter(cells_mm[m, 0], cells_mm[m, 1], cells_mm[m, 2],
                    facecolors="none", edgecolors="black", s=80, linewidths=1.0,
                    label=f"IK fail (n={int((ik_fail_count>0).sum())})")
        ax1.legend(fontsize=8, loc="upper left")
    plt.colorbar(sc, ax=ax1, label="Occlusion rate", shrink=0.7)
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)"); ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D Occlusion (color=rate, ○=any IK fail)")

    # 2D projection helper
    def _heat(ax, axis_i, axis_j, title, xlabel, ylabel,
              bins_i, bins_j, range_i_lo, range_i_hi, range_j_lo, range_j_hi):
        edges_i = np.linspace(range_i_lo, range_i_hi, bins_i + 1)
        edges_j = np.linspace(range_j_lo, range_j_hi, bins_j + 1)
        occ_sum = np.zeros((bins_i, bins_j))
        cnt_sum = np.zeros((bins_i, bins_j))
        for ci, c in enumerate(cells_mm):
            ii = min(int((c[axis_i] - range_i_lo)
                         / (range_i_hi - range_i_lo) * bins_i), bins_i - 1)
            jj = min(int((c[axis_j] - range_j_lo)
                         / (range_j_hi - range_j_lo) * bins_j), bins_j - 1)
            occ_sum[ii, jj] += occ_count[ci]
            cnt_sum[ii, jj] += total_count[ci]
        rate = np.divide(occ_sum, cnt_sum, where=cnt_sum > 0, out=np.zeros_like(occ_sum))

        im = ax.imshow(rate.T, origin="lower", cmap="RdYlGn_r",
                       extent=[range_i_lo, range_i_hi, range_j_lo, range_j_hi],
                       aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Occlusion rate")
        for ii in range(bins_i):
            for jj in range(bins_j):
                ci_c = (edges_i[ii] + edges_i[ii + 1]) / 2
                cj_c = (edges_j[jj] + edges_j[jj + 1]) / 2
                r = rate[ii, jj]
                if r > 0:
                    color = "white" if r > 0.5 else "black"
                    ax.text(ci_c, cj_c, f"{r:.0%}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

    ax2 = fig.add_subplot(222)
    _heat(ax2, 0, 1, "XY (top view)", "X (mm)", "Y (mm)",
          args.bins_xy, args.bins_xy, -args.xy_mm, args.xy_mm, -args.xy_mm, args.xy_mm)
    ax3 = fig.add_subplot(223)
    _heat(ax3, 0, 2, "XZ (front view)", "X (mm)", "Z (mm)",
          args.bins_xy, args.bins_z, -args.xy_mm, args.xy_mm, args.z_min_mm, args.z_max_mm)
    ax4 = fig.add_subplot(224)
    _heat(ax4, 1, 2, "YZ (side view)", "Y (mm)", "Z (mm)",
          args.bins_xy, args.bins_z, -args.xy_mm, args.xy_mm, args.z_min_mm, args.z_max_mm)

    fig.suptitle(
        f"Occlusion Grid — phantom={tuple(args.phantom_pos)}  "
        f"XY±{args.xy_mm}mm Z[{args.z_min_mm},{args.z_max_mm}]mm "
        f"angle±{args.angle_deg}°  occ={n_occ}/{n_total} ({n_occ/n_total*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "occlusion_grid.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")
    print(f"[saved] {out_dir/'occlusion_grid.npz'}")
    print(f"[saved] {out_dir/'occlusion_grid_summary.json'}")


if __name__ == "__main__":
    main()
