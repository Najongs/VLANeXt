"""
Grid test for basic motion joint ranges.

Randomized sampling + multiprocessing for fast validation.
Each worker tests one direction across N random joint configs.
Outputs vis_grid_coverage.py style 3D scatter + 2D projection plots.

Usage:
    python scripts/test_basic_motion_grid.py --samples 200 --steps 50 --workers 10
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import json
import argparse
import numpy as np
import mujoco
from math import sqrt
from multiprocessing import Pool
from pathlib import Path
import time


JOINT_RANGES = [
    (-0.5, 0.5),    # J1
    (-0.5, 0.5),    # J2
    (-0.5, 0.5),    # J3
    (-0.3, 0.3),    # J4
    (-0.5, 0.5),    # J5
    (-1.0, 1.0),    # J6
]

DIRECTION_NAMES = [
    "left", "right", "up", "down",
    "left_up", "left_down", "right_up", "right_down",
    "forward", "backward",
]

FINE_ALIGN_SPEED = 0.005
MODEL_PATH = "Sim/meca_add.xml"


def direction_vec(name, x, y, z):
    s2 = 1.0 / sqrt(2)
    d = {
        "left": -y, "right": y, "up": z, "down": -z,
        "left_up": (-y + z) * s2, "left_down": (-y - z) * s2,
        "right_up": (y + z) * s2, "right_down": (y - z) * s2,
        "forward": x, "backward": -x,
    }
    return d[name]


def test_one_direction(args_tuple):
    """Worker: test one direction across N random configs. Returns per-config results."""
    dir_name, n_samples, steps_per_episode, seed = args_tuple

    np.random.seed(seed)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    n_motors = model.nu
    dof = model.nv
    ik_speed = 0.5

    def run_ik_step(target_tip_pos, target_back_pos):
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back

        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
        offset_angle = np.deg2rad(210)
        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
        current_side_vec = tip_rot_mat @ offset_local_vec

        needle_axis = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side = np.cross(needle_axis, np.array([0, 0, 1]))
        norm_ts = np.linalg.norm(target_side)
        target_side = target_side / norm_ts if norm_ts > 1e-3 else np.array([1, 0, 0])
        err_roll = np.cross(current_side_vec, target_side)

        jac_tip_full = np.zeros((6, dof))
        jac_back_j = np.zeros((3, dof))
        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
        mujoco.mj_jacSite(model, data, jac_back_j, None, back_id)

        J1 = jac_tip_full[:3, :n_motors]
        e1 = err_tip * 50.0
        if np.linalg.norm(e1) > 1.0:
            e1 = e1 / np.linalg.norm(e1)
        J1_pinv = np.linalg.pinv(J1, rcond=1e-4)
        dq1 = J1_pinv @ e1
        P1 = np.eye(n_motors) - J1_pinv @ J1
        J2p = jac_back_j[:, :n_motors] @ P1
        dq2 = np.linalg.pinv(J2p, rcond=1e-4) @ ((err_back * 50.0) - jac_back_j[:, :n_motors] @ dq1)
        P2 = P1 - np.linalg.pinv(J2p, rcond=1e-4) @ J2p
        J3p = jac_tip_full[3:, :n_motors] @ P2
        dq3 = np.linalg.pinv(J3p, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq1 + dq2))
        data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq1 + dq2 + dq3) * ik_speed

    # Store per-config results: (joint_config, ee_start_pos_mm, result, reason)
    trial_results = []

    for ci in range(n_samples):
        config = np.array([np.random.uniform(lo, hi) for lo, hi in JOINT_RANGES])

        mujoco.mj_resetData(model, data)
        data.qpos[:6] = config
        mujoco.mj_forward(model, data)

        for _ in range(200):
            data.ctrl[:n_motors] = data.qpos[:n_motors]
            mujoco.mj_step(model, data)

        # Record EE start position in mm
        ee_start = data.xpos[link6_id].copy() * 1000  # m → mm

        ok = True
        reason = ""

        if np.any(np.isnan(data.qpos[:n_motors])) or np.any(np.isnan(data.site_xpos[tip_id])):
            ok = False
            reason = "nan_init"

        prev_tip = data.site_xpos[tip_id].copy() if ok else None

        for step in range(steps_per_episode):
            if not ok:
                break
            for _ in range(67):
                curr_tip = data.site_xpos[tip_id].copy()
                curr_back = data.site_xpos[back_id].copy()
                if np.any(np.isnan(curr_tip)) or np.any(np.isnan(data.qpos[:n_motors])):
                    ok = False
                    reason = "nan"
                    break
                mat = data.xmat[link6_id].reshape(3, 3)
                dw = direction_vec(dir_name, mat[:, 0], mat[:, 1], mat[:, 2])
                dw = dw / (np.linalg.norm(dw) + 1e-10)
                t_tip = curr_tip + dw * FINE_ALIGN_SPEED * model.opt.timestep
                t_back = curr_back + dw * FINE_ALIGN_SPEED * model.opt.timestep
                run_ik_step(t_tip, t_back)
                mujoco.mj_step(model, data)
            if not ok:
                break

            qpos_now = data.qpos[:n_motors]
            for j in range(min(n_motors, model.njnt)):
                jr = model.jnt_range[j]
                if jr[1] - jr[0] < 1e-6:
                    continue
                margin = (jr[1] - jr[0]) * 0.02
                if qpos_now[j] <= jr[0] + margin or qpos_now[j] >= jr[1] - margin:
                    ok = False
                    reason = f"joint_limit_j{j}"
                    break
            if not ok:
                break
            if np.linalg.norm(data.qvel[:n_motors]) > 50.0:
                ok = False
                reason = "ik_diverge_vel"
                break
            curr_check = data.site_xpos[tip_id].copy()
            if np.linalg.norm(curr_check - prev_tip) * 1000 > 10.0:
                ok = False
                reason = "ik_diverge_jump"
                break
            prev_tip = curr_check

        trial_results.append({
            "config": config.tolist(),
            "ee_pos_mm": ee_start.tolist(),
            "ok": ok,
            "reason": reason if not ok else "",
        })

    return dir_name, trial_results


def save_plots(all_results, output_dir):
    """vis_grid_coverage.py style: 3D scatter + 2D projections per direction + combined."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined plot (all directions) ---
    all_success = []
    all_fail_ik = []
    all_fail_jl = []

    for dir_name, trials in all_results:
        for t in trials:
            pt = t["ee_pos_mm"]
            if t["ok"]:
                all_success.append(pt)
            elif "ik_diverge" in t["reason"] or "nan" in t["reason"]:
                all_fail_ik.append(pt)
            else:
                all_fail_jl.append(pt)

    all_success = np.array(all_success) if all_success else np.empty((0, 3))
    all_fail_ik = np.array(all_fail_ik) if all_fail_ik else np.empty((0, 3))
    all_fail_jl = np.array(all_fail_jl) if all_fail_jl else np.empty((0, 3))

    n_s, n_ik, n_jl = len(all_success), len(all_fail_ik), len(all_fail_jl)
    total = n_s + n_ik + n_jl

    fig = plt.figure(figsize=(16, 12))

    # 3D scatter
    ax1 = fig.add_subplot(221, projection='3d')
    if n_s > 0:
        ax1.scatter(all_success[:, 0], all_success[:, 1], all_success[:, 2],
                    c='dodgerblue', alpha=0.3, s=10, label=f'Success ({n_s})')
    if n_ik > 0:
        ax1.scatter(all_fail_ik[:, 0], all_fail_ik[:, 1], all_fail_ik[:, 2],
                    c='red', alpha=0.8, s=40, marker='x', label=f'IK_FAIL ({n_ik})')
    if n_jl > 0:
        ax1.scatter(all_fail_jl[:, 0], all_fail_jl[:, 1], all_fail_jl[:, 2],
                    c='orange', alpha=0.8, s=40, marker='^', label=f'JointLimit ({n_jl})')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D EE Start Positions')
    ax1.legend(fontsize=8)

    # 2D projections with failure density heatmap
    all_pts = np.concatenate([p for p in [all_success, all_fail_ik, all_fail_jl] if len(p) > 0])
    fail_pts = np.concatenate([p for p in [all_fail_ik, all_fail_jl] if len(p) > 0]) if (n_ik + n_jl) > 0 else np.empty((0, 3))

    projections = [
        (222, 'XY Projection (top view)', 0, 1, 'X (mm)', 'Y (mm)'),
        (223, 'XZ Projection (front view)', 0, 2, 'X (mm)', 'Z (mm)'),
        (224, 'YZ Projection (side view)', 1, 2, 'Y (mm)', 'Z (mm)'),
    ]

    n_bins = 10
    for subplot, title, ai, aj, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot)

        range_i = (all_pts[:, ai].min() - 5, all_pts[:, ai].max() + 5)
        range_j = (all_pts[:, aj].min() - 5, all_pts[:, aj].max() + 5)

        # Failure density heatmap
        fail_count = np.zeros((n_bins, n_bins))
        total_count = np.zeros((n_bins, n_bins))

        for pt in all_pts:
            ii = min(int((pt[ai] - range_i[0]) / (range_i[1] - range_i[0]) * n_bins), n_bins - 1)
            jj = min(int((pt[aj] - range_j[0]) / (range_j[1] - range_j[0]) * n_bins), n_bins - 1)
            ii = max(0, ii)
            jj = max(0, jj)
            total_count[ii, jj] += 1

        if len(fail_pts) > 0:
            for pt in fail_pts:
                ii = min(int((pt[ai] - range_i[0]) / (range_i[1] - range_i[0]) * n_bins), n_bins - 1)
                jj = min(int((pt[aj] - range_j[0]) / (range_j[1] - range_j[0]) * n_bins), n_bins - 1)
                ii = max(0, ii)
                jj = max(0, jj)
                fail_count[ii, jj] += 1

        fail_ratio = np.divide(fail_count, total_count, where=total_count > 0,
                               out=np.zeros_like(fail_count))

        im = ax.imshow(fail_ratio.T, origin='lower', cmap='RdYlGn_r',
                       extent=[range_i[0], range_i[1], range_j[0], range_j[1]],
                       aspect='auto', vmin=0, vmax=max(0.1, fail_ratio.max()))
        plt.colorbar(im, ax=ax, label='Failure rate')

        # Show rate text in bins with failures
        edges_i = np.linspace(range_i[0], range_i[1], n_bins + 1)
        edges_j = np.linspace(range_j[0], range_j[1], n_bins + 1)
        for ii in range(n_bins):
            for jj in range(n_bins):
                rate = fail_ratio[ii, jj]
                if rate > 0:
                    ci = (edges_i[ii] + edges_i[ii + 1]) / 2
                    cj = (edges_j[jj] + edges_j[jj + 1]) / 2
                    color = 'white' if rate > 0.5 else 'black'
                    ax.text(ci, cj, f'{rate:.0%}', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(f'Basic Motion Grid Test: {n_s}/{total} success ({n_s/total*100:.0f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / "basic_motion_grid_coverage.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot → {out_path}")
    plt.close()

    # --- Per-direction bar chart ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    dir_names = [r[0] for r in all_results]
    ok_pcts = [100 * sum(1 for t in r[1] if t["ok"]) / len(r[1]) for r in all_results]

    bars = ax2.bar(dir_names, ok_pcts,
                   color=['green' if p > 80 else 'orange' if p > 50 else 'red' for p in ok_pcts])
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title(f"Per-Direction Success Rate ({len(all_results[0][1])} configs/dir)")
    ax2.set_ylim(0, 105)
    for bar, pct in zip(bars, ok_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.0f}%", ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path2 = output_dir / "basic_motion_per_direction.png"
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved per-direction plot → {out_path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grid test for basic motion (parallel)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Random joint configs per direction (default: 200)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Control steps per trial (default: 50)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers (default: 10, one per direction)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/tmp/basic_motion_grid",
                        help="Output directory for plots and results")
    args = parser.parse_args()

    print(f"Testing {args.samples} random configs x 10 directions = {args.samples * 10} trials")
    print(f"Steps per trial: {args.steps}, Workers: {args.workers}")
    print()

    tasks = [
        (d, args.samples, args.steps, args.seed + i * 1000)
        for i, d in enumerate(DIRECTION_NAMES)
    ]

    t0 = time.time()
    with Pool(processes=min(args.workers, len(DIRECTION_NAMES))) as pool:
        all_results = pool.map(test_one_direction, tasks)
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"Results (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")

    total_ok = 0
    total_n = 0
    all_fail_reasons = {}

    for dir_name, trials in all_results:
        n = len(trials)
        ok = sum(1 for t in trials if t["ok"])
        pct = 100 * ok / n
        total_ok += ok
        total_n += n
        print(f"  {dir_name:12s}: {ok:4d}/{n} OK ({pct:5.1f}%)", end="")
        fails = {}
        for t in trials:
            if not t["ok"]:
                fails[t["reason"]] = fails.get(t["reason"], 0) + 1
        if fails:
            top = max(fails, key=fails.get)
            print(f"  top_fail: {top}({fails[top]})", end="")
        print()
        for r, c in fails.items():
            all_fail_reasons[r] = all_fail_reasons.get(r, 0) + c

    print(f"\n  TOTAL: {total_ok}/{total_n} OK ({100*total_ok/total_n:.1f}%)")

    if all_fail_reasons:
        print(f"\nFailure reasons:")
        for reason, count in sorted(all_fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Save results JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "grid_test_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "total_ok": total_ok,
            "total_n": total_n,
            "fail_reasons": all_fail_reasons,
            "per_direction": {d: {"ok": sum(1 for t in ts if t["ok"]), "n": len(ts)}
                              for d, ts in all_results},
        }, f, indent=2)
    print(f"\nResults JSON → {results_path}")

    # Save plots
    save_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
