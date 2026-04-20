#!/usr/bin/env python3
"""
팬텀 위치 그리드 테스트: pre-alignment 성공/실패 히트맵 생성.

팬텀 X×Y 범위를 격자로 나누고, 각 위치에서:
  1. 랜덤 초기 자세 (JOINT_RANGES 기반)
  2. 팬텀 배치 + 회전
  3. pre-alignment 시도 (최대 50s)
  4. 성공/실패 기록

Usage:
    python scripts/test_phantom_grid.py --out /tmp/phantom_grid_test
    python scripts/test_phantom_grid.py --x-range -0.1 0.1 --y-range -0.4 0.0 --bins-x 10 --bins-y 20
    python scripts/test_phantom_grid.py --trials 3  # 셀당 3회 반복 (다른 초기자세)
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import json
import argparse
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Sim", "meca_add.xml")

JOINT_RANGES = [
    (-0.5, 0.5),    # J1 (base rotation)
    (-0.3, 0.3),    # J2 (shoulder pitch)
    (-0.5, 0.2),    # J3 (elbow pitch)
    (-0.3, 0.3),    # J4 (roll)
    (0.4, 1.0),     # J5 (wrist pitch)
    (-1.0, 1.0),    # J6
]

ALIGN_SPEED = 0.08       # m/s (pre-alignment speed)
ALIGN_THRESHOLD_M = 0.002
ALIGN_HOLD_STEPS = 100
MAX_ALIGN_TIME = 30.0     # seconds (timeout per alignment)


def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def run_ik_step(model, data, tip_id, back_id, target_tip, target_back, n_motors, dof, ik_speed=0.5):
    curr_tip = data.site_xpos[tip_id].copy()
    curr_back = data.site_xpos[back_id].copy()

    err_tip = target_tip - curr_tip
    err_back = target_back - curr_back

    tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
    offset_angle = np.deg2rad(180 + 30)
    offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
    current_side_vec = tip_rot_mat @ offset_local_vec

    needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
    target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
    target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
    err_roll = np.cross(current_side_vec, target_side_vec)

    jac_tip_full = np.zeros((6, dof))
    jac_back = np.zeros((3, dof))
    mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
    mujoco.mj_jacSite(model, data, jac_back, None, back_id)

    J_p1 = jac_tip_full[:3, :n_motors]
    e_p1 = err_tip * 50.0
    if np.linalg.norm(e_p1) > 1.0:
        e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
    J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
    dq_p1 = J_p1_pinv @ e_p1

    P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
    J_p2_proj = jac_back[:, :n_motors] @ P_null_1
    dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)

    P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
    J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
    dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))

    data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * ik_speed


def try_alignment(model, data, tip_id, back_id, target_entry_id, target_depth_id,
                  n_motors, dof, phantom_x, phantom_y, phantom_body_id, rotating_id):
    """
    초기자세 → 팬텀 배치 → pre-alignment 시도.
    Returns: (success: bool, final_error_mm: float, reason: str)
    """
    # 1) 랜덤 초기 자세
    mujoco.mj_resetData(model, data)
    home_pose = np.array([np.random.uniform(*r) for r in JOINT_RANGES])
    data.qpos[:6] = home_pose
    mujoco.mj_forward(model, data)

    # 2) 팬텀 배치
    model.body_pos[phantom_body_id] = np.array([phantom_x, phantom_y, 0.0])
    if phantom_y >= -0.25:
        angle_deg = 0
    else:
        angle_deg = -90
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(angle_deg)], "xyz")
    model.body_quat[rotating_id] = new_quat
    mujoco.mj_forward(model, data)

    # 3) Pre-alignment
    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    curr_tip = data.site_xpos[tip_id].copy()
    curr_back = data.site_xpos[back_id].copy()
    needle_len = np.linalg.norm(curr_tip - curr_back)

    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    goal_tip = p_entry - (axis_dir * 0.0001)
    goal_back = p_entry - (axis_dir * (0.0001 + needle_len))

    start_tip = curr_tip.copy()
    start_back = curr_back.copy()
    align_dist = np.linalg.norm(goal_tip - start_tip)
    duration = max(align_dist / ALIGN_SPEED, 0.1)
    start_time = data.time
    hold_timer = 0

    for _ in range(500000):  # safety limit
        t = (data.time - start_time) / duration
        alpha = smooth_step(min(t, 1.0))
        target_tip = (1 - alpha) * start_tip + alpha * goal_tip
        target_back = (1 - alpha) * start_back + alpha * goal_back

        run_ik_step(model, data, tip_id, back_id, target_tip, target_back, n_motors, dof)
        mujoco.mj_step(model, data)

        if t >= 1.0:
            err = np.linalg.norm(data.site_xpos[tip_id] - goal_tip)
            if err < ALIGN_THRESHOLD_M:
                hold_timer += 1
            else:
                hold_timer = 0
            if hold_timer > ALIGN_HOLD_STEPS:
                return True, err * 1000, "OK"

        if data.time - start_time > MAX_ALIGN_TIME:
            final_err = np.linalg.norm(data.site_xpos[tip_id] - goal_tip) * 1000
            return False, final_err, "TIMEOUT"

    final_err = np.linalg.norm(data.site_xpos[tip_id] - goal_tip) * 1000
    return False, final_err, "MAX_ITER"


def main():
    parser = argparse.ArgumentParser(description="Phantom position grid test")
    parser.add_argument("--out", type=str, default="/tmp/phantom_grid_test")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.1, 0.1])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.4, 0.0])
    parser.add_argument("--bins-x", type=int, default=10)
    parser.add_argument("--bins-y", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per cell (different initial poses)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(Path(MODEL_PATH).resolve())
    print(f"Loading: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
    n_motors = model.nu
    dof = model.nv

    # Grid
    x_centers = np.linspace(args.x_range[0], args.x_range[1], args.bins_x)
    y_centers = np.linspace(args.y_range[0], args.y_range[1], args.bins_y)

    total_cells = args.bins_x * args.bins_y
    success_count = np.zeros((args.bins_x, args.bins_y))
    fail_count = np.zeros((args.bins_x, args.bins_y))
    results = []

    print(f"Grid: X[{args.x_range[0]}, {args.x_range[1]}] ({args.bins_x} bins) "
          f"x Y[{args.y_range[0]}, {args.y_range[1]}] ({args.bins_y} bins)")
    print(f"Total: {total_cells} cells x {args.trials} trials = {total_cells * args.trials} tests")
    print()

    test_idx = 0
    total_tests = total_cells * args.trials
    for xi, px in enumerate(x_centers):
        for yi, py in enumerate(y_centers):
            cell_results = []
            for trial in range(args.trials):
                test_idx += 1
                ok, err_mm, reason = try_alignment(
                    model, data, tip_id, back_id, target_entry_id, target_depth_id,
                    n_motors, dof, px, py, phantom_body_id, rotating_id)

                cell_results.append({"ok": ok, "err_mm": round(err_mm, 1), "reason": reason})
                status = "OK" if ok else f"FAIL({reason}, {err_mm:.1f}mm)"
                print(f"  [{test_idx}/{total_tests}] X={px:.3f} Y={py:.3f} trial={trial} {status}")

                if ok:
                    success_count[xi, yi] += 1
                else:
                    fail_count[xi, yi] += 1

            results.append({
                "x": round(px, 4), "y": round(py, 4),
                "xi": xi, "yi": yi,
                "trials": cell_results,
                "success_rate": sum(1 for r in cell_results if r["ok"]) / len(cell_results),
            })

    # Save JSON
    json_path = out_dir / "phantom_grid_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "x_range": args.x_range, "y_range": args.y_range,
            "bins_x": args.bins_x, "bins_y": args.bins_y,
            "trials": args.trials,
            "joint_ranges": JOINT_RANGES,
            "cells": results,
        }, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ================================================================
    # Visualization: Success rate heatmap
    # ================================================================
    success_rate = success_count / args.trials

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1) Success rate heatmap
    ax = axes[0]
    im = ax.imshow(success_rate.T, origin='lower', cmap='RdYlGn',
                   extent=[args.x_range[0], args.x_range[1],
                           args.y_range[0], args.y_range[1]],
                   aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Success rate')

    # Cell text
    for xi in range(args.bins_x):
        for yi in range(args.bins_y):
            rate = success_rate[xi, yi]
            color = 'white' if rate < 0.5 else 'black'
            ax.text(x_centers[xi], y_centers[yi], f'{rate:.0%}',
                    ha='center', va='center', fontsize=6, color=color, fontweight='bold')

    ax.set_xlabel('Phantom X (m)')
    ax.set_ylabel('Phantom Y (m)')
    ax.set_title(f'Pre-alignment Success Rate\n({args.trials} trials/cell, {total_cells} cells)')

    # Y=-0.25 boundary line (rotation switch)
    ax.axhline(y=-0.25, color='cyan', linestyle='--', linewidth=1, label='Rot switch (Y=-0.25)')
    ax.legend(fontsize=8)

    # 2) Failure count heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(fail_count.T, origin='lower', cmap='Reds',
                     extent=[args.x_range[0], args.x_range[1],
                             args.y_range[0], args.y_range[1]],
                     aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Failure count')

    for xi in range(args.bins_x):
        for yi in range(args.bins_y):
            fc = int(fail_count[xi, yi])
            if fc > 0:
                color = 'white' if fc > args.trials / 2 else 'black'
                ax2.text(x_centers[xi], y_centers[yi], str(fc),
                         ha='center', va='center', fontsize=6, color=color, fontweight='bold')

    ax2.set_xlabel('Phantom X (m)')
    ax2.set_ylabel('Phantom Y (m)')
    ax2.set_title(f'Failure Count (max={args.trials})')
    ax2.axhline(y=-0.25, color='cyan', linestyle='--', linewidth=1, label='Rot switch')
    ax2.legend(fontsize=8)

    total_success = int(success_count.sum())
    total_fail = int(fail_count.sum())
    fig.suptitle(f'Phantom Grid Test: {total_success}/{total_tests} success '
                 f'({total_success/total_tests*100:.0f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    img_path = out_dir / "phantom_grid_heatmap.png"
    fig.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved: {img_path}")
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Total: {total_tests} tests ({total_cells} cells x {args.trials} trials)")
    print(f"  Success: {total_success} ({total_success/total_tests*100:.1f}%)")
    print(f"  Failed:  {total_fail} ({total_fail/total_tests*100:.1f}%)")

    # Safe zone suggestion
    safe_cells = [(r["x"], r["y"]) for r in results if r["success_rate"] == 1.0]
    if safe_cells:
        safe_x = [c[0] for c in safe_cells]
        safe_y = [c[1] for c in safe_cells]
        print(f"\n  100% safe zone:")
        print(f"    X: [{min(safe_x):.3f}, {max(safe_x):.3f}]")
        print(f"    Y: [{min(safe_y):.3f}, {max(safe_y):.3f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
