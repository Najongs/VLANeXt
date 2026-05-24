"""y=-25 region self-occlusion sanity check.

Render tool_camera view at the failing state (~5mm short of goal) for each
eval grid cell with y ∈ {-25, 0, +25}, x ∈ {-10, 0, +10}, angle=0 (single PNG).

Purpose: visually check if needle shaft / robot link / gripper occludes
the trocar entry hole from tool_camera POV when the phantom is at y=-25.

Background: 2026-05-20 cell-by-cell analysis showed y=-25 cells fail
(mean 8.66mm, 2/9 SR), root cause partially attributed to approach_00 PHANTOM_Y
distribution bias. This check tests an ADDITIONAL hypothesis: self-occlusion
when robot reaches the failing position.

Output:
  vqa_samples/yneg_occlusion_check.png  (3 rows × 3 cols)
"""
from __future__ import annotations
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.visualize_robot_perturbation_clean import (
    make_ik, smooth_step, MODEL_PATH, IMG_W, IMG_H,
    ALIGN_SPEED, ALIGN_THRESHOLD_M, ALIGN_HOLD_STEPS,
)

OUT_DIR = Path("/data/public/NAS/VLANeXt/vqa_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Render at this distance from trocar entry (eval retreat)
STUCK_DIST_MM = 2.0  # eval retreat=2mm (perfect alignment baseline)
HOME_POSE = np.array([0.75, -0.5, 0.5, 0, 0.6, 1.0])  # eval starting pose


def set_phantom(model, data, phantom_id, rot_id, x_mm, y_mm, z_mm, angle_deg):
    """Phantom 위치/각도 설정 (mm/deg → m/rad)"""
    model.body_pos[phantom_id] = np.array([x_mm/1000.0, y_mm/1000.0, z_mm/1000.0])
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat


def align_to_phantom(model, data, run_ik, tip_id, back_id, target_entry_id,
                     target_depth_id, n_motors, stuck_dist_mm=STUCK_DIST_MM):
    """Phantom 기준 needle을 trocar entry에서 stuck_dist_mm만큼 떨어진 위치로 IK 정렬"""
    mujoco.mj_forward(model, data)
    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])
    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    # Goal: stuck_dist_mm 떨어진 정확한 axis 상 위치 (이상적 정렬 - 거리만 멈)
    goal_tip = p_entry - axis_dir * (stuck_dist_mm / 1000.0)
    goal_back = p_entry - axis_dir * (stuck_dist_mm / 1000.0 + needle_len)

    start_tip = data.site_xpos[tip_id].copy()
    start_back = data.site_xpos[back_id].copy()
    dur = max(np.linalg.norm(goal_tip - start_tip) / ALIGN_SPEED, 2.0)
    t0 = data.time
    timer = 0
    while True:
        p = smooth_step((data.time - t0) / dur) if dur > 0 else 1.0
        run_ik((1-p)*start_tip + p*goal_tip, (1-p)*start_back + p*goal_back)
        mujoco.mj_step(model, data)
        if p >= 1.0:
            if np.linalg.norm(data.site_xpos[tip_id] - goal_tip) < ALIGN_THRESHOLD_M:
                timer += 1
            else:
                timer = 0
            if timer > ALIGN_HOLD_STEPS:
                return True
        if data.time - t0 > 60:
            return False


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    phantom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Phantom")
    rot_id = phantom_id  # phantom의 body가 rot 컨테이너 역할

    n_motors = model.nu
    dof = model.nv
    run_ik = make_ik(model, data, tip_id, back_id, n_motors, dof)

    # Grid: rows = y_phantom_mm, cols = x_phantom_mm. angle = 0 fixed.
    y_vals_mm = [-25, 0, 25]
    x_vals_mm = [-10, 0, 10]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ri, y_mm in enumerate(y_vals_mm):
        for ci, x_mm in enumerate(x_vals_mm):
            print(f"\nCell: x={x_mm:+}, y={y_mm:+}, angle=0  rendering...")
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = HOME_POSE.copy()  # eval starting pose
            data.ctrl[:6] = HOME_POSE.copy()
            set_phantom(model, data, phantom_id, rot_id, x_mm, y_mm, 0, 0)
            ok = align_to_phantom(model, data, run_ik, tip_id, back_id,
                                   target_entry_id, target_depth_id, n_motors)
            renderer.update_scene(data, camera='tool_camera')
            img = renderer.render()
            ax = axes[ri, ci]
            ax.imshow(img)
            title = f"phantom (x={x_mm:+}, y={y_mm:+})mm"
            if not ok:
                title += " [IK_FAIL]"
            ax.set_title(title, fontsize=12)
            ax.axis('off')

            # Visualize tip position vs trocar entry — measurement
            p_entry = data.site_xpos[target_entry_id]
            p_tip = data.site_xpos[tip_id]
            dist_mm = np.linalg.norm(p_tip - p_entry) * 1000
            ax.text(0.02, 0.98, f"tip_dist={dist_mm:.1f}mm",
                    transform=ax.transAxes, va='top', fontsize=10,
                    color='yellow', bbox=dict(facecolor='black', alpha=0.5))

    fig.suptitle(f"y=-25 Self-Occlusion Sanity Check (tool_camera, tip {STUCK_DIST_MM:.0f}mm from trocar entry, angle=0)",
                 fontsize=13, fontweight='bold', y=1.0)
    fig.text(0.5, 0.01,
             "Row 1: y=-25 (fail region)  |  Row 2: y=0 (mid)  |  Row 3: y=+25 (best region)",
             ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    out_path = OUT_DIR / "yneg_occlusion_check.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
