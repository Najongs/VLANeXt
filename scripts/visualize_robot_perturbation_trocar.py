"""Robot perturbation visualization — Trocar LOCAL frame (tool + side cameras).

Frame:
  z_local = -axis_dir (trocar 안 = Z 음수)
  x_local, y_local = trocar 입구 평면 orthogonal basis

Outputs (4 PNG):
  vqa_samples/robot_perturb_X_views_trocar.png   - trocar X-axis
  vqa_samples/robot_perturb_Y_views_trocar.png   - trocar Y-axis
  vqa_samples/robot_perturb_Z_views_trocar.png   - trocar Z-axis (Z- = 안)
  vqa_samples/robot_perturb_angle_views_trocar.png - pitch (local X-rot) + yaw (local Y-rot)
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
    make_ik, smooth_step,
    ALIGN_SPEED, ALIGN_THRESHOLD_M, ALIGN_HOLD_STEPS, RETREAT_MM,
    IMG_W, IMG_H, MODEL_PATH,
)

OUT_DIR = Path("/home/najo/NAS/VLANeXt/vqa_samples")


def make_trocar_basis(axis_dir):
    z_local = -axis_dir / np.linalg.norm(axis_dir)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_local[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_local = np.cross(ref, z_local)
    x_local /= np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)
    return x_local, y_local, z_local


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    n_motors = model.nu
    dof = model.nv
    run_ik = make_ik(model, data, tip_id, back_id, n_motors, dof)

    # Pre-align
    print("Pre-aligning...")
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = np.zeros(6)
    mujoco.mj_forward(model, data)
    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])
    axis_dir = (p_depth - p_entry) / np.linalg.norm(p_depth - p_entry)
    goal_tip = p_entry - axis_dir * RETREAT_MM/1000
    goal_back = p_entry - axis_dir * (RETREAT_MM/1000 + needle_len)
    start_tip = data.site_xpos[tip_id].copy()
    start_back = data.site_xpos[back_id].copy()
    dur = np.linalg.norm(goal_tip - start_tip) / ALIGN_SPEED
    t0 = data.time; timer = 0
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
                break
        if data.time - t0 > 50: return
    aligned_qpos = data.qpos[:n_motors].copy()
    aligned_qvel = data.qvel[:n_motors].copy()

    x_loc, y_loc, z_loc = make_trocar_basis(axis_dir)
    print(f"axis_dir world: {axis_dir}")
    print(f"trocar local x: {x_loc}")
    print(f"trocar local y: {y_loc}")
    print(f"trocar local z: {z_loc}  (Z- = trocar 안)")

    def move_local(x_mm, y_mm, z_mm, angle_deg, rot_axis_w):
        # No-op: just reset and return (avoids IK degeneracy at zero perturb)
        if abs(x_mm) < 1e-6 and abs(y_mm) < 1e-6 and abs(z_mm) < 1e-6 and abs(angle_deg) < 1e-6:
            mujoco.mj_resetData(model, data)
            data.qpos[:n_motors] = aligned_qpos
            data.qvel[:n_motors] = aligned_qvel
            mujoco.mj_forward(model, data)
            return True
        offset_world = x_loc * (x_mm/1000) + y_loc * (y_mm/1000) + z_loc * (z_mm/1000)
        perturbed_tip = goal_tip + offset_world
        rot_axis_w = rot_axis_w / (np.linalg.norm(rot_axis_w) + 1e-10)
        rot_angle_rad = np.deg2rad(angle_deg)
        rot_mat = np.eye(3)
        if abs(rot_angle_rad) > 1e-6:
            K = np.array([
                [0, -rot_axis_w[2], rot_axis_w[1]],
                [rot_axis_w[2], 0, -rot_axis_w[0]],
                [-rot_axis_w[1], rot_axis_w[0], 0],
            ])
            rot_mat = np.eye(3) + np.sin(rot_angle_rad) * K + (1 - np.cos(rot_angle_rad)) * (K @ K)
        perturbed_back = perturbed_tip + rot_mat @ (goal_back - goal_tip)

        mujoco.mj_resetData(model, data)
        data.qpos[:n_motors] = aligned_qpos
        data.qvel[:n_motors] = aligned_qvel
        mujoco.mj_forward(model, data)
        move_speed = 0.05
        move_dist = np.linalg.norm(perturbed_tip - goal_tip) + abs(rot_angle_rad) * 0.05
        mdur = max(move_dist / move_speed, 0.1)
        t_start = data.time
        for _ in range(5000):
            t = (data.time - t_start) / mdur
            a = smooth_step(min(t, 1.0))
            run_ik((1-a)*goal_tip + a*perturbed_tip, (1-a)*goal_back + a*perturbed_back)
            mujoco.mj_step(model, data)
            if t >= 1.0 and np.linalg.norm(data.site_xpos[tip_id] - perturbed_tip) < 0.001:
                for _ in range(100):
                    run_ik(perturbed_tip, perturbed_back)
                    mujoco.mj_step(model, data)
                return True
        return False

    def render(cam):
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    def make_2row_png(title, bins, label_fn, perturb_fn, out_path):
        n = len(bins)
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))
        for col_i, b in enumerate(bins):
            ok = perturb_fn(b)
            tool = render('tool_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            side = render('side_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            axes[0, col_i].imshow(tool)
            axes[1, col_i].imshow(side)
            axes[0, col_i].set_title(label_fn(b), fontsize=13)
            axes[0, col_i].axis('off'); axes[1, col_i].axis('off')
        axes[0, 0].text(-0.08, 0.5, 'tool_camera', rotation=90, transform=axes[0, 0].transAxes,
                        va='center', fontsize=11, fontweight='bold')
        axes[1, 0].text(-0.08, 0.5, 'side_camera', rotation=90, transform=axes[1, 0].transAxes,
                        va='center', fontsize=11, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    def make_angle_png(title, bins, out_path):
        """4 rows × n: pitch (local X-rot) tool+side, yaw (local Y-rot) tool+side."""
        n = len(bins)
        fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 17))
        for row_i, (rot_axis_w, axis_label) in enumerate([
            (x_loc, 'PITCH (local X-rot)'),
            (y_loc, 'YAW (local Y-rot)'),
        ]):
            for col_i, b in enumerate(bins):
                ok = move_local(0, 0, 0, b, rot_axis_w)
                tool = render('tool_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                side = render('side_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                ax_t = axes[row_i*2, col_i]; ax_s = axes[row_i*2 + 1, col_i]
                ax_t.imshow(tool); ax_s.imshow(side)
                ax_t.set_title(f"{b:+}°", fontsize=13)
                ax_t.axis('off'); ax_s.axis('off')
            axes[row_i*2, 0].text(-0.10, 0.5, f'{axis_label}\ntool', rotation=90,
                                  transform=axes[row_i*2, 0].transAxes, va='center', fontsize=11, fontweight='bold')
            axes[row_i*2 + 1, 0].text(-0.10, 0.5, f'{axis_label}\nside', rotation=90,
                                       transform=axes[row_i*2 + 1, 0].transAxes, va='center', fontsize=11, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    bins_pos = [-10, -7, -4, 0, 4, 7, 10]
    bins_angle = [-20, -15, -10, 0, 10, 15, 20]

    print("\n[1/4] X perturbation (trocar local)...")
    make_2row_png("Robot X perturbation — TROCAR LOCAL frame (Y=Z=angle=0)",
                  bins_pos, lambda b: f"x_local={b:+}mm",
                  lambda b: move_local(b, 0, 0, 0, x_loc),
                  OUT_DIR / "robot_perturb_X_views_trocar.png")
    print("\n[2/4] Y perturbation (trocar local)...")
    make_2row_png("Robot Y perturbation — TROCAR LOCAL frame (X=Z=angle=0)",
                  bins_pos, lambda b: f"y_local={b:+}mm",
                  lambda b: move_local(0, b, 0, 0, x_loc),
                  OUT_DIR / "robot_perturb_Y_views_trocar.png")
    print("\n[3/4] Z perturbation (trocar local, Z- = trocar 안)...")
    make_2row_png("Robot Z perturbation — TROCAR LOCAL frame (Z- = INSIDE trocar)",
                  bins_pos, lambda b: f"z_local={b:+}mm",
                  lambda b: move_local(0, 0, b, 0, x_loc),
                  OUT_DIR / "robot_perturb_Z_views_trocar.png")
    print("\n[4/4] Angle perturbation (trocar pitch + yaw)...")
    make_angle_png(
        "Robot Angle perturbation — TROCAR LOCAL (PITCH = X-rot, YAW = Y-rot)",
        bins_angle,
        OUT_DIR / "robot_perturb_angle_views_trocar.png",
    )


if __name__ == "__main__":
    main()
