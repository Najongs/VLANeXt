"""Robot perturbation visualization — World frame (tool + side cameras).

Outputs (4 PNG × 2 rows × 7 col):
  vqa_samples/robot_perturb_X_views_clean.png   - world X axis
  vqa_samples/robot_perturb_Y_views_clean.png   - world Y axis
  vqa_samples/robot_perturb_Z_views_clean.png   - world Z axis
  vqa_samples/robot_perturb_angle_views_clean.png - pitch (X-rot) + yaw (Y-rot)

각 PNG row 1 = tool_camera, row 2 = side_camera.
"""
from __future__ import annotations
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "/home/najo/NAS/VLANeXt/Sim/meca_add.xml"
OUT_DIR = Path("/home/najo/NAS/VLANeXt/vqa_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 640, 480
ALIGN_SPEED = 0.15
ALIGN_THRESHOLD_M = 0.002
ALIGN_HOLD_STEPS = 10
RETREAT_MM = 2.0
ACTION_CLIP_MM = 1.0


def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def make_ik(model, data, tip_id, back_id, n_motors, dof):
    ik_speed = 0.5
    def run_ik_step(target_tip_pos, target_back_pos):
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back

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
    return run_ik_step


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

    print("Pre-aligning robot to phantom...")
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = np.zeros(6, dtype=np.float64)
    mujoco.mj_forward(model, data)
    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])
    axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
    goal_tip = p_entry - axis_dir * RETREAT_MM / 1000
    goal_back = p_entry - axis_dir * (RETREAT_MM / 1000 + needle_len)

    start_tip = data.site_xpos[tip_id].copy()
    start_back = data.site_xpos[back_id].copy()
    dur = np.linalg.norm(goal_tip - start_tip) / ALIGN_SPEED
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
                break
        if data.time - t0 > 50:
            return
    aligned_qpos = data.qpos[:n_motors].copy()
    aligned_qvel = data.qvel[:n_motors].copy()
    print(f"  aligned. tip-goal dist: {np.linalg.norm(data.site_xpos[tip_id] - goal_tip)*1000:.3f}mm")

    def move_to_perturb(perturb_xyz_m, perturb_angle_rad, rotation_axis):
        mujoco.mj_resetData(model, data)
        data.qpos[:n_motors] = aligned_qpos
        data.qvel[:n_motors] = aligned_qvel
        mujoco.mj_forward(model, data)

        # Zero perturb → just return aligned state
        if np.linalg.norm(perturb_xyz_m) < 1e-6 and abs(perturb_angle_rad) < 1e-6:
            return True

        perturbed_tip = goal_tip + perturb_xyz_m
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-10)
        rot_mat = np.eye(3)
        if abs(perturb_angle_rad) > 1e-6:
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ])
            rot_mat = np.eye(3) + np.sin(perturb_angle_rad) * K + (1 - np.cos(perturb_angle_rad)) * (K @ K)
        perturbed_back = perturbed_tip + rot_mat @ (goal_back - goal_tip)

        move_speed = 0.05
        move_dist = np.linalg.norm(perturbed_tip - goal_tip) + abs(perturb_angle_rad) * 0.05
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

    def make_2row_png(title, bins, perturb_fn, out_path, label_fn=None):
        n = len(bins)
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))
        for col_i, b in enumerate(bins):
            ok = perturb_fn(b)
            tool = render('tool_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            side = render('side_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            axes[0, col_i].imshow(tool)
            axes[1, col_i].imshow(side)
            axes[0, col_i].set_title(label_fn(b) if label_fn else f"{b}", fontsize=13)
            axes[0, col_i].axis('off')
            axes[1, col_i].axis('off')
        axes[0, 0].text(-0.08, 0.5, 'tool_camera', rotation=90,
                        transform=axes[0, 0].transAxes, va='center', fontsize=11, fontweight='bold')
        axes[1, 0].text(-0.08, 0.5, 'side_camera', rotation=90,
                        transform=axes[1, 0].transAxes, va='center', fontsize=11, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    def make_angle_png(title, bins, rot_axis_pitch, rot_axis_yaw, out_path):
        """4 rows × n_cols: tool+side for pitch (axis X), tool+side for yaw (axis Y)."""
        n = len(bins)
        fig, axes = plt.subplots(4, n, figsize=(4.2 * n, 17))
        for row_i, (rot_axis, axis_label) in enumerate([
            (rot_axis_pitch, 'PITCH (X-rot)'),
            (rot_axis_yaw, 'YAW (Y-rot)'),
        ]):
            for col_i, b in enumerate(bins):
                ok = move_to_perturb(np.zeros(3), np.deg2rad(b), rot_axis)
                tool = render('tool_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                side = render('side_camera') if ok else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                ax_t = axes[row_i*2, col_i]
                ax_s = axes[row_i*2 + 1, col_i]
                ax_t.imshow(tool)
                ax_s.imshow(side)
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

    print("\n[1/4] X perturbation...")
    make_2row_png(
        "Robot X perturbation — WORLD frame (Y=Z=angle=0)",
        bins_pos,
        lambda b: move_to_perturb(np.array([b/1000, 0, 0]), 0.0, np.array([1, 0, 0])),
        OUT_DIR / "robot_perturb_X_views_clean.png",
        label_fn=lambda b: f"x={b:+}mm",
    )
    print("\n[2/4] Y perturbation...")
    make_2row_png(
        "Robot Y perturbation — WORLD frame (X=Z=angle=0)",
        bins_pos,
        lambda b: move_to_perturb(np.array([0, b/1000, 0]), 0.0, np.array([1, 0, 0])),
        OUT_DIR / "robot_perturb_Y_views_clean.png",
        label_fn=lambda b: f"y={b:+}mm",
    )
    print("\n[3/4] Z perturbation...")
    make_2row_png(
        "Robot Z perturbation — WORLD frame (X=Y=angle=0)",
        bins_pos,
        lambda b: move_to_perturb(np.array([0, 0, b/1000]), 0.0, np.array([1, 0, 0])),
        OUT_DIR / "robot_perturb_Z_views_clean.png",
        label_fn=lambda b: f"z={b:+}mm",
    )
    print("\n[4/4] Angle perturbation (pitch X-rot + yaw Y-rot)...")
    make_angle_png(
        "Robot Angle perturbation — WORLD frame (PITCH around X-axis, YAW around Y-axis)",
        bins_angle,
        rot_axis_pitch=np.array([1.0, 0.0, 0.0]),  # X-axis rotation = pitch
        rot_axis_yaw=np.array([0.0, 1.0, 0.0]),    # Y-axis rotation = yaw
        out_path=OUT_DIR / "robot_perturb_angle_views_clean.png",
    )


if __name__ == "__main__":
    main()
