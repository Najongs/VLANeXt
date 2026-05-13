"""Sensor signature characterization.

Place needle aligned with trocar (no perturbation), then apply known
tip-frame deltas and log sensor_dist to confirm:
  through-hole  → max reading (or -1)
  misaligned    → small reading (mm to phantom surface)

Usage:
  MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  python -u -m scripts.probe_sensor
"""
import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sim_eval_align_only import AlignSimEnv, SIM_MODEL_PATH


def set_aligned(env):
    """Force needle to pre-aligned state (no perturbation). Mirrors reset() first half."""
    # Force phantom angle=0 once, then suppress re-randomization in _ensure_aligned_state.
    env._set_fixed_phantom(env.phantom_pos, pz=0.0, angle_deg=0.0)
    saved_pos, saved_rand = env.phantom_pos, env.randomize_phantom
    env.phantom_pos, env.randomize_phantom = None, False
    env._aligned_qpos = None  # invalidate to force fresh pre-align
    try:
        env._ensure_aligned_state()
    finally:
        env.phantom_pos, env.randomize_phantom = saved_pos, saved_rand
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:env.n_motors] = env._aligned_qpos
    env.data.qvel[:env.n_motors] = env._aligned_qvel
    env.data.ctrl[:env.n_motors] = env._aligned_qpos[:env.n_motors]
    mujoco.mj_forward(env.model, env.data)


def run_at(env, dx=0.0, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0, sim_steps=200):
    set_aligned(env)
    env.apply_delta_ee(np.array([dx, dy, dz, drx, dry, drz]), n_sim_steps=sim_steps)
    m = env.get_spatial_metrics()
    s = env.get_sensor_dist()
    return m, s


def main():
    env = AlignSimEnv(
        os.path.abspath(SIM_MODEL_PATH),
        randomize_phantom=False,
        use_sensor_success=False,
        phantom_pos=(0.0, 0.0),
        retreat_mm=5.0,
    )
    set_aligned(env)
    m = env.get_spatial_metrics()
    s = env.get_sensor_dist()
    print("=" * 70)
    print("Sensor signature: needle pre-aligned to trocar axis, retreat 5mm")
    print("=" * 70)
    print(f"\n[Baseline aligned] dist={m['dist_mm']:.2f}mm lateral={m['lateral_mm']:.2f}mm "
          f"angle={m['angle_deg']:.2f}deg sensor={s:.2f}mm  (− = ray miss = through)\n")

    print("--- Lateral sweep tip-X ---")
    print(f"{'dx_mm':>8}  {'lateral_mm':>11}  {'sensor_mm':>10}  {'dist_mm':>9}")
    for dx in [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        m, s = run_at(env, dx=dx)
        print(f"{dx:>8.2f}  {m['lateral_mm']:>11.2f}  {s:>10.2f}  {m['dist_mm']:>9.2f}")

    print("\n--- Lateral sweep tip-Y ---")
    print(f"{'dy_mm':>8}  {'lateral_mm':>11}  {'sensor_mm':>10}  {'dist_mm':>9}")
    for dy in [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        m, s = run_at(env, dy=dy)
        print(f"{dy:>8.2f}  {m['lateral_mm']:>11.2f}  {s:>10.2f}  {m['dist_mm']:>9.2f}")

    print("\n--- Forward sweep tip-Z (+Z = into trocar) ---")
    print(f"{'dz_mm':>8}  {'dist_mm':>9}  {'sensor_mm':>10}")
    for dz in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0]:
        m, s = run_at(env, dz=dz)
        print(f"{dz:>8.2f}  {m['dist_mm']:>9.2f}  {s:>10.2f}")

    print("\n--- Angle sweep rx ---")
    print(f"{'rx_rad':>8}  {'angle_deg':>10}  {'sensor_mm':>10}")
    for rx in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
        m, s = run_at(env, drx=rx)
        print(f"{rx:>8.3f}  {m['angle_deg']:>10.2f}  {s:>10.2f}")


if __name__ == "__main__":
    main()
