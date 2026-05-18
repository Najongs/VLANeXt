"""Flange ↔ Needle-tip frame conversion (TCP shift).

The needle tool is rigidly mounted to robot link6 with a pure +Z translation
of 177.5mm and zero rotation offset (see Sim/meca_add.xml:87 — site
"needle_tip" pos="0 0 0.1775" inside body 260107_Total_assembly which has no
pos/quat under 6_Link). Therefore tip frame and flange frame share the same
orientation; only the position differs by R_flange @ TIP_OFFSET.

Conventions used elsewhere in this repo (matching get_ee_pose_6d_scaled in
Sim/Save_dataset_align_only.py):
- Position in WORLD frame, mm.
- Orientation as XYZ-extrinsic / ZYX-intrinsic Euler angles, rad
  (decomposed by arctan2 from xmat).
- ee_pose 6-vector layout: [x, y, z, roll, pitch, yaw] (mm + rad).

Action delta is element-wise pose difference (current - last) with euler
wrapping to [-pi, pi]. Because rotation is identical between flange and tip,
swapping the recorded ee_pose to tip frame automatically yields tip-frame
action deltas without changing the diff code.
"""
from __future__ import annotations

import numpy as np

# Source: Sim/meca_add.xml:87 — pos="0 0 0.1775" in 6_Link body local frame.
TIP_OFFSET_M = np.array([0.0, 0.0, 0.1775], dtype=np.float64)
TIP_OFFSET_MM = TIP_OFFSET_M * 1000.0  # [0, 0, 177.5]


def euler_to_mat(rpy: np.ndarray) -> np.ndarray:
    """Convert XYZ-extrinsic / ZYX-intrinsic euler (r, p, y) [rad] to 3x3 rotation.

    Inverse of the decomposition in Save_dataset_align_only.py:351-359.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_euler(mat: np.ndarray) -> np.ndarray:
    """Inverse of euler_to_mat — same convention as Save_dataset_align_only.py:351-359."""
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy > 1e-6:
        r = np.arctan2(mat[2, 1], mat[2, 2])
        p = np.arctan2(-mat[2, 0], sy)
        y = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        r = np.arctan2(-mat[1, 2], mat[1, 1])
        p = np.arctan2(-mat[2, 0], sy)
        y = 0.0
    return np.array([r, p, y], dtype=np.float64)


def flange_to_tip_pose(p_flange: np.ndarray, R_flange: np.ndarray, *, mm: bool = True):
    """Shift translation by R @ TIP_OFFSET. Rotation unchanged.

    p_flange in mm by default (set mm=False for meters).
    """
    offset = TIP_OFFSET_MM if mm else TIP_OFFSET_M
    p_tip = np.asarray(p_flange, dtype=np.float64) + R_flange @ offset
    return p_tip, np.asarray(R_flange, dtype=np.float64).copy()


def tip_to_flange_pose(p_tip: np.ndarray, R_tip: np.ndarray, *, mm: bool = True):
    """Inverse of flange_to_tip_pose."""
    offset = TIP_OFFSET_MM if mm else TIP_OFFSET_M
    p_flange = np.asarray(p_tip, dtype=np.float64) - R_tip @ offset
    return p_flange, np.asarray(R_tip, dtype=np.float64).copy()


def flange_to_tip_euler6(pose6_flange: np.ndarray, *, mm: bool = True) -> np.ndarray:
    """6-vector [x,y,z,r,p,y] flange → tip. Rotation columns unchanged."""
    pose6 = np.asarray(pose6_flange, dtype=np.float64)
    R = euler_to_mat(pose6[3:6])
    p_tip, _ = flange_to_tip_pose(pose6[:3], R, mm=mm)
    return np.concatenate([p_tip, pose6[3:6]]).astype(pose6_flange.dtype if hasattr(pose6_flange, "dtype") else np.float64)


def tip_to_flange_euler6(pose6_tip: np.ndarray, *, mm: bool = True) -> np.ndarray:
    """6-vector [x,y,z,r,p,y] tip → flange. Rotation columns unchanged."""
    pose6 = np.asarray(pose6_tip, dtype=np.float64)
    R = euler_to_mat(pose6[3:6])
    p_flange, _ = tip_to_flange_pose(pose6[:3], R, mm=mm)
    return np.concatenate([p_flange, pose6[3:6]]).astype(pose6_tip.dtype if hasattr(pose6_tip, "dtype") else np.float64)


# ---------------------------------------------------------------------------
# Self-test (round-trip + math sanity). Run: python -m src.utils.tip_frame
# ---------------------------------------------------------------------------
def _self_test(n: int = 1000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    max_err_pos = 0.0
    max_err_rot = 0.0
    max_err_euler = 0.0
    for _ in range(n):
        p = rng.uniform(-1000, 1000, size=3)         # mm
        rpy = rng.uniform(-np.pi, np.pi, size=3)
        # Avoid gimbal lock for this test; keep pitch within (-pi/2 + eps, pi/2 - eps)
        rpy[1] = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
        R = euler_to_mat(rpy)

        # round-trip pose
        p_tip, R_tip = flange_to_tip_pose(p, R)
        p_back, R_back = tip_to_flange_pose(p_tip, R_tip)
        max_err_pos = max(max_err_pos, np.max(np.abs(p_back - p)))
        max_err_rot = max(max_err_rot, np.max(np.abs(R_back - R)))

        # round-trip euler6
        pose6 = np.concatenate([p, rpy])
        pose6_tip = flange_to_tip_euler6(pose6)
        pose6_back = tip_to_flange_euler6(pose6_tip)
        max_err_euler = max(max_err_euler, np.max(np.abs(pose6_back - pose6)))

        # Math sanity: position offset must equal R @ [0,0,177.5]
        expected = R @ TIP_OFFSET_MM
        actual = p_tip - p
        assert np.allclose(expected, actual), "translation offset mismatch"

    print(f"[tip_frame self-test] n={n}")
    print(f"  pose round-trip pos max err: {max_err_pos:.3e} mm")
    print(f"  pose round-trip rot max err: {max_err_rot:.3e}")
    print(f"  euler6 round-trip max err:   {max_err_euler:.3e}")
    print(f"  TIP_OFFSET_MM = {TIP_OFFSET_MM}")
    assert max_err_pos < 1e-9
    assert max_err_rot < 1e-12
    assert max_err_euler < 1e-9
    print("OK")


if __name__ == "__main__":
    _self_test()
