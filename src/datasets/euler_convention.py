"""Euler convention conversion between MuJoCo (sim) and Mecademic Meca500 (real).

- MuJoCo `mju_mat2euler`: extrinsic XYZ  (scipy 'xyz', lowercase)
- Mecademic GetPose:     intrinsic XYZ  (scipy 'XYZ', uppercase)

Same physical rotation R yields different Euler triples in the two conventions
(notably opposite signs on rx/rz when ry != 0). To unify training data without
touching either control loop, sim ee_pose / action orientation is converted to
the Mecademic convention at dataset load time.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def mujoco_to_mecademic_euler(euler_xyz):
    """MuJoCo extrinsic-XYZ Euler (rad) → Mecademic intrinsic-XYZ Euler (rad)."""
    e = np.asarray(euler_xyz, dtype=np.float64)
    flat = e.reshape(-1, 3)
    out = R.from_euler("xyz", flat, degrees=False).as_euler("XYZ", degrees=False)
    return out.reshape(e.shape).astype(np.float32)


def mecademic_to_mujoco_euler(euler_XYZ):
    """Mecademic intrinsic-XYZ Euler (rad) → MuJoCo extrinsic-XYZ Euler (rad)."""
    e = np.asarray(euler_XYZ, dtype=np.float64)
    flat = e.reshape(-1, 3)
    out = R.from_euler("XYZ", flat, degrees=False).as_euler("xyz", degrees=False)
    return out.reshape(e.shape).astype(np.float32)


def convert_ee_pose_to_mecademic(ee_pose, src="mujoco"):
    """Convert ee_pose[:, 3:6] orientation to Mecademic convention.

    Args:
        ee_pose: (N, D) — D>=6, [0:3]=pos(mm), [3:6]=Euler(rad), rest untouched.
        src:     'mujoco' or 'mecademic'. 'mecademic' returns input as-is.
    """
    if src == "mecademic":
        return ee_pose
    out = ee_pose.copy()
    out[:, 3:6] = mujoco_to_mecademic_euler(out[:, 3:6])
    return out


def recompute_delta_orientation(action, ee_pose_converted):
    """Recompute action[:, 3:6] = wrap(ee[t+1, 3:6] - ee[t, 3:6]) in target convention.

    Last-frame delta kept as 0 (sim recorder pads same way).
    """
    out = action.copy().astype(np.float32)
    eul = ee_pose_converted[:, 3:6].astype(np.float32)
    delta = np.zeros_like(eul)
    if len(eul) >= 2:
        delta[:-1] = eul[1:] - eul[:-1]
        delta = np.arctan2(np.sin(delta), np.cos(delta))  # wrap to [-pi, pi]
    out[:, 3:6] = delta
    return out


def infer_convention(path):
    """Auto-detect from path: 'real' substring → 'mecademic', else 'mujoco'."""
    p = str(path).lower()
    return "mecademic" if ("real" in p or "collected_data_real" in p) else "mujoco"
