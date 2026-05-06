"""Convert an existing flange-frame HDF5 dataset to needle-tip frame.

Reads each *.h5 under SRC_DIR, rewrites:
  - observations/ee_pose[:, :3]   (mm)  →  flange + R @ TIP_OFFSET_MM
  - observations/ee_pose[:, 3:6]  (rad) →  unchanged (flange and tip share orientation)
  - observations/ee_pose[:, 6]          unchanged (gripper proprio)
  - action[:, :6]                       recomputed as element-wise diff of new ee_pose,
                                        with the same euler-wrap + ACTION_CLIP_MM clip
                                        as Save_dataset_align_only.py:720-725
  - action[:, 6]                        unchanged (gripper)

All other keys (qpos, sensor_dist, images, needle_tip_pos, trocar_entry_pos,
keypoints_*, phase, metadata, timestamp) are copied verbatim.

The converted dataset has identical schema, so existing dataset loaders work
unchanged. After conversion, action_max_sim_align must be recomputed and
config/sim_train_align_config.yaml's pos_denorm_mm synced.

Usage:
    python -m Sim.convert_dataset_to_tip_frame \
        --src dataset/fine_align/10mm_fine_align_00/collected_data_merged \
        --dst dataset/fine_align/10mm_fine_align_00_tip/collected_data_merged
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.tip_frame import TIP_OFFSET_MM, euler_to_mat  # noqa: E402

ACTION_CLIP_MM = 1.0  # must match Save_dataset_align_only.py:98


def convert_pose_array_flange_to_tip(ee_pose_flange: np.ndarray) -> np.ndarray:
    """ee_pose: (N, 7) — last col gripper preserved. Cols [:3] mm, [3:6] rad."""
    out = ee_pose_flange.copy()
    for i in range(ee_pose_flange.shape[0]):
        rpy = ee_pose_flange[i, 3:6]
        R = euler_to_mat(rpy)
        out[i, :3] = ee_pose_flange[i, :3] + (R @ TIP_OFFSET_MM).astype(out.dtype)
        # rotation [3:6] and gripper [6] unchanged
    return out


def recompute_action(ee_pose_tip: np.ndarray, action_flange: np.ndarray) -> np.ndarray:
    """Recompute element-wise action delta in tip frame.

    Frame 0: invert original flange action to recover the pre-recording pose,
    convert that recovered pose to tip frame, take diff. Approximate when
    action[0] was clipped, but only affects 1 frame per episode.
    """
    n = ee_pose_tip.shape[0]
    # Recover pre-recording flange pose (frame "-1") from original action[0]:
    # last_ee_flange ≈ ee_flange[0] - action[0]   (ignoring wrap, since deltas are small)
    # Then convert to tip frame.
    # ee_pose_tip[0] - prev_tip_pose = action_tip[0]
    # We need prev_flange_pose ≈ ee_pose_flange[0] - action_flange[0,:6]
    # ee_pose_flange[0] = ee_pose_tip[0] - R(rpy[0]) @ TIP_OFFSET (since rpy unchanged)
    rpy0 = ee_pose_tip[0, 3:6]
    R0 = euler_to_mat(rpy0)
    ee_flange_0 = ee_pose_tip[0, :6].copy()
    ee_flange_0[:3] = ee_pose_tip[0, :3] - R0 @ TIP_OFFSET_MM

    prev_flange = ee_flange_0.copy()
    prev_flange[:6] = ee_flange_0[:6] - action_flange[0, :6]
    # Convert prev_flange to prev_tip
    R_prev = euler_to_mat(prev_flange[3:6])
    prev_tip_pos = prev_flange[:3] + R_prev @ TIP_OFFSET_MM
    prev_tip = np.concatenate([prev_tip_pos, prev_flange[3:6]])

    new_action = np.zeros_like(action_flange)
    new_action[:, 6] = action_flange[:, 6]  # gripper unchanged

    for i in range(n):
        cur_tip = ee_pose_tip[i, :6]
        prev = prev_tip if i == 0 else ee_pose_tip[i - 1, :6]
        d = cur_tip - prev
        # euler wrap (matches Save_dataset_align_only.py:721)
        d[3:6] = (d[3:6] + np.pi) % (2 * np.pi) - np.pi
        # ACTION_CLIP_MM clip on position magnitude (matches lines 723-725)
        pos_mag = float(np.linalg.norm(d[:3]))
        if pos_mag > ACTION_CLIP_MM:
            d[:3] *= ACTION_CLIP_MM / pos_mag
        new_action[i, :6] = d
    return new_action


def convert_one(src_path: str, dst_path: str) -> tuple[str, bool, str]:
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            # Copy entire structure first (preserves images, metadata, etc.)
            def _copy(name, obj):
                if isinstance(obj, h5py.Group):
                    if name not in dst:
                        dst.create_group(name)
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v
                else:
                    # dataset
                    parent = dst
                    if "/" in name:
                        gname = name.rsplit("/", 1)[0]
                        if gname not in dst:
                            dst.create_group(gname)
                    dst.create_dataset(name, data=obj[()], compression=obj.compression,
                                       compression_opts=obj.compression_opts)
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v
            src.visititems(_copy)
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            # Now overwrite ee_pose and action
            ee_flange = src["observations/ee_pose"][()]
            action_flange = src["action"][()]

            ee_tip = convert_pose_array_flange_to_tip(ee_flange)
            action_tip = recompute_action(ee_tip, action_flange)

            del dst["observations/ee_pose"]
            del dst["action"]
            dst["observations"].create_dataset(
                "ee_pose", data=ee_tip.astype(np.float32), compression="gzip"
            )
            dst.create_dataset(
                "action", data=action_tip.astype(np.float32), compression="gzip"
            )
            # Annotate frame for traceability
            dst.attrs["ee_pose_frame"] = "needle_tip"
            dst.attrs["action_frame"] = "needle_tip"
            dst.attrs["tip_offset_mm"] = TIP_OFFSET_MM.astype(np.float32)
        return src_path, True, ""
    except Exception as e:
        # Clean up partial file
        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except OSError:
                pass
        return src_path, False, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source dir of *.h5 (flange frame)")
    ap.add_argument("--dst", required=True, help="Destination dir for converted *.h5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Convert only first N (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.src, "*.h5")))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No .h5 files in {args.src}")
        return

    if os.path.exists(args.dst) and args.overwrite:
        shutil.rmtree(args.dst)
    os.makedirs(args.dst, exist_ok=True)

    print(f"Converting {len(files)} files: {args.src} → {args.dst}")
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, f, os.path.join(args.dst, os.path.basename(f))): f
            for f in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), unit="ep"):
            src_path, ok, err = fut.result()
            if not ok:
                fails.append((src_path, err))

    print(f"\nDone. ok={len(files) - len(fails)}, fail={len(fails)}")
    for p, e in fails[:5]:
        print(f"  FAIL {p}: {e}")


if __name__ == "__main__":
    main()
