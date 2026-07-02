"""
convert_to_lerobot.py

Convert VLANeXt fine-align HDF5 episodes to a LeRobotDataset.

Output features (baseline — no sensor flags):
  observation.images.tool_camera  : (480, 640, 3) uint8 RGB native resolution (JPEG round-trip, no resize by default)
  observation.state               : (6,)  float32  [x, y, z, roll, pitch, yaw]  Mecademic intrinsic XYZ, mm + rad
  action                          : (6,)  float32  delta in same convention, RAW (lerobot computes stats)

Usage:
    python -m dataset.convert_to_lerobot \
        --src /home/najo/NAS/VLANeXt/dataset/approach/approach_00 \
        --repo-id vlanext/sim_align_baseline \
        --root /home/najo/NAS/VLANeXt/dataset/lerobot \
        --fps 15

Notes:
- Sensor binary flags (sensor_close / hole_through) are intentionally dropped for baseline.
  When introducing sensor signal later, add `observation.sensor` feature here.
- Convention: episodes under paths matching infer_convention rules from src.datasets.euler_convention
  may be MuJoCo extrinsic XYZ — those are converted to Mecademic intrinsic XYZ here, matching
  the existing VLANeXt training pipeline (sim_act_align.py:168-170).
- Action is stored RAW (NOT [-1,1] normalized), since lerobot computes per-feature stats during dataset.save_episode().
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets.euler_convention import (
    convert_ee_pose_to_mecademic,
    recompute_delta_orientation,
    infer_convention,
)


IMG_H = 480
IMG_W = 640


def _decode_jpeg(jpeg_data) -> np.ndarray:
    if isinstance(jpeg_data, np.ndarray):
        buf = jpeg_data.flatten().astype(np.uint8)
    elif isinstance(jpeg_data, (bytes, bytearray)):
        buf = np.frombuffer(jpeg_data, dtype=np.uint8)
    else:
        buf = np.array(jpeg_data, dtype=np.uint8).flatten()
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"JPEG decode failed (size={buf.size})")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _preprocess_image(img_rgb: np.ndarray, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    # JPEG round-trip emulates camera pipeline noise so train ↔ eval distribution match.
    # No resize when source already matches target (default: keep native 480x640).
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    bgr2 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    if rgb2.shape[0] != h or rgb2.shape[1] != w:
        rgb2 = cv2.resize(rgb2, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return rgb2


def _load_episode(h5_path: str, convention: str, img_h: int, img_w: int):
    with h5py.File(h5_path, "r") as f:
        traj_len = f["action"].shape[0]
        actions_raw = f["action"][:].astype(np.float32)              # (N, 7)
        ee_pose_raw = f["observations"]["ee_pose"][:].astype(np.float32)  # (N, 7)

        if convention == "mujoco":
            ee_pose_raw = convert_ee_pose_to_mecademic(ee_pose_raw, src="mujoco")
            actions_raw = recompute_delta_orientation(actions_raw, ee_pose_raw)

        # Drop gripper (last dim) — 6-DoF only.
        actions = actions_raw[:, :6].astype(np.float32)
        ee_pose = ee_pose_raw[:, :6].astype(np.float32)

        img_grp = f["observations"]["images"]
        if "tool_camera" not in img_grp:
            raise KeyError(f"{h5_path}: tool_camera missing in observations/images")
        images = np.stack(
            [_preprocess_image(_decode_jpeg(img_grp["tool_camera"][i]), img_h, img_w) for i in range(traj_len)],
            axis=0,
        )  # (N, H, W, 3) uint8

    return images, ee_pose, actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, action="append", required=True,
                   help="HDF5 root (recursively scanned). Can be passed multiple times.")
    p.add_argument("--repo-id", type=str, required=True,
                   help="LeRobotDataset repo_id (e.g., vlanext/sim_align_baseline)")
    p.add_argument("--root", type=str, default=None,
                   help="Local root for the LeRobotDataset (default: ~/.cache/huggingface/lerobot)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--task", type=str,
                   default="Align the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap total episodes (debug).")
    p.add_argument("--convention", type=str, default=None,
                   choices=[None, "mujoco", "mecademic"],
                   help="Override convention (default: auto via infer_convention).")
    p.add_argument("--img-h", type=int, default=IMG_H, help="Output image height (default: 480, native).")
    p.add_argument("--img-w", type=int, default=IMG_W, help="Output image width (default: 640, native).")
    p.add_argument("--vcodec", type=str, default="h264",
                   choices=["libsvtav1", "h264", "hevc", "auto"],
                   help="Video codec. h264 is ~3x faster than libsvtav1 (default).")
    p.add_argument("--image-writer-threads", type=int, default=4,
                   help="Async image writer threads.")
    p.add_argument("--image-writer-processes", type=int, default=0,
                   help="Async image writer processes (0 = threads only).")
    p.add_argument("--batch-encoding-size", type=int, default=1,
                   help="Episodes per batch encoding. >1 hits a NoneType bug in lerobot 0.5.2.")
    args = p.parse_args()

    # Lerobot v0.5+ layout (src/lerobot/datasets/...).
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e_new:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # < 0.2
        except ImportError as e_old:
            print(f"ERROR importing LeRobotDataset:\n  v0.5 path: {e_new}\n  legacy:    {e_old}\n"
                  "Try: pip install -e '/home/najo/NAS/VLANeXt/lerobot[dataset,training]'",
                  file=sys.stderr)
            sys.exit(1)

    # Collect episodes.
    episode_paths = []
    path_to_conv = {}
    for src in args.src:
        eps = sorted(glob.glob(os.path.join(src, "**", "*.h5"), recursive=True))
        conv = args.convention or infer_convention(src)
        for e in eps:
            path_to_conv[e] = conv
        episode_paths.extend(eps)
    episode_paths = sorted(set(episode_paths))
    if not episode_paths:
        print("ERROR: no .h5 files under any --src.", file=sys.stderr)
        sys.exit(1)
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    print(f"Found {len(episode_paths)} episode(s).")

    features = {
        "observation.images.tool_camera": {
            "dtype": "video",
            "shape": (args.img_h, args.img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x_mm", "y_mm", "z_mm", "roll_rad", "pitch_rad", "yaw_rad"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dx_mm", "dy_mm", "dz_mm", "droll_rad", "dpitch_rad", "dyaw_rad"],
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        batch_encoding_size=args.batch_encoding_size,
    )

    for i, ep_path in enumerate(episode_paths):
        conv = path_to_conv[ep_path]
        try:
            images, ee_pose, actions = _load_episode(ep_path, conv, args.img_h, args.img_w)
        except Exception as e:
            print(f"  [skip] {ep_path}: {e}")
            continue
        n = images.shape[0]
        for t in range(n):
            # lerobot v0.5: task is part of the frame dict.
            ds.add_frame({
                "task": args.task,
                "observation.images.tool_camera": images[t],
                "observation.state": ee_pose[t],
                "action": actions[t],
            })
        ds.save_episode()
        print(f"[{i+1}/{len(episode_paths)}] {Path(ep_path).name}: {n} steps  (conv={conv})")

    ds.finalize()
    print(f"Done. LeRobotDataset saved to {ds.root}")


if __name__ == "__main__":
    main()
