"""
Compute action normalization statistics (min/max, p99, p95) from HDF5 episode files.

Usage:
    python dataset/compute_action_stats.py --data_dir /data/public/NAS/VLANeXt/dataset/fine_align/collected_data_merged
"""

import argparse
import glob
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .h5 episode files")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(f"{args.data_dir}/*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    all_actions = []
    total_steps = 0

    for f_path in h5_files:
        try:
            with h5py.File(f_path, "r") as f:
                actions = f["action"][:].astype(np.float32)
                total_steps += actions.shape[0]
                all_actions.append(actions)
        except Exception as e:
            print(f"[Warn] Skipping {f_path}: {e}")

    all_actions = np.concatenate(all_actions, axis=0)

    global_min = all_actions.min(axis=0)
    global_max = all_actions.max(axis=0)
    p1 = np.percentile(all_actions, 1, axis=0)
    p5 = np.percentile(all_actions, 5, axis=0)
    p95 = np.percentile(all_actions, 95, axis=0)
    p99 = np.percentile(all_actions, 99, axis=0)

    labels = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
    ndim = all_actions.shape[1]

    print(f"Episodes: {len(h5_files)}")
    print(f"Total steps: {total_steps:,}")
    print(f"Action dim: {ndim}")
    print()

    header = f"{'dim':>5} {'min':>11} {'p1':>11} {'p5':>11} {'p50':>11} {'p95':>11} {'p99':>11} {'max':>11}"
    print(header)
    for d in range(ndim):
        label = labels[d] if d < len(labels) else f"d{d}"
        col = all_actions[:, d]
        print(f"{label:>5} {global_min[d]:11.6f} {p1[d]:11.6f} {p5[d]:11.6f} {np.median(col):11.6f} {p95[d]:11.6f} {p99[d]:11.6f} {global_max[d]:11.6f}")

    print()
    print("=" * 60)
    print("100% (min/max):")
    print(f"  action_min = {global_min.tolist()}")
    print(f"  action_max = {global_max.tolist()}")
    print()
    print("99th percentile:")
    print(f"  action_min = {p1.tolist()}")
    print(f"  action_max = {p99.tolist()}")
    print()
    print("95th percentile:")
    print(f"  action_min = {p5.tolist()}")
    print(f"  action_max = {p95.tolist()}")


if __name__ == "__main__":
    main()
