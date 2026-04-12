"""
Compute action/proprio normalization statistics (min/max, p99, p95) from HDF5 episode files.

Usage:
    python dataset/compute_action_stats.py --data_dir /data/public/NAS/VLANeXt/dataset/fine_align/approach_data/collected_data_merged
    python dataset/compute_action_stats.py --data_dir /data/public/NAS/VLANeXt/dataset/fine_align/approach_test/collected_data_merged --proprio   # include proprio stats
"""

import argparse
import glob
import numpy as np
import h5py


def print_stats(data, labels, title):
    ndim = data.shape[1]
    global_min = data.min(axis=0)
    global_max = data.max(axis=0)
    p1 = np.percentile(data, 1, axis=0)
    p5 = np.percentile(data, 5, axis=0)
    p95 = np.percentile(data, 95, axis=0)
    p99 = np.percentile(data, 99, axis=0)

    print(f"\n{'=' * 60}")
    print(f"  {title}  (dim={ndim}, steps={data.shape[0]:,})")
    print(f"{'=' * 60}")

    header = f"{'dim':>8} {'min':>11} {'p1':>11} {'p5':>11} {'p50':>11} {'p95':>11} {'p99':>11} {'max':>11}"
    print(header)
    for d in range(ndim):
        label = labels[d] if d < len(labels) else f"d{d}"
        col = data[:, d]
        print(f"{label:>8} {global_min[d]:11.6f} {p1[d]:11.6f} {p5[d]:11.6f} {np.median(col):11.6f} {p95[d]:11.6f} {p99[d]:11.6f} {global_max[d]:11.6f}")

    print()
    print("100% (min/max):")
    print(f"  min = {global_min.tolist()}")
    print(f"  max = {global_max.tolist()}")
    print("99th percentile:")
    print(f"  min = {p1.tolist()}")
    print(f"  max = {p99.tolist()}")
    print("95th percentile:")
    print(f"  min = {p5.tolist()}")
    print(f"  max = {p95.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .h5 episode files")
    parser.add_argument("--proprio", action="store_true", help="Also compute proprio (ee_pose + sensor_dist) stats")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(f"{args.data_dir}/**/*.h5", recursive=True))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    all_actions = []
    all_proprios = []

    for f_path in h5_files:
        try:
            with h5py.File(f_path, "r") as f:
                all_actions.append(f["action"][:].astype(np.float32))

                if args.proprio:
                    ee = f["observations"]["ee_pose"][:].astype(np.float32)
                    if "sensor_dist" in f["observations"]:
                        sd = f["observations"]["sensor_dist"][:].astype(np.float32)
                        if sd.ndim == 1:
                            sd = sd[:, None]
                        sd = np.where((sd < 0) | (sd > 20.0), 20.0, sd)
                        proprio = np.concatenate([ee, sd], axis=-1)
                    else:
                        proprio = ee
                    all_proprios.append(proprio)
        except Exception as e:
            print(f"[Warn] Skipping {f_path}: {e}")

    print(f"Episodes: {len(h5_files)}")

    # Action stats
    all_actions = np.concatenate(all_actions, axis=0)
    action_labels = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
    print_stats(all_actions, action_labels, "Action Stats")

    # Proprio stats
    if args.proprio and all_proprios:
        all_proprios = np.concatenate(all_proprios, axis=0)
        proprio_labels = ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz", "grip", "sensor"]
        print_stats(all_proprios, proprio_labels, "Proprio Stats (raw, before normalization)")


if __name__ == "__main__":
    main()
