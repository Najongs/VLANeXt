"""
Outlier episode filter for simulation dataset.

Detects and removes episodes with action spikes caused by IK discontinuities
at phase transitions (align → insert).

Usage:
    # Dry run (just report, don't move files):
    python Sim/filter_outliers.py --data-dir dataset/New_1/collected_data_merged --spike-ratio 2.0

    # Actually move outliers to quarantine folder:
    python Sim/filter_outliers.py --data-dir dataset/New_1/collected_data_merged --execute

    # Custom threshold:
    python Sim/filter_outliers.py --data-dir dataset/New_1/collected_data_merged --spike-ratio 5.0 --execute
"""

import os
import glob
import shutil
import argparse
import numpy as np
import h5py


def analyze_episode(h5_path):
    """Analyze a single episode for action spikes. Returns dict with stats."""
    with h5py.File(h5_path, 'r') as f:
        act = f['action'][:].astype(np.float32)
        phase = f['phase'][:].astype(np.int32)
        n_steps = act.shape[0]

    pos_mag = np.linalg.norm(act[:, :3], axis=1)
    peak_idx = int(np.argmax(pos_mag))
    peak_mag = float(pos_mag[peak_idx])

    # Compute spike ratio: peak vs median of neighbors (±5 steps)
    window = 5
    start = max(0, peak_idx - window)
    end = min(n_steps, peak_idx + window + 1)
    neighbors = np.concatenate([pos_mag[start:peak_idx], pos_mag[peak_idx + 1:end]])
    median_neighbor = float(np.median(neighbors)) if len(neighbors) > 0 else peak_mag
    spike_ratio = peak_mag / max(median_neighbor, 1e-6)

    return {
        'path': h5_path,
        'n_steps': n_steps,
        'peak_idx': peak_idx,
        'peak_mag': peak_mag,
        'peak_phase': int(phase[peak_idx]),
        'median_neighbor': median_neighbor,
        'spike_ratio': spike_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter outlier episodes from sim dataset")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--spike-ratio', type=float, default=5.0,
                        help="Flag episodes where peak/neighbor_median > this ratio (default: 5.0)")
    parser.add_argument('--min-peak-mag', type=float, default=1.0,
                        help="Only flag if peak magnitude also exceeds this (default: 1.0)")
    parser.add_argument('--execute', action='store_true',
                        help="Actually move files. Without this flag, only reports.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    if not files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Scanning {len(files)} episodes...")
    print(f"Criteria: spike_ratio > {args.spike_ratio} AND peak_mag > {args.min_peak_mag}")
    print()

    outliers = []
    for i, f_path in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(files)}")
        try:
            stats = analyze_episode(f_path)
        except Exception as e:
            print(f"  [WARN] Failed to read {f_path}: {e}")
            outliers.append({'path': f_path, 'reason': f'read_error: {e}'})
            continue

        if stats['spike_ratio'] > args.spike_ratio and stats['peak_mag'] > args.min_peak_mag:
            stats['reason'] = 'action_spike'
            outliers.append(stats)

    print(f"\nResults: {len(outliers)} outliers / {len(files)} total ({len(outliers)/len(files)*100:.2f}%)")
    print(f"Clean episodes: {len(files) - len(outliers)}")

    if outliers:
        print(f"\nTop outliers:")
        sorted_outliers = sorted(outliers, key=lambda x: x.get('peak_mag', 0), reverse=True)
        for o in sorted_outliers[:15]:
            name = os.path.basename(o['path'])
            if 'spike_ratio' in o:
                print(f"  {name}  peak={o['peak_mag']:.3f} at step {o['peak_idx']} "
                      f"(phase={o['peak_phase']})  ratio={o['spike_ratio']:.1f}x")
            else:
                print(f"  {name}  reason={o['reason']}")

    if args.execute and outliers:
        quarantine_dir = os.path.join(args.data_dir, '_outliers')
        os.makedirs(quarantine_dir, exist_ok=True)

        for o in outliers:
            src = o['path']
            dst = os.path.join(quarantine_dir, os.path.basename(src))
            shutil.move(src, dst)

        print(f"\nMoved {len(outliers)} files to {quarantine_dir}/")

        # Recompute action stats on clean data
        clean_files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
        print(f"\nRecomputing action normalization on {len(clean_files)} clean episodes...")
        all_actions = []
        for f_path in clean_files:
            with h5py.File(f_path, 'r') as f:
                all_actions.append(f['action'][:].astype(np.float32))
        all_actions = np.concatenate(all_actions, axis=0)

        new_min = np.min(all_actions, axis=0)
        new_max = np.max(all_actions, axis=0)
        print(f"Total clean steps: {len(all_actions)}")
        print(f"\naction_min_sim = {new_min.tolist()}")
        print(f"action_max_sim = {new_max.tolist()}")
        print("\n>>> Update these values in src/datasets/sim_act.py <<<")
    elif not args.execute and outliers:
        print(f"\nDry run complete. Add --execute to actually move files.")


if __name__ == '__main__':
    main()
