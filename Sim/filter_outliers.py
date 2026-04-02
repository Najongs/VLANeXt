"""
Outlier episode filter for simulation dataset.

Detects and removes episodes with action spikes caused by IK discontinuities
at phase transitions (align → insert).

Usage:
    # Dry run (just report, don't move files):
    python Sim/filter_outliers.py --data-dir dataset/fine_align/collected_data_merged --spike-ratio 2.0

    # Custom threshold:
    python Sim/filter_outliers.py --data-dir dataset/fine_align/collected_data_merged --spike-ratio 2.0 --execute

    # Dry run — position outlier 3sigma + trajectory range 100mm 초과 필터
    python Sim/filter_outliers.py \
        --data-dir dataset/fine_align/collected_data_merged \
        --spike-ratio 2.0 \
        --pos-sigma 2.5 \
        --max-range 100

    # 실행
    python Sim/filter_outliers.py \
        --data-dir dataset/fine_align/collected_data_merged \
        --spike-ratio 2.0 \
        --pos-sigma 2.5 \
        --max-range 100 \
        --execute
"""

import os
import glob
import shutil
import argparse
import numpy as np
import h5py


def analyze_episode(h5_path):
    """Analyze a single episode for action spikes and trajectory outliers. Returns dict with stats."""
    with h5py.File(h5_path, 'r') as f:
        act = f['action'][:].astype(np.float32)
        phase = f['phase'][:].astype(np.int32)
        ee_pose = f['observations/ee_pose'][:].astype(np.float32)
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

    # Trajectory position stats (ee_pose[:3] = xyz in mm)
    start_pos = ee_pose[0, :3]     # 시작점 xyz (mm)
    all_pos = ee_pose[:, :3]       # 전체 trajectory xyz
    pos_range = np.max(all_pos, axis=0) - np.min(all_pos, axis=0)  # trajectory 범위
    max_range = float(np.max(pos_range))  # 가장 큰 축의 범위

    return {
        'path': h5_path,
        'n_steps': n_steps,
        'peak_idx': peak_idx,
        'peak_mag': peak_mag,
        'peak_phase': int(phase[peak_idx]),
        'median_neighbor': median_neighbor,
        'spike_ratio': spike_ratio,
        'start_pos': start_pos,
        'max_range': max_range,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter outlier episodes from sim dataset")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--spike-ratio', type=float, default=5.0,
                        help="Flag episodes where peak/neighbor_median > this ratio (default: 5.0)")
    parser.add_argument('--min-peak-mag', type=float, default=1.0,
                        help="Only flag if peak magnitude also exceeds this (default: 1.0)")
    parser.add_argument('--pos-sigma', type=float, default=0.0,
                        help="Flag episodes with start position > N sigma from median. 0=disable (default: 0)")
    parser.add_argument('--max-range', type=float, default=0.0,
                        help="Flag episodes with trajectory range > this (mm). 0=disable (default: 0)")
    parser.add_argument('--execute', action='store_true',
                        help="Actually move files. Without this flag, only reports.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    if not files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Scanning {len(files)} episodes...")
    print(f"Criteria:")
    print(f"  1. Action spike: spike_ratio > {args.spike_ratio} AND peak_mag > {args.min_peak_mag}")
    print(f"  2. Position outlier: start_pos > {args.pos_sigma} sigma from median")
    if args.max_range > 0:
        print(f"  3. Trajectory range: max_range > {args.max_range} mm")
    print()

    all_stats = []
    errors = []
    for i, f_path in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(files)}")
        try:
            stats = analyze_episode(f_path)
            all_stats.append(stats)
        except Exception as e:
            print(f"  [WARN] Failed to read {f_path}: {e}")
            errors.append({'path': f_path, 'reason': f'read_error: {e}'})

    # --- 1. Action spike filter ---
    outliers = list(errors)
    for stats in all_stats:
        if stats['spike_ratio'] > args.spike_ratio and stats['peak_mag'] > args.min_peak_mag:
            stats['reason'] = 'action_spike'
            outliers.append(stats)

    # --- 2. Position outlier filter (median + sigma) ---
    if all_stats and args.pos_sigma > 0:
        start_positions = np.array([s['start_pos'] for s in all_stats])
        median_pos = np.median(start_positions, axis=0)
        dists = np.linalg.norm(start_positions - median_pos, axis=1)
        dist_median = np.median(dists)
        dist_std = np.std(dists)
        threshold = dist_median + args.pos_sigma * dist_std

        print(f"Position stats: median={median_pos}, dist_median={dist_median:.1f}mm, "
              f"dist_std={dist_std:.1f}mm, threshold={threshold:.1f}mm")

        outlier_paths = {o['path'] for o in outliers}
        for s, d in zip(all_stats, dists):
            if d > threshold and s['path'] not in outlier_paths:
                s['reason'] = f'pos_outlier (dist={d:.1f}mm > {threshold:.1f}mm)'
                outliers.append(s)

    # --- 3. Trajectory range filter ---
    if all_stats and args.max_range > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['max_range'] > args.max_range and s['path'] not in outlier_paths:
                s['reason'] = f'large_range ({s["max_range"]:.1f}mm > {args.max_range}mm)'
                outliers.append(s)

    print(f"\nResults: {len(outliers)} outliers / {len(files)} total ({len(outliers)/len(files)*100:.2f}%)")
    print(f"Clean episodes: {len(files) - len(outliers)}")

    if outliers:
        print(f"\nOutliers by reason:")
        reasons = {}
        for o in outliers:
            r = o.get('reason', 'unknown').split(' ')[0]
            reasons[r] = reasons.get(r, 0) + 1
        for r, c in sorted(reasons.items()):
            print(f"  {r}: {c}")

        print(f"\nTop outliers:")
        sorted_outliers = sorted(outliers, key=lambda x: x.get('peak_mag', 0), reverse=True)
        for o in sorted_outliers[:20]:
            name = os.path.basename(o['path'])
            reason = o.get('reason', 'unknown')
            if 'spike_ratio' in o:
                print(f"  {name}  peak={o['peak_mag']:.3f} ratio={o['spike_ratio']:.1f}x  [{reason}]")
            else:
                print(f"  {name}  [{reason}]")

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
