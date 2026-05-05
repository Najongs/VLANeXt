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

python Sim/filter_outliers.py \
    --data-dir /home/najo/NAS/VLANeXt/dataset/fine_align/fine_align_00 \
    --spike-ratio 2.0 \
    --pos-sigma 2.5 \
    --max-range 50 \
    --max-detour 3.0 \
    --max-path-length 50 \
    --max-rot 0.02 \
    --max-steps 250 \
    --min-steps 30 \
    --execute

python Sim/filter_outliers.py \
    --data-dir /data/public/NAS/VLANeXt/dataset/fine_align/10mm_fine_align_00 \
    --spike-ratio 2.0 \
    --pos-sigma 2.5 \
    --max-range 50 \
    --max-detour 3.0 \
    --max-path-length 50 \
    --max-rot 0.02 \
    --max-steps 250 \
    --min-steps 30 \
    --high-action-thr 0.4 \
    --high-action-frames 5 \
    --high-rot-thr 0.005 \
    --high-rot-frames 5 \
    --execute

"""

import os
import glob
import shutil
import argparse
import numpy as np
import h5py


def _print_action_stats(h5_files, prefix=""):
    """Load all actions and print min/max, p99, p95 normalization stats."""
    print(f"\nRecomputing action normalization on {len(h5_files)} {prefix} episodes...")
    all_actions = []
    for f_path in h5_files:
        with h5py.File(f_path, 'r') as f:
            all_actions.append(f['action'][:].astype(np.float32))
    all_actions = np.concatenate(all_actions, axis=0)

    new_min = np.min(all_actions, axis=0)
    new_max = np.max(all_actions, axis=0)
    p1 = np.percentile(all_actions, 1, axis=0)
    p5 = np.percentile(all_actions, 5, axis=0)
    p95 = np.percentile(all_actions, 95, axis=0)
    p99 = np.percentile(all_actions, 99, axis=0)

    print(f"Total {prefix} steps: {len(all_actions)}")
    print(f"\n--- 100% (min/max) ---")
    print(f"action_min = {new_min.tolist()}")
    print(f"action_max = {new_max.tolist()}")
    print(f"\n--- 99th percentile ---")
    print(f"action_min = {p1.tolist()}")
    print(f"action_max = {p99.tolist()}")
    print(f"\n--- 95th percentile ---")
    print(f"action_min = {p5.tolist()}")
    print(f"action_max = {p95.tolist()}")
    print("\n>>> Update these values in src/datasets/sim_act_align.py <<<")


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

    # Path length: 총 이동 거리 (mm) — 돌아가는 trajectory는 이 값이 큼
    diffs = np.diff(all_pos, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Direct distance: 시작→끝 직선 거리 (mm)
    direct_dist = float(np.linalg.norm(all_pos[-1] - all_pos[0]))

    # Detour ratio: path_length / direct_dist — 1에 가까우면 직선, 클수록 돌아감
    detour_ratio = path_length / max(direct_dist, 1e-6)

    # Rotation delta stats (dims 3:6 = roll, pitch, yaw)
    rot_deltas = act[:, 3:6]
    max_abs_rot = float(np.max(np.abs(rot_deltas)))
    n_rot_outlier_frames = 0  # will be set by caller if threshold given

    # Per-frame max abs trans/rot for plateau-style outlier detection
    per_frame_max_trans = np.max(np.abs(act[:, :3]), axis=1)  # (N,)
    per_frame_max_rot = np.max(np.abs(act[:, 3:6]), axis=1)   # (N,)

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
        'path_length': path_length,
        'direct_dist': direct_dist,
        'detour_ratio': detour_ratio,
        'max_abs_rot': max_abs_rot,
        'per_frame_max_trans': per_frame_max_trans,
        'per_frame_max_rot': per_frame_max_rot,
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
    parser.add_argument('--max-detour', type=float, default=0.0,
                        help="Flag episodes with detour_ratio (path_length/direct_dist) > this. 0=disable (default: 0)")
    parser.add_argument('--max-path-length', type=float, default=0.0,
                        help="Flag episodes with total path length > this (mm). 0=disable (default: 0)")
    parser.add_argument('--max-rot', type=float, default=0.0,
                        help="Flag episodes with any rotation delta > this (rad). 0=disable (default: 0)")
    parser.add_argument('--min-steps', type=int, default=0,
                        help="Flag episodes with fewer than this many steps. 0=disable (default: 0)")
    parser.add_argument('--max-steps', type=int, default=0,
                        help="Flag episodes with more than this many steps. 0=disable (default: 0)")
    parser.add_argument('--high-action-thr', type=float, default=0.0,
                        help="Per-frame |trans action| threshold for plateau-outlier detection (mm). 0=disable")
    parser.add_argument('--high-action-frames', type=int, default=0,
                        help="Flag episodes with > this many frames exceeding --high-action-thr. 0=disable")
    parser.add_argument('--high-rot-thr', type=float, default=0.0,
                        help="Per-frame |rot action| threshold (rad). 0=disable")
    parser.add_argument('--high-rot-frames', type=int, default=0,
                        help="Flag episodes with > this many frames exceeding --high-rot-thr. 0=disable")
    parser.add_argument('--execute', action='store_true',
                        help="Actually move files. Without this flag, only reports.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.h5'), recursive=True))
    if not files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Scanning {len(files)} episodes...")
    print(f"Criteria:")
    print(f"  1. Action spike: spike_ratio > {args.spike_ratio} AND peak_mag > {args.min_peak_mag}")
    if args.pos_sigma > 0:
        print(f"  2. Position outlier: start_pos > {args.pos_sigma} sigma from median")
    if args.max_range > 0:
        print(f"  3. Trajectory range: max_range > {args.max_range} mm")
    if args.max_detour > 0:
        print(f"  4. Detour ratio: path_length/direct_dist > {args.max_detour}")
    if args.max_path_length > 0:
        print(f"  5. Path length: total > {args.max_path_length} mm")
    if args.max_rot > 0:
        print(f"  6. Rotation outlier: max |rot delta| > {args.max_rot} rad")
    if args.min_steps > 0:
        print(f"  7. Min step count: n_steps < {args.min_steps}")
    if args.max_steps > 0:
        print(f"  8. Max step count: n_steps > {args.max_steps}")
    if args.high_action_thr > 0 and args.high_action_frames > 0:
        print(f"  9. Trans plateau: > {args.high_action_frames} frames with |trans| > {args.high_action_thr}")
    if args.high_rot_thr > 0 and args.high_rot_frames > 0:
        print(f" 10. Rot plateau:   > {args.high_rot_frames} frames with |rot|   > {args.high_rot_thr}")
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

    # Print trajectory stats for threshold tuning
    if all_stats:
        path_lengths = [s['path_length'] for s in all_stats]
        detour_ratios = [s['detour_ratio'] for s in all_stats]
        print(f"Trajectory stats ({len(all_stats)} episodes):")
        print(f"  path_length: median={np.median(path_lengths):.1f}mm, "
              f"mean={np.mean(path_lengths):.1f}mm, "
              f"std={np.std(path_lengths):.1f}mm, "
              f"max={np.max(path_lengths):.1f}mm")
        print(f"  detour_ratio: median={np.median(detour_ratios):.2f}, "
              f"mean={np.mean(detour_ratios):.2f}, "
              f"std={np.std(detour_ratios):.2f}, "
              f"max={np.max(detour_ratios):.2f}")
        print()

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

    # --- 4. Detour ratio filter (갔다가 돌아오는 trajectory) ---
    if all_stats and args.max_detour > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['detour_ratio'] > args.max_detour and s['path'] not in outlier_paths:
                s['reason'] = f'detour (ratio={s["detour_ratio"]:.1f}x, path={s["path_length"]:.0f}mm, direct={s["direct_dist"]:.0f}mm)'
                outliers.append(s)

    # --- 5. Absolute path length filter ---
    if all_stats and args.max_path_length > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['path_length'] > args.max_path_length and s['path'] not in outlier_paths:
                s['reason'] = f'long_path ({s["path_length"]:.0f}mm > {args.max_path_length}mm)'
                outliers.append(s)

    # --- 6. Rotation outlier filter ---
    if all_stats and args.max_rot > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['max_abs_rot'] > args.max_rot and s['path'] not in outlier_paths:
                s['reason'] = f'rot_outlier (max_rot={s["max_abs_rot"]:.4f} > {args.max_rot})'
                outliers.append(s)

    # --- 7. Min step count filter ---
    if all_stats and args.min_steps > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['n_steps'] < args.min_steps and s['path'] not in outlier_paths:
                s['reason'] = f'too_few_steps ({s["n_steps"]} < {args.min_steps})'
                outliers.append(s)

    # --- 8. Max step count filter ---
    if all_stats and args.max_steps > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            if s['n_steps'] > args.max_steps and s['path'] not in outlier_paths:
                s['reason'] = f'too_many_steps ({s["n_steps"]} > {args.max_steps})'
                outliers.append(s)

    # --- 9. Trans plateau outlier (many frames with high |trans| action) ---
    if all_stats and args.high_action_thr > 0 and args.high_action_frames > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            n_high = int(np.sum(s['per_frame_max_trans'] > args.high_action_thr))
            if n_high > args.high_action_frames and s['path'] not in outlier_paths:
                s['reason'] = f'trans_plateau (n_high={n_high} > {args.high_action_frames}, thr={args.high_action_thr})'
                outliers.append(s)

    # --- 10. Rot plateau outlier ---
    if all_stats and args.high_rot_thr > 0 and args.high_rot_frames > 0:
        outlier_paths = {o['path'] for o in outliers}
        for s in all_stats:
            n_high = int(np.sum(s['per_frame_max_rot'] > args.high_rot_thr))
            if n_high > args.high_rot_frames and s['path'] not in outlier_paths:
                s['reason'] = f'rot_plateau (n_high={n_high} > {args.high_rot_frames}, thr={args.high_rot_thr})'
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
            # 하위폴더 구조 유지: data_dir/subdir/file.h5 → _outliers/subdir/file.h5
            rel = os.path.relpath(src, args.data_dir)
            dst = os.path.join(quarantine_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

        print(f"\nMoved {len(outliers)} files to {quarantine_dir}/")

        # Recompute action stats on clean data
        clean_files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.h5'), recursive=True))
        _print_action_stats(clean_files, prefix="clean")
    elif not args.execute and outliers:
        print(f"\nDry run complete. Add --execute to actually move files.")

    # Always print stats for all data (dry run or execute)
    if all_stats and not args.execute:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.h5'), recursive=True))
        _print_action_stats(all_files, prefix="all")


if __name__ == '__main__':
    main()
