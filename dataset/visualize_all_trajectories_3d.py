#!/usr/bin/env python3
"""

python dataset/visualize_all_trajectories_3d.py --dataset_path "/data/public/NAS/VLANeXt/dataset/fine_align/approach_test/collected_data_merged"
python dataset/visualize_all_trajectories_3d.py --analyze --dataset_path "/data/public/NAS/VLANeXt/dataset/fine_align/approach_test/collected_data_merged"
Visualize all trajectories from Eye_trocar dataset in 3D space
Shows all trajectories with their start and end points marked
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob
from tqdm import tqdm

def load_trajectory(h5_file_path):
    """Load trajectory data from h5 file"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Get end-effector pose from observations
            # ee_pose contains [x, y, z, roll, pitch, yaw] in absolute coordinates
            ee_pose = f['observations']['ee_pose'][:]
            # Extract position (first 3 dimensions)
            position = ee_pose[:, :3]

            # Convert to mm if values are in meters (< 10 means likely in meters)
            if np.abs(position).max() < 10:
                position = position * 1000.0  # Convert m to mm

            # Load needle_tip_pos if available
            needle_tip = None
            if 'needle_tip_pos' in f['observations']:
                needle_tip = f['observations']['needle_tip_pos'][:][:, :3]
                if np.abs(needle_tip).max() < 10:
                    needle_tip = needle_tip * 1000.0

            # Load trocar_entry_pos if available
            trocar_entry = None
            if 'trocar_entry_pos' in f['observations']:
                trocar_entry = f['observations']['trocar_entry_pos'][:][:, :3]
                if np.abs(trocar_entry).max() < 10:
                    trocar_entry = trocar_entry * 1000.0

            return position, needle_tip, trocar_entry
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None, None, None

def visualize_all_trajectories(dataset_path, output_path=None, max_trajectories=None):
    """
    Visualize all trajectories in 3D space

    Args:
        dataset_path: Path to Eye_trocar dataset
        output_path: Path to save the visualization (optional)
        max_trajectories: Maximum number of trajectories to visualize (for testing)
    """
    # Find all h5 files
    h5_files = glob.glob(str(Path(dataset_path) / "**/*.h5"), recursive=True)

    if max_trajectories:
        h5_files = h5_files[:max_trajectories]

    print(f"Found {len(h5_files)} trajectory files")

    # Load all trajectories and group by folder (date/session)
    trajectories = []
    start_points = []
    end_points = []
    needle_tip_ends = []
    trocar_positions = []
    folder_labels = []  # Track which folder each trajectory belongs to

    print("Loading trajectories...")
    for h5_file in tqdm(h5_files):
        position, needle_tip, trocar_entry = load_trajectory(h5_file)
        if position is not None and len(position) > 0:
            trajectories.append(position)
            start_points.append(position[0])
            end_points.append(position[-1])
            if needle_tip is not None:
                needle_tip_ends.append(needle_tip[-1])
            if trocar_entry is not None:
                trocar_positions.append(trocar_entry[0])

            # Extract folder name hierarchy
            # For real data: date/person (e.g., "260106/1_MIN")
            # For sim data: just use single folder
            path_parts = Path(h5_file).parts

            # Check if this is real or sim data
            is_real_data = 'Eye_trocar_sim' not in h5_file

            if is_real_data:
                # Real data: find person folder only (ignore date)
                person_folder = None

                for i, part in enumerate(path_parts):
                    # Check if this is a person folder (e.g., 1_MIN, 2_JYT)
                    if '_' in part and part[0].isdigit():
                        person_folder = part
                        break

                # If person folder found, use it; otherwise treat as 1_KTY
                if person_folder:
                    folder_name = person_folder
                else:
                    # All "기타" episodes go to 1_KTY
                    folder_name = "1_KTY"
            else:
                # Sim data: just use a simple label
                folder_name = "Simulation"

            folder_labels.append(folder_name)

    print(f"Successfully loaded {len(trajectories)} trajectories")

    if len(trajectories) == 0:
        print("No valid trajectories found!")
        return

    # Convert to numpy arrays
    start_points = np.array(start_points)
    end_points = np.array(end_points)
    has_needle_tip = len(needle_tip_ends) > 0
    has_trocar = len(trocar_positions) > 0
    if has_needle_tip:
        needle_tip_ends = np.array(needle_tip_ends)
    if has_trocar:
        trocar_positions = np.array(trocar_positions)

    # Create folder to color mapping
    unique_folders = sorted(list(set(folder_labels)))
    folder_to_color = {folder: i for i, folder in enumerate(unique_folders)}

    # Print summary with colors
    print(f"\nTotal Contributions by Worker:")
    print("=" * 80)

    # Count contributions per worker
    worker_counts = {}
    for folder in unique_folders:
        worker_counts[folder] = folder_labels.count(folder)

    # Sort by contribution count (descending)
    sorted_workers = sorted(worker_counts.items(), key=lambda x: x[1], reverse=True)

    # Print worker contributions
    for worker, count in sorted_workers:
        color_idx = folder_to_color[worker]
        print(f"  👤 {worker:30s}: {count:4d} episodes (Color #{color_idx})")

    print("=" * 80)
    print(f"Total: {len(trajectories)} trajectories across {len(unique_folders)} workers")

    # Create 3D visualization with legend
    fig = plt.figure(figsize=(20, 12))

    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot all trajectories colored by folder and collect for legend
    print("Plotting trajectories...")
    plotted_folders = {}
    for i, traj in enumerate(trajectories):
        # Use colormap based on folder
        folder_name = folder_labels[i]
        folder_idx = folder_to_color[folder_name]
        color = plt.cm.tab20(folder_idx / len(unique_folders))

        # Plot trajectory
        line, = ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                alpha=0.4, linewidth=0.5, color=color)

        # Save one line per folder for legend
        if folder_name not in plotted_folders:
            plotted_folders[folder_name] = line

    # Plot start points (green) - 시작점
    start_scatter = ax1.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2],
               c='green', marker='o', s=50, alpha=0.8, edgecolors='darkgreen', linewidths=1)

    # Plot end points (red) - 끝점
    end_scatter = ax1.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2],
               c='red', marker='x', s=50, alpha=0.8, linewidths=2)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'All Trajectories 3D View (n={len(trajectories)})')

    # Create legend with folder counts (only for first plot to avoid clutter)
    legend_elements = [start_scatter, end_scatter]
    legend_labels = ['Start Points', 'End Points']

    # Add up to 10 most common folders to legend
    folder_counts = [(folder, folder_labels.count(folder)) for folder in unique_folders]
    folder_counts.sort(key=lambda x: x[1], reverse=True)
    for folder, count in folder_counts[:10]:
        if folder in plotted_folders:
            legend_elements.append(plotted_folders[folder])
            legend_labels.append(f'{folder} (n={count})')

    ax1.legend(legend_elements, legend_labels, loc='upper left', fontsize=8)
    ax1.grid(True)

    # XY projection (Top View)
    ax2 = fig.add_subplot(222)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=0.5, color=color)
    ax2.scatter(start_points[:, 0], start_points[:, 1], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax2.scatter(end_points[:, 0], end_points[:, 1], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # XZ projection (Front View)
    ax3 = fig.add_subplot(223)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax3.plot(traj[:, 0], traj[:, 2], alpha=0.4, linewidth=0.5, color=color)
    ax3.scatter(start_points[:, 0], start_points[:, 2], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax3.scatter(end_points[:, 0], end_points[:, 2], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection (Front View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # YZ projection (Side View)
    ax4 = fig.add_subplot(224)
    for i, traj in enumerate(trajectories):
        folder_idx = folder_to_color[folder_labels[i]]
        color = plt.cm.tab20(folder_idx / len(unique_folders))
        ax4.plot(traj[:, 1], traj[:, 2], alpha=0.4, linewidth=0.5, color=color)
    ax4.scatter(start_points[:, 1], start_points[:, 2], c='green', marker='o', s=30, alpha=0.8,
                label='Start', edgecolors='darkgreen', linewidths=1)
    ax4.scatter(end_points[:, 1], end_points[:, 2], c='red', marker='x', s=30, alpha=0.8,
                label='End', linewidths=2)
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection (Side View)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()

    # Print statistics
    print("\n=== Trajectory Statistics ===")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nStart Points Statistics:")
    print(f"  X: min={start_points[:, 0].min():.2f}, max={start_points[:, 0].max():.2f}, mean={start_points[:, 0].mean():.2f}")
    print(f"  Y: min={start_points[:, 1].min():.2f}, max={start_points[:, 1].max():.2f}, mean={start_points[:, 1].mean():.2f}")
    print(f"  Z: min={start_points[:, 2].min():.2f}, max={start_points[:, 2].max():.2f}, mean={start_points[:, 2].mean():.2f}")
    print(f"\nEnd Points Statistics:")
    print(f"  X: min={end_points[:, 0].min():.2f}, max={end_points[:, 0].max():.2f}, mean={end_points[:, 0].mean():.2f}")
    print(f"  Y: min={end_points[:, 1].min():.2f}, max={end_points[:, 1].max():.2f}, mean={end_points[:, 1].mean():.2f}")
    print(f"  Z: min={end_points[:, 2].min():.2f}, max={end_points[:, 2].max():.2f}, mean={end_points[:, 2].mean():.2f}")

    # Calculate average trajectory length
    traj_lengths = [len(traj) for traj in trajectories]
    print(f"\nTrajectory Length Statistics:")
    print(f"  Min: {min(traj_lengths)} steps")
    print(f"  Max: {max(traj_lengths)} steps")
    print(f"  Mean: {np.mean(traj_lengths):.1f} steps")
    print(f"  Median: {np.median(traj_lengths):.1f} steps")

    return trajectories, start_points, end_points


def analyze_dataset(dataset_path, output_path=None):
    """
    Analyze training dataset distribution: perturbation, actions, trajectory quality.
    Designed to diagnose directional bias and coverage gaps.
    """
    h5_files = sorted(glob.glob(str(Path(dataset_path) / "**/*.h5"), recursive=True))
    print(f"Found {len(h5_files)} episodes")

    # Collect per-episode metadata
    perturb_xyz_list = []
    perturb_angle_list = []
    traj_lengths = []
    start_positions = []
    end_positions = []
    needle_tip_starts = []
    needle_tip_ends = []
    trocar_positions = []
    final_dists = []
    all_actions = []
    action_per_episode_stats = []  # (mean, std, min, max) per episode
    ee_pose_ranges = []  # (min, max) per dimension across episode

    print("Loading episodes...")
    for h5_file in tqdm(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                traj_len = f['action'].shape[0]
                traj_lengths.append(traj_len)

                # Perturbation metadata
                if 'metadata' in f and 'perturb_xyz_mm' in f['metadata']:
                    perturb_xyz_list.append(f['metadata']['perturb_xyz_mm'][:])
                    perturb_angle_list.append(float(f['metadata']['perturb_angle_deg'][()]))

                # EE pose
                ee_pose = f['observations']['ee_pose'][:].astype(np.float32)
                start_positions.append(ee_pose[0, :3])
                end_positions.append(ee_pose[-1, :3])
                ee_pose_ranges.append((ee_pose.min(axis=0), ee_pose.max(axis=0)))

                # Needle tip & trocar
                if 'needle_tip_pos' in f['observations']:
                    needle_tip_starts.append(f['observations']['needle_tip_pos'][0, :3])
                    needle_tip_ends.append(f['observations']['needle_tip_pos'][-1, :3])
                if 'trocar_entry_pos' in f['observations']:
                    trocar_positions.append(f['observations']['trocar_entry_pos'][0, :3])

                # Final distance (needle tip to trocar)
                if 'needle_tip_pos' in f['observations'] and 'trocar_entry_pos' in f['observations']:
                    nt = f['observations']['needle_tip_pos'][-1, :3]
                    te = f['observations']['trocar_entry_pos'][-1, :3]
                    final_dists.append(np.linalg.norm(nt - te))

                # Actions
                actions = f['action'][:].astype(np.float32)
                all_actions.append(actions)
                action_per_episode_stats.append({
                    'mean': actions.mean(axis=0),
                    'std': actions.std(axis=0),
                    'min': actions.min(axis=0),
                    'max': actions.max(axis=0),
                    'abs_mean': np.abs(actions).mean(axis=0),
                })
        except Exception as e:
            print(f"  [Warn] Skipping {h5_file}: {e}")
            continue

    # Convert to arrays
    traj_lengths = np.array(traj_lengths)
    start_positions = np.array(start_positions)
    end_positions = np.array(end_positions)
    has_perturb = len(perturb_xyz_list) > 0
    if has_perturb:
        perturb_xyz = np.array(perturb_xyz_list)
        perturb_angle = np.array(perturb_angle_list)
    has_final_dist = len(final_dists) > 0
    if has_final_dist:
        final_dists = np.array(final_dists)

    # Concatenate all actions for global stats
    all_actions_cat = np.concatenate(all_actions, axis=0)

    out_dir = Path(output_path or dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # TEXT REPORT
    # ================================================================
    report_lines = []
    def p(s=""):
        report_lines.append(s)
        print(s)

    p("=" * 70)
    p(f"  Dataset Analysis: {dataset_path}")
    p(f"  Episodes: {len(traj_lengths)}")
    p("=" * 70)

    p(f"\n--- Trajectory Length ---")
    p(f"  Mean: {traj_lengths.mean():.1f}, Median: {np.median(traj_lengths):.1f}")
    p(f"  Min: {traj_lengths.min()}, Max: {traj_lengths.max()}, Std: {traj_lengths.std():.1f}")
    p(f"  Total timesteps: {traj_lengths.sum():,}")

    if has_final_dist:
        p(f"\n--- Final Distance (needle tip → trocar) ---")
        p(f"  Mean: {final_dists.mean():.2f}mm, Median: {np.median(final_dists):.2f}mm")
        p(f"  Min: {final_dists.min():.2f}mm, Max: {final_dists.max():.2f}mm")
        p(f"  < 1mm: {(final_dists < 1).sum()} ({(final_dists < 1).mean()*100:.1f}%)")
        p(f"  < 3mm: {(final_dists < 3).sum()} ({(final_dists < 3).mean()*100:.1f}%)")
        p(f"  < 5mm: {(final_dists < 5).sum()} ({(final_dists < 5).mean()*100:.1f}%)")

    if has_perturb:
        p(f"\n--- Perturbation Distribution ---")
        perturb_dist = np.linalg.norm(perturb_xyz, axis=1)
        p(f"  Distance — Mean: {perturb_dist.mean():.2f}mm, Std: {perturb_dist.std():.2f}mm, Range: [{perturb_dist.min():.2f}, {perturb_dist.max():.2f}]")
        p(f"  Angle   — Mean: {np.abs(perturb_angle).mean():.2f}deg, Std: {perturb_angle.std():.2f}deg, Range: [{perturb_angle.min():.2f}, {perturb_angle.max():.2f}]")
        for ax_i, ax_name in enumerate(['X', 'Y', 'Z']):
            vals = perturb_xyz[:, ax_i]
            neg = vals[vals < -0.5]
            pos = vals[vals > 0.5]
            p(f"  {ax_name}: mean={vals.mean():+.2f}, std={vals.std():.2f}, range=[{vals.min():+.2f}, {vals.max():+.2f}]")
            p(f"     negative(<-0.5): {len(neg)} ({len(neg)/len(vals)*100:.1f}%), positive(>+0.5): {len(pos)} ({len(pos)/len(vals)*100:.1f}%)")

    p(f"\n--- Action Statistics (all timesteps, {len(all_actions_cat)} total) ---")
    dim_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    for d in range(min(7, all_actions_cat.shape[1])):
        vals = all_actions_cat[:, d]
        p(f"  {dim_names[d]:8s}: mean={vals.mean():+.6f}, std={vals.std():.6f}, "
          f"range=[{vals.min():+.6f}, {vals.max():+.6f}], "
          f"|mean|={np.abs(vals).mean():.6f}")

    # Action direction bias: how much more positive vs negative
    p(f"\n--- Action Direction Bias ---")
    for d in range(min(6, all_actions_cat.shape[1])):
        vals = all_actions_cat[:, d]
        n_pos = (vals > 0).sum()
        n_neg = (vals < 0).sum()
        pos_mean = vals[vals > 0].mean() if n_pos > 0 else 0
        neg_mean = vals[vals < 0].mean() if n_neg > 0 else 0
        p(f"  {dim_names[d]:8s}: pos={n_pos} ({n_pos/len(vals)*100:.1f}%, mean={pos_mean:+.4f}), "
          f"neg={n_neg} ({n_neg/len(vals)*100:.1f}%, mean={neg_mean:+.4f})")

    p(f"\n--- EE Pose Range (position, mm) ---")
    for ax_i, ax_name in enumerate(['X', 'Y', 'Z']):
        starts = start_positions[:, ax_i]
        ends = end_positions[:, ax_i]
        p(f"  {ax_name} start: [{starts.min():.1f}, {starts.max():.1f}], mean={starts.mean():.1f}, std={starts.std():.1f}")
        p(f"  {ax_name} end  : [{ends.min():.1f}, {ends.max():.1f}], mean={ends.mean():.1f}, std={ends.std():.1f}")

    # ================================================================
    # FIGURES
    # ================================================================
    plt.style.use('default')

    if has_perturb:
        # Figure 1: Perturbation distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Perturbation Distribution (n={len(perturb_xyz)})', fontsize=14)

        for ax_i, ax_name in enumerate(['X', 'Y', 'Z']):
            ax = axes[0, ax_i]
            vals = perturb_xyz[:, ax_i]
            ax.hist(vals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(vals.mean(), color='orange', linestyle='-', linewidth=2, label=f'mean={vals.mean():+.2f}')
            ax.set_xlabel(f'{ax_name} perturbation (mm)')
            ax.set_ylabel('Count')
            ax.set_title(f'{ax_name} Distribution')
            ax.legend()

        # 2D scatter: XY, XZ, YZ
        pairs = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
        for i, (n1, n2, i1, i2) in enumerate(pairs):
            ax = axes[1, i]
            ax.scatter(perturb_xyz[:, i1], perturb_xyz[:, i2], alpha=0.3, s=5, c='steelblue')
            ax.axhline(0, color='red', linestyle='--', alpha=0.3)
            ax.axvline(0, color='red', linestyle='--', alpha=0.3)
            ax.set_xlabel(f'{n1} (mm)')
            ax.set_ylabel(f'{n2} (mm)')
            ax.set_title(f'{n1}{n2} Perturbation')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'analysis_perturbation.png', dpi=200, bbox_inches='tight')
        plt.close()
        p(f"\nSaved: {out_dir / 'analysis_perturbation.png'}")

    # Figure 2: Action distribution per dimension
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'Action Distribution per Dimension (n={len(all_actions_cat)})', fontsize=14)
    for d in range(min(7, all_actions_cat.shape[1])):
        ax = axes[d // 4, d % 4]
        vals = all_actions_cat[:, d]
        # Clip for visualization (99.5th percentile)
        p995 = np.percentile(np.abs(vals), 99.5)
        vals_clip = vals[np.abs(vals) <= p995]
        ax.hist(vals_clip, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(vals.mean(), color='orange', linewidth=2, label=f'mean={vals.mean():+.4f}')
        ax.set_title(f'{dim_names[d]}')
        ax.legend(fontsize=7)
    # Hide unused subplot
    if all_actions_cat.shape[1] < 8:
        axes[1, 3].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'analysis_actions.png', dpi=200, bbox_inches='tight')
    plt.close()
    p(f"Saved: {out_dir / 'analysis_actions.png'}")

    # Figure 3: Start/End position distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Start vs End Position Distribution', fontsize=14)
    for ax_i, ax_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, ax_i]
        ax.hist(start_positions[:, ax_i], bins=50, alpha=0.6, color='green', label='Start', edgecolor='none')
        ax.hist(end_positions[:, ax_i], bins=50, alpha=0.6, color='red', label='End', edgecolor='none')
        ax.set_xlabel(f'{ax_name} (mm)')
        ax.set_ylabel('Count')
        ax.set_title(f'{ax_name} Position')
        ax.legend()

    # Start position 2D scatter
    pairs = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    for i, (n1, n2, i1, i2) in enumerate(pairs):
        ax = axes[1, i]
        ax.scatter(start_positions[:, i1], start_positions[:, i2], alpha=0.2, s=3, c='green', label='Start')
        ax.scatter(end_positions[:, i1], end_positions[:, i2], alpha=0.2, s=3, c='red', label='End')
        ax.set_xlabel(f'{n1} (mm)')
        ax.set_ylabel(f'{n2} (mm)')
        ax.set_title(f'{n1}{n2}')
        ax.legend(markerscale=5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'analysis_positions.png', dpi=200, bbox_inches='tight')
    plt.close()
    p(f"Saved: {out_dir / 'analysis_positions.png'}")

    # Figure 4: Trajectory length & final distance
    fig, axes = plt.subplots(1, 2 if has_final_dist else 1, figsize=(12 if has_final_dist else 6, 5))
    if not has_final_dist:
        axes = [axes]
    axes[0].hist(traj_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Trajectory Length (steps)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Trajectory Length Distribution (mean={traj_lengths.mean():.1f})')
    if has_final_dist:
        axes[1].hist(final_dists, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(3.0, color='green', linestyle='--', label='3mm threshold')
        axes[1].set_xlabel('Final Distance (mm)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Final Needle-Trocar Distance (mean={final_dists.mean():.2f}mm)')
        axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'analysis_traj_quality.png', dpi=200, bbox_inches='tight')
    plt.close()
    p(f"Saved: {out_dir / 'analysis_traj_quality.png'}")

    # Save report
    report_path = out_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    p(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize and analyze trajectories')
    parser.add_argument('--dataset_path', type=str,
                       default='/data/public/NAS/VLANeXt/dataset/fine_align/uniform_new/collected_data_merged',
                       help='Path to dataset')
    parser.add_argument('--output', type=str, default='all_trajectories_3d.png',
                       help='Output image path')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to visualize (for testing)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run dataset analysis (perturbation, action, position distributions)')
    parser.add_argument('--analyze_output', type=str, default=None,
                       help='Output directory for analysis results (default: dataset_path)')

    args = parser.parse_args()

    if args.analyze:
        analyze_dataset(args.dataset_path, args.analyze_output)
    else:
        visualize_all_trajectories(args.dataset_path, args.output, args.max_trajectories)

