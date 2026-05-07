"""
Merge parallel eval shard results into a single CSV + run analysis.

Usage:
    python scripts/merge_eval_shards.py /path/to/checkpoint --num-shards 3
"""

import argparse
import glob
import shutil
import pandas as pd
from pathlib import Path


def _collect_shard_files(shard_dirs, merged_dir):
    """Copy mp4, npz, png files from shard directories into merged directory.
    Preserves subdirectory structure (e.g., direction_name/ for basic eval)."""
    count = 0
    for shard_dir in shard_dirs:
        for pattern in ("**/*.mp4", "**/*.npz", "**/*.png"):
            for f in shard_dir.glob(pattern):
                # Preserve relative path from shard dir
                rel = f.relative_to(shard_dir)
                dst = merged_dir / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    count += 1
    print(f"  Collected {count} files (mp4/npz/png) into merged directory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--exec-steps", type=int, default=1)
    parser.add_argument("--diff-steps", type=int, default=10)
    parser.add_argument("--prefix", type=str, default="align", help="Eval type prefix (align or approach)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    step_str = ckpt_path.stem.split("_")[-1]
    base_name = f"{args.prefix}_eval_step{step_str}_exec{args.exec_steps}_diff{args.diff_steps}"

    # Find shard directories (handles _SR rename)
    shard_csvs = []
    shard_dirs = []
    for shard_id in range(args.num_shards):
        shard_pattern = f"{base_name}_shard{shard_id}"
        # Try exact match first, then glob for _SR suffix
        shard_dir = ckpt_path.parent / shard_pattern
        if not shard_dir.exists():
            matches = list(ckpt_path.parent.glob(f"{shard_pattern}_SR*"))
            if matches:
                shard_dir = matches[0]
        csv_path = shard_dir / "metrics_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            shard_csvs.append(df)
            shard_dirs.append(shard_dir)
            print(f"  Loaded shard {shard_id}: {len(df)} episodes ({shard_dir.name})")
        else:
            print(f"  WARNING: shard {shard_id} CSV not found (looked for {shard_pattern}*)")

    if not shard_csvs:
        print("No shard results found!")
        return

    # Merge and sort
    merged = pd.concat(shard_csvs, ignore_index=True)
    if "direction" in merged.columns:
        merged = merged.sort_values(["direction", "episode"]).reset_index(drop=True)
    else:
        merged = merged.sort_values("episode").reset_index(drop=True)

    # Save merged CSV
    merged_dir = ckpt_path.parent / base_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = merged_dir / "metrics_summary.csv"
    merged.to_csv(merged_csv, index=False)

    # Collect all shard files (mp4, npz, png) into merged directory
    _collect_shard_files(shard_dirs, merged_dir)

    n_success = merged["success"].sum()
    n_total = len(merged)
    sr = n_success / n_total * 100

    print(f"\n{'='*60}")
    print(f"  Merged: {n_total} episodes from {len(shard_csvs)} shards")
    print(f"  Success Rate: {sr:.1f}% ({n_success}/{n_total})")
    print(f"  Saved to: {merged_csv}")
    print(f"{'='*60}")

    # Generate merged trajectory plot from all npz files
    _generate_merged_trajectory_plot(merged_dir)

    # Auto-run analysis (skip for basic eval — different CSV format)
    if args.prefix == "basic":
        print(f"\nGenerating basic motion summary...")
        _generate_basic_summary(merged_dir, merged)
    else:
        print(f"\nRunning analysis...")
        import subprocess
        result = subprocess.run(
            ["python", "scripts/analyze_eval.py", str(merged_csv)],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Persist final summary (SR + analysis) to txt for later reference.
        summary_txt = merged_dir / "eval_summary.txt"
        with open(summary_txt, "w") as f:
            f.write(f"Checkpoint: {ckpt_path}\n")
            f.write(f"Merged episodes: {n_total} from {len(shard_csvs)} shards\n")
            f.write(f"Success Rate: {sr:.1f}% ({n_success}/{n_total})\n")
            f.write(f"Merged CSV: {merged_csv}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("  analyze_eval.py output\n")
            f.write("=" * 60 + "\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n[stderr]\n" + result.stderr)
        print(f"  Eval summary saved: {summary_txt}")


def _generate_merged_trajectory_plot(merged_dir):
    """Regenerate trajectory plot from all npz files in merged directory."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    npz_files = sorted(merged_dir.glob("traj_ep*.npz"))
    if not npz_files:
        return

    trajectories = []
    successes = []
    for f in npz_files:
        data = np.load(f)
        trajectories.append(data["ee_pose"])
        successes.append("_S." in f.name)

    fig = plt.figure(figsize=(20, 12))
    n_succ = sum(successes)
    n_fail = len(successes) - n_succ
    fig.suptitle(f"Eval Trajectories (n={len(trajectories)}, S={n_succ}, F={n_fail})", fontsize=14)

    ax1 = fig.add_subplot(221, projection='3d')
    for i, traj in enumerate(trajectories):
        color = 'green' if successes[i] else 'red'
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5, linewidth=0.8, color=color)
        ax1.scatter(*traj[0], marker='o', s=30, color=color, alpha=0.6)
        ax1.scatter(*traj[-1], marker='x', s=30, color=color, alpha=0.8)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D View'); ax1.grid(True)

    for subplot, ax_a, ax_b, xlabel, ylabel, title in [
        (222, 0, 1, 'X (mm)', 'Y (mm)', 'XY (Top View)'),
        (223, 0, 2, 'X (mm)', 'Z (mm)', 'XZ (Front View)'),
        (224, 1, 2, 'Y (mm)', 'Z (mm)', 'YZ (Side View)'),
    ]:
        ax = fig.add_subplot(subplot)
        for i, traj in enumerate(trajectories):
            color = 'green' if successes[i] else 'red'
            ax.plot(traj[:, ax_a], traj[:, ax_b], alpha=0.5, linewidth=0.8, color=color)
            ax.scatter(traj[0, ax_a], traj[0, ax_b], marker='o', s=20, color=color, alpha=0.6)
            ax.scatter(traj[-1, ax_a], traj[-1, ax_b], marker='x', s=20, color=color, alpha=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True); ax.axis('equal')

    legend_elements = [
        Line2D([0], [0], color='green', label='Success'),
        Line2D([0], [0], color='red', label='Fail'),
        Line2D([0], [0], marker='o', color='gray', label='Start', linestyle='None', markersize=6),
        Line2D([0], [0], marker='x', color='gray', label='End', linestyle='None', markersize=6),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = merged_dir / "eval_trajectories.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Merged trajectory plot saved: {out_path}")


def _generate_basic_summary(merged_dir, df):
    """Generate bar chart summary for basic motion eval."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directions = sorted(df["direction"].unique())
    success_rates = []
    avg_cosines = []
    avg_displacements = []

    for d in directions:
        g = df[df["direction"] == d]
        success_rates.append(g["success"].mean() * 100)
        avg_cosines.append(g["cosine_sim"].astype(float).mean())
        avg_displacements.append(g["displacement_mm"].astype(float).mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    overall_sr = df["success"].mean() * 100
    fig.suptitle(f"Basic Motion Eval Summary (Overall SR: {overall_sr:.1f}%)", fontsize=14)

    x = np.arange(len(directions))

    colors_sr = ['green' if s >= 50 else 'orange' if s >= 25 else 'red' for s in success_rates]
    axes[0].bar(x, success_rates, color=colors_sr)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(directions, rotation=45, ha='right')
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate per Direction")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=8)

    colors_cos = ['green' if c >= 0.5 else 'orange' if c >= 0.0 else 'red' for c in avg_cosines]
    axes[1].bar(x, avg_cosines, color=colors_cos)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(directions, rotation=45, ha='right')
    axes[1].set_ylabel("Avg Cosine Similarity")
    axes[1].set_title("Direction Accuracy")
    axes[1].set_ylim(-1, 1.1)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    axes[2].bar(x, avg_displacements, color='steelblue')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(directions, rotation=45, ha='right')
    axes[2].set_ylabel("Avg Displacement (mm)")
    axes[2].set_title("Movement Magnitude")

    plt.tight_layout()
    out_path = merged_dir / "eval_analysis.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot saved: {out_path}")


if __name__ == "__main__":
    main()
