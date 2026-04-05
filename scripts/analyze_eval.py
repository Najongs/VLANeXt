"""
Analyze eval results from sim_eval_align_only.

Usage:
    python scripts/analyze_eval.py /path/to/eval_dir/metrics_summary.csv

Outputs:
  1. Overall stats (SR, dist distribution)
  2. Success vs Fail comparison
  3. Failure mode classification
  4. Near-miss analysis
  5. Single-axis perturbation direction analysis
  6. Combined XY/XZ quadrant analysis
  7. Auto-generated data collection commands for weak directions
  8. Scatter plots
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    has_perturb = "perturb_x_mm" in df.columns
    return df, has_perturb


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def overall_stats(df):
    print_section("Overall Statistics")
    n = len(df)
    n_success = df["success"].sum()
    sr = n_success / n * 100

    print(f"  Episodes: {n}")
    print(f"  Success Rate: {sr:.1f}% ({n_success}/{n})")
    print(f"  Final dist  — mean: {df['final_dist_mm'].mean():.2f}mm, "
          f"median: {df['final_dist_mm'].median():.2f}mm, "
          f"std: {df['final_dist_mm'].std():.2f}mm")
    print(f"  Min dist    — mean: {df['min_dist_mm'].mean():.2f}mm, "
          f"median: {df['min_dist_mm'].median():.2f}mm")
    print(f"  Lateral     — mean: {df['final_lateral_mm'].mean():.2f}mm")
    print(f"  Angle       — mean: {df['final_angle_deg'].mean():.2f}deg")

    succ = df[df["success"] == 1]
    if len(succ) > 0:
        print(f"  Success steps — mean: {succ['steps'].mean():.0f}, "
              f"median: {succ['steps'].median():.0f}, "
              f"range: [{succ['steps'].min()}, {succ['steps'].max()}]")


def success_vs_fail(df):
    print_section("Success vs Fail Comparison")
    succ = df[df["success"] == 1]
    fail = df[df["success"] == 0]

    if len(succ) == 0 or len(fail) == 0:
        print("  Not enough data for comparison.")
        return

    for col, label in [("final_dist_mm", "Final dist (mm)"),
                        ("min_dist_mm", "Min dist (mm)"),
                        ("final_lateral_mm", "Lateral (mm)"),
                        ("final_angle_deg", "Angle (deg)")]:
        print(f"  {label:20s} | Success: {succ[col].mean():6.2f} +- {succ[col].std():5.2f} | "
              f"Fail: {fail[col].mean():6.2f} +- {fail[col].std():5.2f}")


def near_miss_analysis(df, threshold_mm=2.0):
    print_section(f"Near-Miss Analysis (min_dist < {threshold_mm + 1}mm but failed)")
    fail = df[df["success"] == 0]
    near = fail[fail["min_dist_mm"] < threshold_mm + 1.0]

    if len(near) == 0:
        print("  No near-misses found.")
        return

    print(f"  Near-miss episodes: {len(near)} / {len(fail)} failures "
          f"({len(near)/len(fail)*100:.0f}%)")
    print(f"  These episodes got close but couldn't hold or overshot.")
    print(f"\n  {'ep':>4s} {'min_dist':>8s} {'final_dist':>10s} {'lateral':>8s} {'angle':>6s}")
    print(f"  {'-'*40}")
    for _, r in near.sort_values("min_dist_mm").iterrows():
        print(f"  {int(r['episode']):4d} {r['min_dist_mm']:8.2f} {r['final_dist_mm']:10.2f} "
              f"{r['final_lateral_mm']:8.2f} {r['final_angle_deg']:6.2f}")


def _axis_sr(df, col, lo, hi):
    """SR for subset where df[col] is in [lo, hi)."""
    subset = df[(df[col] >= lo) & (df[col] < hi)]
    if len(subset) == 0:
        return None, 0, 0
    return subset["success"].mean() * 100, int(subset["success"].sum()), len(subset)


def perturbation_analysis(df):
    print_section("Single-Axis Perturbation Analysis")

    # By perturbation magnitude
    print("\n  --- By perturbation distance ---")
    bins = [0, 5, 8, 11, 15, 999]
    labels = ["0-5mm", "5-8mm", "8-11mm", "11-15mm", "15mm+"]
    df["dist_bin"] = pd.cut(df["perturb_dist_mm"], bins=bins, labels=labels)
    for label in labels:
        subset = df[df["dist_bin"] == label]
        if len(subset) == 0:
            continue
        sr = subset["success"].mean() * 100
        print(f"  {label:>8s}: SR={sr:5.1f}% ({subset['success'].sum()}/{len(subset)}) "
              f"avg_min_dist={subset['min_dist_mm'].mean():.2f}mm")

    # Single axis analysis
    for axis_name, col, limit in [("X", "perturb_x_mm", 2),
                                   ("Y", "perturb_y_mm", 2),
                                   ("Z", "perturb_z_mm", 2)]:
        print(f"\n  --- By {axis_name} perturbation direction ---")
        for name, lo, hi in [(f"{axis_name}<-{limit}", -999, -limit),
                              ("Center", -limit, limit),
                              (f"{axis_name}>+{limit}", limit, 999)]:
            subset = df[(df[col] >= lo) & (df[col] < hi)]
            if len(subset) == 0:
                continue
            sr = subset["success"].mean() * 100
            print(f"  {name:>15s}: SR={sr:5.1f}% ({subset['success'].sum()}/{len(subset)}) "
                  f"avg_min_dist={subset['min_dist_mm'].mean():.2f}mm")

    # By angle magnitude
    print("\n  --- By perturbation angle ---")
    for name, lo, hi in [("<3 deg", 0, 3), ("3-5 deg", 3, 5), ("5+ deg", 5, 999)]:
        subset = df[(df["perturb_angle_deg"].abs() >= lo) & (df["perturb_angle_deg"].abs() < hi)]
        if len(subset) == 0:
            continue
        sr = subset["success"].mean() * 100
        print(f"  {name:>15s}: SR={sr:5.1f}% ({subset['success'].sum()}/{len(subset)}) "
              f"avg_min_dist={subset['min_dist_mm'].mean():.2f}mm")


def quadrant_analysis(df):
    """Combined multi-axis direction analysis."""
    print_section("Combined Direction Analysis (Quadrants)")

    overall_sr = df["success"].mean() * 100

    # XY quadrants
    print(f"\n  --- XY Quadrants (overall SR={overall_sr:.1f}%) ---")
    print(f"  {'Quadrant':>20s} {'SR':>8s} {'N':>6s} {'min_dist':>10s} {'gap':>8s}")
    print(f"  {'-'*56}")

    quadrant_results = []
    for x_name, x_lo, x_hi in [("X-", -999, 0), ("X+", 0, 999)]:
        for y_name, y_lo, y_hi in [("Y-", -999, 0), ("Y+", 0, 999)]:
            subset = df[(df["perturb_x_mm"] >= x_lo) & (df["perturb_x_mm"] < x_hi) &
                        (df["perturb_y_mm"] >= y_lo) & (df["perturb_y_mm"] < y_hi)]
            if len(subset) == 0:
                continue
            sr = subset["success"].mean() * 100
            avg_min = subset["min_dist_mm"].mean()
            gap = sr - overall_sr
            label = f"{x_name},{y_name}"
            sign = "+" if gap >= 0 else ""
            print(f"  {label:>20s} {sr:7.1f}% {len(subset):6d} {avg_min:9.2f}mm {sign}{gap:7.1f}%")
            quadrant_results.append({
                "label": label, "sr": sr, "n": len(subset),
                "avg_min_dist": avg_min, "gap": gap,
                "x_sign": x_name, "y_sign": y_name,
            })

    # XZ quadrants
    print(f"\n  --- XZ Quadrants ---")
    print(f"  {'Quadrant':>20s} {'SR':>8s} {'N':>6s} {'min_dist':>10s} {'gap':>8s}")
    print(f"  {'-'*56}")

    for x_name, x_lo, x_hi in [("X-", -999, 0), ("X+", 0, 999)]:
        for z_name, z_lo, z_hi in [("Z-", -999, 0), ("Z+", 0, 999)]:
            subset = df[(df["perturb_x_mm"] >= x_lo) & (df["perturb_x_mm"] < x_hi) &
                        (df["perturb_z_mm"] >= z_lo) & (df["perturb_z_mm"] < z_hi)]
            if len(subset) == 0:
                continue
            sr = subset["success"].mean() * 100
            avg_min = subset["min_dist_mm"].mean()
            gap = sr - overall_sr
            label = f"{x_name},{z_name}"
            sign = "+" if gap >= 0 else ""
            print(f"  {label:>20s} {sr:7.1f}% {len(subset):6d} {avg_min:9.2f}mm {sign}{gap:7.1f}%")

    # YZ quadrants
    print(f"\n  --- YZ Quadrants ---")
    print(f"  {'Quadrant':>20s} {'SR':>8s} {'N':>6s} {'min_dist':>10s} {'gap':>8s}")
    print(f"  {'-'*56}")

    for y_name, y_lo, y_hi in [("Y-", -999, 0), ("Y+", 0, 999)]:
        for z_name, z_lo, z_hi in [("Z-", -999, 0), ("Z+", 0, 999)]:
            subset = df[(df["perturb_y_mm"] >= y_lo) & (df["perturb_y_mm"] < y_hi) &
                        (df["perturb_z_mm"] >= z_lo) & (df["perturb_z_mm"] < z_hi)]
            if len(subset) == 0:
                continue
            sr = subset["success"].mean() * 100
            avg_min = subset["min_dist_mm"].mean()
            gap = sr - overall_sr
            label = f"{y_name},{z_name}"
            sign = "+" if gap >= 0 else ""
            print(f"  {label:>20s} {sr:7.1f}% {len(subset):6d} {avg_min:9.2f}mm {sign}{gap:7.1f}%")

    # XYZ octants
    print(f"\n  --- XYZ Octants (3-axis combined) ---")
    print(f"  {'Octant':>20s} {'SR':>8s} {'N':>6s} {'min_dist':>10s} {'gap':>8s}")
    print(f"  {'-'*56}")

    octant_results = []
    for x_name, x_lo, x_hi in [("X-", -999, 0), ("X+", 0, 999)]:
        for y_name, y_lo, y_hi in [("Y-", -999, 0), ("Y+", 0, 999)]:
            for z_name, z_lo, z_hi in [("Z-", -999, 0), ("Z+", 0, 999)]:
                subset = df[(df["perturb_x_mm"] >= x_lo) & (df["perturb_x_mm"] < x_hi) &
                            (df["perturb_y_mm"] >= y_lo) & (df["perturb_y_mm"] < y_hi) &
                            (df["perturb_z_mm"] >= z_lo) & (df["perturb_z_mm"] < z_hi)]
                if len(subset) < 2:
                    continue
                sr = subset["success"].mean() * 100
                avg_min = subset["min_dist_mm"].mean()
                gap = sr - overall_sr
                label = f"{x_name},{y_name},{z_name}"
                sign = "+" if gap >= 0 else ""
                print(f"  {label:>20s} {sr:7.1f}% {len(subset):6d} {avg_min:9.2f}mm {sign}{gap:7.1f}%")
                octant_results.append({
                    "label": label, "sr": sr, "n": len(subset),
                    "avg_min_dist": avg_min, "gap": gap,
                })

    return octant_results


def generate_collection_commands(df, octant_results, out_dir, sr_threshold=None):
    """Auto-generate data collection commands for weak directions."""
    print_section("Recommended Data Collection Commands")

    overall_sr = df["success"].mean() * 100
    if sr_threshold is None:
        sr_threshold = overall_sr * 0.8  # below 80% of overall SR

    # Find weak octants
    weak = [o for o in octant_results if o["sr"] < sr_threshold and o["n"] >= 2]
    weak.sort(key=lambda x: x["sr"])

    if not weak:
        print(f"  No weak directions found (threshold: SR < {sr_threshold:.1f}%)")
        return

    print(f"  Weak direction threshold: SR < {sr_threshold:.1f}% (80% of overall {overall_sr:.1f}%)")
    print(f"  Found {len(weak)} weak octants:\n")

    # Parse octant labels into bias configs
    bias_configs = []
    for o in weak:
        parts = o["label"].split(",")
        axes = {}
        for p in parts:
            axis = p[0].lower()  # x, y, z
            sign = "neg" if p[1] == "-" else "pos"
            axes[axis] = sign
        bias_configs.append({
            "label": o["label"],
            "sr": o["sr"],
            "n": o["n"],
            "axes": axes,
        })

    # Generate commands
    commands = []
    for i, bc in enumerate(bias_configs):
        axes_str = ",".join(f"{a}_{s}" for a, s in bc["axes"].items())
        dir_name = "_".join(f"{a}{s[0]}" for a, s in bc["axes"].items())
        episodes = max(500, int(1000 * (1 - bc["sr"] / 100)))  # more episodes for lower SR

        cmd = (f"python Sim/run_parallel.py \\\n"
               f"    --script align --workers 10 --episodes {episodes} \\\n"
               f"    --base-dir dataset/fine_align/bias_{dir_name} \\\n"
               f"    --bias {axes_str}")

        print(f"  [{i+1}] {bc['label']} (SR={bc['sr']:.1f}%, n={bc['n']})")
        print(f"      {cmd}")
        print()
        commands.append(cmd)

    # Save as shell script
    sh_path = out_dir / "collect_weak_directions.sh"
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated data collection for weak directions\n")
        f.write(f"# Based on eval analysis (overall SR={overall_sr:.1f}%)\n")
        f.write(f"# Threshold: SR < {sr_threshold:.1f}%\n\n")
        for i, (cmd, bc) in enumerate(zip(commands, bias_configs)):
            f.write(f'echo "=== [{i+1}/{len(commands)}] {bc["label"]} (SR={bc["sr"]:.1f}%) ==="\n')
            f.write(cmd + "\n\n")
    sh_path.chmod(0o755)
    print(f"  Shell script saved: {sh_path}")


def failure_mode_classification(df):
    print_section("Failure Mode Classification")
    fail = df[df["success"] == 0].copy()
    if len(fail) == 0:
        print("  No failures!")
        return

    modes = []
    for _, r in fail.iterrows():
        min_d = r["min_dist_mm"]
        final_d = r["final_dist_mm"]
        final_ang = r["final_angle_deg"]

        if min_d < 3.0 and final_d > min_d + 2.0:
            modes.append("overshoot")
        elif min_d < 3.0 and final_d - min_d < 2.0:
            modes.append("oscillate")
        elif final_d > 8.0:
            modes.append("diverge")
        elif final_ang > 4.0:
            modes.append("angle_fail")
        else:
            modes.append("insufficient")

    fail["failure_mode"] = modes

    print(f"\n  Total failures: {len(fail)}")
    for mode in ["overshoot", "oscillate", "diverge", "angle_fail", "insufficient"]:
        subset = fail[fail["failure_mode"] == mode]
        if len(subset) == 0:
            continue
        pct = len(subset) / len(fail) * 100
        print(f"  {mode:>15s}: {len(subset):3d} ({pct:4.1f}%) — "
              f"avg final_dist={subset['final_dist_mm'].mean():.1f}mm, "
              f"avg min_dist={subset['min_dist_mm'].mean():.1f}mm")

    print(f"\n  --- Worst 5 episodes (by min_dist) ---")
    worst = fail.nlargest(5, "min_dist_mm")
    for _, r in worst.iterrows():
        mode = r.get("failure_mode", "?")
        extra = ""
        if "perturb_x_mm" in r:
            extra = (f" perturb=({r['perturb_x_mm']:+.1f}, "
                     f"{r['perturb_y_mm']:+.1f}, {r['perturb_z_mm']:+.1f})")
        print(f"  ep{int(r['episode']):04d}: min_dist={r['min_dist_mm']:.1f}mm "
              f"final={r['final_dist_mm']:.1f}mm mode={mode}{extra}")


def plot_analysis(df, has_perturb, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distance over episodes
    ax = axes[0, 0]
    colors = ["green" if s else "red" for s in df["success"]]
    ax.bar(df["episode"], df["min_dist_mm"], color=colors, alpha=0.7, label="min_dist")
    ax.axhline(y=2.0, color="blue", linestyle="--", linewidth=1, label="threshold (2mm)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Min Distance (mm)")
    ax.set_title("Min Distance per Episode (green=success, red=fail)")
    ax.legend()

    # 2. Final dist vs min dist (overshoot detection)
    ax = axes[0, 1]
    succ = df[df["success"] == 1]
    fail = df[df["success"] == 0]
    if len(fail) > 0:
        ax.scatter(fail["min_dist_mm"], fail["final_dist_mm"], c="red", alpha=0.6, label="fail", s=40)
    if len(succ) > 0:
        ax.scatter(succ["min_dist_mm"], succ["final_dist_mm"], c="green", alpha=0.6, label="success", s=40)
    lim = max(df["final_dist_mm"].max(), df["min_dist_mm"].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="final=min (no overshoot)")
    ax.set_xlabel("Min Distance (mm)")
    ax.set_ylabel("Final Distance (mm)")
    ax.set_title("Overshoot Detection (above diagonal = overshoot)")
    ax.legend()

    # 3. Distribution of min distances
    ax = axes[1, 0]
    ax.hist(df["min_dist_mm"], bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="threshold (2mm)")
    ax.set_xlabel("Min Distance (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Min Distances")
    ax.legend()

    # 4. Perturbation scatter
    ax = axes[1, 1]
    if has_perturb:
        ax.scatter(df["perturb_x_mm"], df["perturb_y_mm"],
                   c=df["success"], cmap="RdYlGn", vmin=0, vmax=1,
                   s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("Perturbation X (mm)")
        ax.set_ylabel("Perturbation Y (mm)")
        ax.set_title("Perturbation XY: Success (green) / Fail (red)")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        # Quadrant labels
        for (x, y, label) in [(-5, 5, "X-Y+"), (5, 5, "X+Y+"), (-5, -5, "X-Y-"), (5, -5, "X+Y-")]:
            ax.text(x, y, label, fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    else:
        ax.scatter(df["final_lateral_mm"], df["final_angle_deg"],
                   c=["green" if s else "red" for s in df["success"]],
                   alpha=0.6, s=40)
        ax.set_xlabel("Final Lateral (mm)")
        ax.set_ylabel("Final Angle (deg)")
        ax.set_title("Lateral vs Angle at Termination")

    plt.tight_layout()
    plot_path = out_dir / "eval_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Extra plots
    if has_perturb:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        # Perturbation dist → min dist
        ax = axes2[0]
        if len(fail) > 0:
            ax.scatter(fail["perturb_dist_mm"], fail["min_dist_mm"], c="red", alpha=0.5, label="fail", s=30)
        if len(succ) > 0:
            ax.scatter(succ["perturb_dist_mm"], succ["min_dist_mm"], c="green", alpha=0.5, label="success", s=30)
        ax.axhline(y=2.0, color="blue", linestyle="--", linewidth=1)
        ax.set_xlabel("Perturbation Distance (mm)")
        ax.set_ylabel("Min Distance Achieved (mm)")
        ax.set_title("Perturbation Magnitude vs Best Distance")
        ax.legend()

        # Perturbation XZ
        ax = axes2[1]
        ax.scatter(df["perturb_x_mm"], df["perturb_z_mm"],
                   c=df["success"], cmap="RdYlGn", vmin=0, vmax=1,
                   s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("Perturbation X (mm)")
        ax.set_ylabel("Perturbation Z (mm)")
        ax.set_title("Perturbation XZ: Success/Fail")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)

        # Perturbation YZ
        ax = axes2[2]
        ax.scatter(df["perturb_y_mm"], df["perturb_z_mm"],
                   c=df["success"], cmap="RdYlGn", vmin=0, vmax=1,
                   s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("Perturbation Y (mm)")
        ax.set_ylabel("Perturbation Z (mm)")
        ax.set_title("Perturbation YZ: Success/Fail")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)

        plt.tight_layout()
        plot_path2 = out_dir / "eval_analysis_perturbation.png"
        plt.savefig(plot_path2, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path2}")


def main():
    parser = argparse.ArgumentParser(description="Analyze eval results")
    parser.add_argument("csv_path", type=str, help="Path to metrics_summary.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df, has_perturb = load_csv(csv_path)
    out_dir = csv_path.parent

    overall_stats(df)
    success_vs_fail(df)
    failure_mode_classification(df)
    near_miss_analysis(df)

    octant_results = []
    if has_perturb:
        perturbation_analysis(df)
        octant_results = quadrant_analysis(df)
        generate_collection_commands(df, octant_results, out_dir)
    else:
        print(f"\n  [Note] No perturbation columns in CSV. "
              f"Re-run eval with updated sim_eval_align_only.py for full analysis.")

    plot_analysis(df, has_perturb, out_dir)

    print(f"\n{'='*60}")
    print(f"  Analysis complete. Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
