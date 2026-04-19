#!/usr/bin/env python3
"""
Dataset statistics checker — Phase별 step 수 비교

# Phase1만 비교
python scripts/dataset_stats.py dataset/approach_00/collected_data_merged dataset/approach_04/collected_data_merged
--phase 1

# 전체 phase 비교
python scripts/dataset_stats.py dataset/approach_00/collected_data_merged dataset/approach_04/collected_data_merged

# 3개 이상도 가능
python scripts/dataset_stats.py dataset/align_00/... dataset/align_04/... dataset/align_02/...

밸런싱이 필요하면 마지막에 max_episodes 추천값도 나옴.
  
Usage:
    python scripts/dataset_stats.py /path/to/dataset1 /path/to/dataset2 ...
    python scripts/dataset_stats.py dataset/approach_00/collected_data_merged dataset/approach_04/collected_data_merged
    python scripts/dataset_stats.py dataset/approach_00/collected_data_merged dataset/approach_04/collected_data_merged --phase 1
    python scripts/dataset_stats.py dataset/approach_00/collected_data_merged --phase 2
"""

import argparse
import glob
import os
import numpy as np
import h5py


def analyze_dataset(path, phase_filter=None):
    files = sorted(glob.glob(os.path.join(path, "**", "*.h5"), recursive=True))
    if not files:
        return None

    total_steps = []
    phase_steps = {1: [], 2: [], 3: []}

    for f_path in files:
        try:
            with h5py.File(f_path, "r") as f:
                traj_len = f["action"].shape[0]
                total_steps.append(traj_len)
                if "phase" in f:
                    phase = f["phase"][:]
                    for p in [1, 2, 3]:
                        phase_steps[p].append(int(np.sum(phase == p)))
                else:
                    phase_steps[1].append(traj_len)
                    phase_steps[2].append(0)
                    phase_steps[3].append(0)
        except Exception as e:
            print(f"[Warn] Skipping {f_path}: {e}")
            continue

    total_steps = np.array(total_steps)
    for p in phase_steps:
        phase_steps[p] = np.array(phase_steps[p])

    return {
        "path": path,
        "name": os.path.join(
            os.path.basename(os.path.dirname(path.rstrip("/"))),
            os.path.basename(path.rstrip("/"))
        ),
        "episodes": len(files),
        "total_steps": total_steps,
        "phase_steps": phase_steps,
    }


def print_table(datasets, phase_filter=None):
    # Header
    names = [d["name"] for d in datasets]
    col_width = max(20, max(len(n) for n in names) + 2)

    def row(label, values, fmt=None):
        cells = [f"{label:<20}"]
        for v in values:
            if fmt:
                cells.append(f"{fmt(v):>{col_width}}")
            else:
                cells.append(f"{str(v):>{col_width}}")
        print(" | ".join(cells))

    def sep():
        print("-" * (22 + (col_width + 3) * len(datasets)))

    # Title
    phases_to_show = [phase_filter] if phase_filter else [None, 1, 2, 3]

    for phase in phases_to_show:
        if phase is None:
            label = "Total (all phases)"
            get_steps = lambda d: d["total_steps"]
        else:
            label = f"Phase {phase}"
            get_steps = lambda d, p=phase: d["phase_steps"][p]

        # Skip if all zeros
        all_sums = [get_steps(d).sum() for d in datasets]
        if all(s == 0 for s in all_sums):
            continue

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

        sep()
        row("", [d["name"] for d in datasets])
        sep()
        row("Episodes", [d["episodes"] for d in datasets])
        row("Mean steps/ep", [get_steps(d).mean() for d in datasets], lambda v: f"{v:.1f}")
        row("Std steps/ep", [get_steps(d).std() for d in datasets], lambda v: f"{v:.1f}")
        row("Min steps", [get_steps(d).min() for d in datasets])
        row("Max steps", [get_steps(d).max() for d in datasets])
        row("Total samples", [get_steps(d).sum() for d in datasets], lambda v: f"{v:,}")
        sep()

        # Ratio comparison (first dataset as baseline)
        if len(datasets) > 1:
            base_mean = get_steps(datasets[0]).mean()
            base_total = get_steps(datasets[0]).sum()
            if base_mean > 0 and base_total > 0:
                ratios_mean = [get_steps(d).mean() / base_mean for d in datasets]
                ratios_total = [get_steps(d).sum() / base_total for d in datasets]
                row("Ratio (mean)", ratios_mean, lambda v: f"{v:.2f}x")
                row("Ratio (total)", ratios_total, lambda v: f"{v:.2f}x")
                sep()

                # Suggestion
                print(f"\n  Baseline: {datasets[0]['name']}")
                for i, d in enumerate(datasets[1:], 1):
                    r = ratios_total[i]
                    if r > 1.1:
                        suggested = int(d["episodes"] / r)
                        print(f"  {d['name']}: {r:.2f}x more samples → max_episodes: {suggested} to balance")
                    elif r < 0.9:
                        suggested = int(datasets[0]["episodes"] * r)
                        print(f"  {datasets[0]['name']}: reduce to max_episodes: {suggested} to balance with {d['name']}")
                    else:
                        print(f"  {d['name']}: balanced (ratio {r:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics checker")
    parser.add_argument("paths", nargs="+", help="Dataset paths to compare")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3],
                        help="Show only specific phase (1=Align, 2=Insert, 3=Hold)")
    args = parser.parse_args()

    datasets = []
    for p in args.paths:
        print(f"Scanning {p} ...", end=" ", flush=True)
        result = analyze_dataset(p, args.phase)
        if result is None:
            print(f"No .h5 files found!")
            continue
        print(f"{result['episodes']} episodes")
        datasets.append(result)

    if not datasets:
        print("No datasets found.")
        return

    print_table(datasets, args.phase)


if __name__ == "__main__":
    main()
