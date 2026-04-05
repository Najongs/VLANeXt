"""
Merge parallel eval shard results into a single CSV + run analysis.

Usage:
    python scripts/merge_eval_shards.py /path/to/checkpoint --num-shards 3
"""

import argparse
import glob
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--exec-steps", type=int, default=1)
    parser.add_argument("--diff-steps", type=int, default=10)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    step_str = ckpt_path.stem.split("_")[-1]
    base_name = f"align_eval_step{step_str}_exec{args.exec_steps}_diff{args.diff_steps}"

    # Find shard directories (handles _SR rename)
    shard_csvs = []
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
            print(f"  Loaded shard {shard_id}: {len(df)} episodes ({shard_dir.name})")
        else:
            print(f"  WARNING: shard {shard_id} CSV not found (looked for {shard_pattern}*)")

    if not shard_csvs:
        print("No shard results found!")
        return

    # Merge and sort by episode number
    merged = pd.concat(shard_csvs, ignore_index=True)
    merged = merged.sort_values("episode").reset_index(drop=True)

    # Save merged CSV
    merged_dir = ckpt_path.parent / base_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = merged_dir / "metrics_summary.csv"
    merged.to_csv(merged_csv, index=False)

    n_success = merged["success"].sum()
    n_total = len(merged)
    sr = n_success / n_total * 100

    print(f"\n{'='*60}")
    print(f"  Merged: {n_total} episodes from {len(shard_csvs)} shards")
    print(f"  Success Rate: {sr:.1f}% ({n_success}/{n_total})")
    print(f"  Saved to: {merged_csv}")
    print(f"{'='*60}")

    # Auto-run analysis
    print(f"\nRunning analysis...")
    import subprocess
    subprocess.run(["python", "scripts/analyze_eval.py", str(merged_csv)])


if __name__ == "__main__":
    main()
