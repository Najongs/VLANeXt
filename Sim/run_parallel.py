#!/usr/bin/env python3
"""
Unified Parallel Data Collection

Supports both full pipeline (Save_dataset.py) and align-only (Save_dataset_align_only.py).

Usage:
    python Sim/run_parallel.py --script align --workers 10 --episodes 1000 \
        --base-dir dataset/fine_align/bias_x_neg --bias x_neg

    python Sim/run_parallel.py --script full --workers 5 --episodes 500 \
        --base-dir dataset/new_data

    python run_parallel.py --grid --grid-bins-xy 8 --grid-bins-z 6 --workers 10 --base-dir dataset/grid_data
"""
import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
import shutil
import numpy as np


SCRIPT_MAP = {
    "align": "Save_dataset_align_only",
    "full": "Save_dataset",
}


def main():
    parser = argparse.ArgumentParser(description="Parallel data collection")
    parser.add_argument("--script", type=str, default="align",
                        choices=list(SCRIPT_MAP.keys()),
                        help="Which collection script to run")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Episodes per worker")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--bias", type=str, default=None,
                        help="(align only) Bias direction(s). Single: 'x_neg', Combined: 'x_neg,y_neg'")
    parser.add_argument("--bias-ratio", type=float, default=0.8,
                        help="(align only) Fraction of biased perturbations")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging worker directories")
    # Grid mode
    parser.add_argument("--grid", action="store_true",
                        help="(align only) Use grid-based stratified sampling")
    parser.add_argument("--grid-bins-xy", type=int, default=8,
                        help="Grid bins per XY axis (default: 8)")
    parser.add_argument("--grid-bins-z", type=int, default=6,
                        help="Grid bins for Z axis (default: 6)")
    args = parser.parse_args()

    module_name = SCRIPT_MAP[args.script]
    sim_dir = Path(__file__).parent.absolute()
    base_path = Path(args.base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # --- Grid mode: 셀 생성 및 워커 분배 ---
    grid_worker_files = {}  # worker_id -> json file path
    if args.grid and args.script == "align":
        xy_mm = 30.0   # PERTURB_POS_XY_MM
        z_mm = 20.0    # PERTURB_POS_Z_MM
        bx, by, bz = args.grid_bins_xy, args.grid_bins_xy, args.grid_bins_z

        # 셀 경계 생성
        x_edges = np.linspace(-xy_mm, xy_mm, bx + 1)
        y_edges = np.linspace(-xy_mm, xy_mm, by + 1)
        z_edges = np.linspace(-z_mm, z_mm, bz + 1)

        # 전체 셀 목록: [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]
        all_cells = []
        for xi in range(bx):
            for yi in range(by):
                for zi in range(bz):
                    all_cells.append([
                        float(x_edges[xi]), float(x_edges[xi+1]),
                        float(y_edges[yi]), float(y_edges[yi+1]),
                        float(z_edges[zi]), float(z_edges[zi+1]),
                    ])

        # 셔플
        np.random.seed(42)
        np.random.shuffle(all_cells)

        total_cells = len(all_cells)
        # 워커별 균등 분배
        cells_per_worker = total_cells // args.workers
        remainder = total_cells % args.workers

        offset = 0
        for i in range(args.workers):
            n = cells_per_worker + (1 if i < remainder else 0)
            worker_cells = all_cells[offset:offset + n]
            offset += n

            json_path = base_path / f"_grid_cells_worker_{i}.json"
            with open(json_path, 'w') as f:
                json.dump(worker_cells, f)
            grid_worker_files[i] = json_path

        total = total_cells
        print("=" * 70)
        print(f"Parallel Data Collection ({args.script}) — GRID MODE")
        print("=" * 70)
        print(f"  Script:   {module_name}.py")
        print(f"  Grid:     {bx} x {by} x {bz} = {total_cells} cells")
        print(f"  Workers:  {args.workers}")
        print(f"  Output:   {base_path}")
        print("=" * 70)
        print()
    else:
        total = args.workers * args.episodes
        print("=" * 70)
        print(f"Parallel Data Collection ({args.script})")
        print("=" * 70)
        print(f"  Script:   {module_name}.py")
        print(f"  Workers:  {args.workers}")
        print(f"  Episodes: {args.episodes} per worker ({total} total)")
        print(f"  Output:   {base_path}")
        if args.bias:
            print(f"  Bias:     {args.bias} (ratio={args.bias_ratio})")
        print("=" * 70)
        print()

    # Build bias override lines for align-only script
    bias_lines = ""
    if args.bias and args.script == "align" and not args.grid:
        bias_lines = f"""
{module_name}.BIAS_DIRECTION = '{args.bias}'
{module_name}.BIAS_RATIO = {args.bias_ratio}
"""

    # Launch workers
    processes = []
    for i in range(args.workers):
        worker_dir = base_path / f"worker_{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Grid mode: 워커별 셀 파일 경로로 스크립트 생성
        if i in grid_worker_files:
            grid_json = grid_worker_files[i]
            n_cells = len(json.load(open(grid_json)))
            script_content = f"""
import sys, os, time, json
os.chdir('{sim_dir}')
sys.path.insert(0, '{sim_dir}')

import {module_name}

{module_name}.SAVE_DIR = r'{worker_dir}'
with open(r'{grid_json}', 'r') as _f:
    {module_name}.GRID_CELLS = json.load(_f)
{module_name}.MAX_EPISODES = len({module_name}.GRID_CELLS)

if __name__ == "__main__":
    print(f"[Worker {i}] Starting: {n_cells} grid cells -> {worker_dir}")
    {module_name}.main()
    time.sleep(2)
    print(f"[Worker {i}] Done!")
"""
        else:
            script_content = f"""
import sys, os, time
os.chdir('{sim_dir}')
sys.path.insert(0, '{sim_dir}')

import {module_name}

{module_name}.SAVE_DIR = r'{worker_dir}'
{module_name}.MAX_EPISODES = {args.episodes}
{bias_lines}

if __name__ == "__main__":
    print(f"[Worker {i}] Starting: {args.episodes} episodes -> {worker_dir}")
    {module_name}.main()
    time.sleep(2)
    print(f"[Worker {i}] Done!")
"""

        script_path = base_path / f"_temp_worker_{i}.py"
        script_path.write_text(script_content)

        log_file = open(base_path / f"worker_{i}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=sim_dir,
        )
        processes.append((proc, log_file, script_path))
        print(f"  [Worker {i}] Started (PID: {proc.pid})")
        time.sleep(0.5)

    print()
    print(f"All {args.workers} workers launched.")
    print(f"Monitor: tail -f {base_path}/worker_*.log")
    print("Waiting for completion...")
    print()

    # Wait
    start_time = time.time()
    for i, (proc, log_file, script_path) in enumerate(processes):
        proc.wait()
        log_file.close()

        worker_dir = base_path / f"worker_{i}"
        h5_count = len(list(worker_dir.glob("*.h5"))) if worker_dir.exists() else 0
        status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
        print(f"  [Worker {i}] {status} — {h5_count} episodes")

        script_path.unlink(missing_ok=True)
        # Clean up grid cell JSON
        if i in grid_worker_files:
            grid_worker_files[i].unlink(missing_ok=True)

    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)

    # Merge
    if not args.no_merge:
        print()
        print("Merging worker data...")

        final_dir = base_path / "collected_data_merged"
        final_dir.mkdir(parents=True, exist_ok=True)

        total_episodes = 0
        for i in range(args.workers):
            worker_dir = base_path / f"worker_{i}"
            if not worker_dir.exists():
                continue
            h5_files = list(worker_dir.glob("*.h5"))
            if h5_files:
                for h5_file in h5_files:
                    new_name = f"w{i}_{h5_file.name}"
                    shutil.move(str(h5_file), str(final_dir / new_name))
                    total_episodes += 1
                try:
                    worker_dir.rmdir()
                except OSError:
                    pass

        print(f"  Merged {total_episodes} episodes -> {final_dir}")
    else:
        total_episodes = sum(
            len(list((base_path / f"worker_{i}").glob("*.h5")))
            for i in range(args.workers)
            if (base_path / f"worker_{i}").exists()
        )

    print()
    print("=" * 70)
    print(f"  Done! {total_episodes} episodes in {minutes}m {seconds}s")
    print(f"  Location: {base_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
