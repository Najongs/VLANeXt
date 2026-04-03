#!/usr/bin/env python3
"""
Parallel Fine-Alignment Data Collection

Usage:

python run_parallel_align_only.py --workers 25 --episodes 400 \
    --base_dir /data/public/NAS/VLANeXt/dataset/fine_align

"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--episodes', type=int, default=1000, help='Episodes per worker')
    parser.add_argument('--base_dir', type=str,
                       default='/data/public/NAS/VLANeXt/dataset/fine_align',
                       help='Base output directory')
    args = parser.parse_args()

    print("=" * 80)
    print("Parallel Fine-Alignment Data Collection")
    print("=" * 80)
    print(f"Workers: {args.workers}")
    print(f"Episodes per worker: {args.episodes}")
    print(f"Total episodes: {args.workers * args.episodes}")
    print(f"Output: {args.base_dir}/worker_*")
    print("=" * 80)
    print()

    base_path = Path(args.base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    sim_dir = Path(__file__).parent.absolute()

    processes = []
    for i in range(args.workers):
        worker_dir = base_path / f"worker_{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        script_content = f"""
import sys
import os
import time

os.chdir('{sim_dir}')
sys.path.insert(0, '{sim_dir}')

import Save_dataset_align_only

Save_dataset_align_only.SAVE_DIR = r'{worker_dir}'
Save_dataset_align_only.MAX_EPISODES = {args.episodes}

if __name__ == "__main__":
    print(f"[Worker {i}] Starting: {args.episodes} episodes -> {worker_dir}")
    Save_dataset_align_only.main()
    print(f"[Worker {i}] Done!")
"""

        script_path = base_path / f"temp_worker_{i}.py"
        script_path.write_text(script_content)

        log_file = open(base_path / f"worker_{i}.log", 'w')
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=sim_dir
        )
        processes.append((proc, log_file, script_path))
        print(f"[Worker {i}] Started (PID: {proc.pid})")
        time.sleep(0.5)

    print()
    print("All workers started!")
    print(f"Monitor logs: tail -f {base_path}/worker_*.log")
    print()
    print("Waiting for completion...")

    start_time = time.time()
    for i, (proc, log_file, script_path) in enumerate(processes):
        proc.wait()
        log_file.close()

        worker_dir = base_path / f"worker_{i}"
        h5_count = len(list(worker_dir.glob("*.h5"))) if worker_dir.exists() else 0
        print(f"[Worker {i}] Finished (exit code: {proc.returncode}, files: {h5_count})")

        script_path.unlink()

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 80)
    print("Merging data...")

    final_dir = base_path / "collected_data_merged"
    final_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    for i in range(args.workers):
        worker_dir = base_path / f"worker_{i}"
        if worker_dir.exists():
            h5_files = list(worker_dir.glob("*.h5"))
            if h5_files:
                print(f"[Worker {i}] Moving {len(h5_files)} episodes...")
                for h5_file in h5_files:
                    new_name = f"worker{i}_{h5_file.name}"
                    shutil.move(str(h5_file), str(final_dir / new_name))
                    total_episodes += 1
                try:
                    worker_dir.rmdir()
                except:
                    pass

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print(f"Time: {minutes}m {seconds}s")
    print(f"Location: {final_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
