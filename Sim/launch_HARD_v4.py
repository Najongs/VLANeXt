"""Launch HARD_v4 data generation with expanded Y range [-25, +75]mm.

Bypass run_parallel.py's symmetric grid: write per-worker JSON cells covering
asymmetric Y range, spawn N workers manually with --grid-cells-file.

Reason: WIDE eval samples y in (-25, +75) per sim_eval_align_only.py:239-240;
existing HARD training only covers y in [-15, +15]. dy=+25/+50/+75 cells fail
0/18 in eval — pure OOD.
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import numpy as np

BASE_DIR_DEFAULT = "/home/najo/NAS/VLANeXt/dataset/fine_align/HARD_v4_y75"
NUM_WORKERS_DEFAULT = 10
EPISODES_PER_WORKER_DEFAULT = 100   # → 1000 total

# Cell grid (asymmetric Y to match eval distribution).
X_EDGES = [-15.0, -5.0, 5.0, 15.0]                   # 3 bins
Y_EDGES = [-25.0, -5.0, 15.0, 35.0, 55.0, 75.0]      # 5 bins
Z_EDGES = [-10.0, 10.0]                              # 1 bin (full range)


def build_cells():
    cells = []
    for xi in range(len(X_EDGES) - 1):
        for yi in range(len(Y_EDGES) - 1):
            for zi in range(len(Z_EDGES) - 1):
                cells.append([
                    X_EDGES[xi], X_EDGES[xi + 1],
                    Y_EDGES[yi], Y_EDGES[yi + 1],
                    Z_EDGES[zi], Z_EDGES[zi + 1],
                ])
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT)
    ap.add_argument("--episodes-per-worker", type=int, default=EPISODES_PER_WORKER_DEFAULT)
    ap.add_argument("--base-dir", default=BASE_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    cells = build_cells()
    np.random.seed(args.seed)
    np.random.shuffle(cells)

    # IMPORTANT: HARD generator forces MAX_EPISODES = len(GRID_CELLS) in grid mode
    # (Save_dataset_align_HARD.py:991). So to get N episodes per worker, duplicate cells N times.
    repeats = max(1, args.episodes_per_worker // len(cells))
    worker_jsons = []
    for w in range(args.workers):
        p = base / f"_grid_cells_worker_{w}.json"
        # Shuffle per-worker so episode order varies.
        rng_w = np.random.default_rng(args.seed + w * 17)
        cells_w = list(cells) * repeats
        rng_w.shuffle(cells_w)
        with open(p, "w") as f:
            json.dump(cells_w, f)
        worker_jsons.append(p)
    print(f"  cells × {repeats} repeats per worker → {len(cells_w)} ep/worker")

    print("=" * 70)
    print(f"HARD_v4 Generation — asymmetric Y [{Y_EDGES[0]}, {Y_EDGES[-1]}]mm")
    print("=" * 70)
    print(f"  cells:               {len(cells)} ({len(X_EDGES)-1}x{len(Y_EDGES)-1}x{len(Z_EDGES)-1})")
    print(f"  workers:             {args.workers}")
    print(f"  episodes/worker:     {args.episodes_per_worker}")
    print(f"  total target:        {args.workers * args.episodes_per_worker}")
    print(f"  base dir:            {base}")
    print("=" * 70)

    procs = []
    log_dir = Path("/home/najo/NAS/VLANeXt/logs")
    log_dir.mkdir(exist_ok=True)
    for w in range(args.workers):
        save_dir = base / f"worker_{w}"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", "-u",
            "/home/najo/NAS/VLANeXt/Sim/Save_dataset_align_HARD.py",
            "--save-dir", str(save_dir),
            "--num-episodes", str(args.episodes_per_worker),
            "--grid-cells-file", str(worker_jsons[w]),
            "--seed", str(args.seed + w * 1000),
            "--randomize-phantom-pos",
            "--no-side-camera",
        ]
        log_path = log_dir / f"hard_v4_worker_{w}.log"
        f = open(log_path, "wb")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only render via MuJoCo (mjpython)
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                       "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                              cwd="/home/najo/NAS/VLANeXt/Sim")
        procs.append((p, log_path))
        print(f"  worker {w}: pid={p.pid}  log={log_path}")

    print(f"\nSpawned {len(procs)} workers. PIDs: {[p.pid for p,_ in procs]}")
    print("Monitor: tail -f logs/hard_v4_worker_*.log")


if __name__ == "__main__":
    main()
