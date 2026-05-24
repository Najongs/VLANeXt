"""Cell-by-cell failure pattern analysis for v3 champion eval.

Reproduces eval grid (3 xy × 3 y × 3 angle = 27 cells) and maps each
episode index → phantom position/angle → min_dist. Identifies which
phantom configurations cause worst performance.

Usage:
  python -m scripts.analyze_cell_failures <eval_dir>
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reproduce_grid_27(xy_mm=10.0, y_min_mm=-25.0, y_max_mm=25.0, z_mm=0.0, angle_deg=5.0):
    """27-cell grid in default eval config: xy 3 × y 3 × z 1 × angle 3 × rep 1.

    Matches sim_eval_align_only.build_perturb_grid order (x outer, y mid, z inner,
    angle inner-most, repeats inner-most-most).
    """
    xs = np.linspace(-xy_mm, xy_mm, 3)        # -10, 0, 10
    ys = np.linspace(y_min_mm, y_max_mm, 3)   # -25, 0, 25
    zs = np.array([z_mm])
    angles = np.linspace(-angle_deg, angle_deg, 3)  # -5, 0, 5

    cells = []
    for x in xs:
        for y in ys:
            for z in zs:
                for a in angles:
                    cells.append({"x_mm": float(x), "y_mm": float(y), "z_mm": float(z), "angle_deg": float(a)})
    return cells


def analyze(eval_dir):
    eval_dir = Path(eval_dir)
    npz_files = sorted(eval_dir.glob("traj_ep*.npz"))
    if not npz_files:
        print(f"No traj_ep*.npz in {eval_dir}", file=sys.stderr)
        return

    cells = reproduce_grid_27()
    if len(npz_files) != len(cells):
        print(f"Warning: {len(npz_files)} npz vs {len(cells)} expected cells")

    rows = []
    for i, f in enumerate(npz_files):
        if i >= len(cells):
            break
        z = np.load(f)
        dist = z["dist_mm"]
        dist = dist[~np.isnan(dist)]
        min_d = float(dist.min()) if dist.size else float("nan")
        final_d = float(dist[-1]) if dist.size else float("nan")
        # Did robot try? (any movement)
        moved_mm = float(np.abs(np.diff(dist)).sum()) if dist.size > 1 else 0.0
        success = "_S" in f.name
        cell = cells[i]
        rows.append({
            **cell,
            "ep": i + 1,
            "min_dist": min_d,
            "final_dist": final_d,
            "moved_mm": moved_mm,
            "success": success,
        })

    # Per-cell summary table
    print(f"\n=== {eval_dir.name} — cell-by-cell breakdown ===\n")
    print(f"{'ep':>3} {'x':>5} {'y':>5} {'ang':>5} {'min':>7} {'final':>7} {'moved':>7} {'S':>2}")
    print("-" * 50)
    for r in rows:
        flag = "✓" if r["success"] else "✗"
        print(f"{r['ep']:>3} {r['x_mm']:>+5.0f} {r['y_mm']:>+5.0f} {r['angle_deg']:>+5.0f} "
              f"{r['min_dist']:>7.2f} {r['final_dist']:>7.2f} {r['moved_mm']:>7.1f} {flag:>2}")

    # Aggregate by axis
    print("\n=== By y position (3 cells × 9 ep each) ===")
    for y in sorted(set(r["y_mm"] for r in rows)):
        subset = [r for r in rows if r["y_mm"] == y]
        mins = [r["min_dist"] for r in subset]
        print(f"  y={y:+.0f}mm: mean_min={np.mean(mins):.2f}, median_min={np.median(mins):.2f}, "
              f"n_under_2mm={sum(1 for m in mins if m <= 2.0)}, n_success={sum(1 for r in subset if r['success'])}/{len(subset)}")

    print("\n=== By x position ===")
    for x in sorted(set(r["x_mm"] for r in rows)):
        subset = [r for r in rows if r["x_mm"] == x]
        mins = [r["min_dist"] for r in subset]
        print(f"  x={x:+.0f}mm: mean_min={np.mean(mins):.2f}, median_min={np.median(mins):.2f}, "
              f"n_under_2mm={sum(1 for m in mins if m <= 2.0)}, n_success={sum(1 for r in subset if r['success'])}/{len(subset)}")

    print("\n=== By angle ===")
    for a in sorted(set(r["angle_deg"] for r in rows)):
        subset = [r for r in rows if r["angle_deg"] == a]
        mins = [r["min_dist"] for r in subset]
        print(f"  angle={a:+.0f}°: mean_min={np.mean(mins):.2f}, median_min={np.median(mins):.2f}, "
              f"n_under_2mm={sum(1 for m in mins if m <= 2.0)}, n_success={sum(1 for r in subset if r['success'])}/{len(subset)}")

    # Worst & best cells
    sorted_rows = sorted(rows, key=lambda r: r["min_dist"])
    print("\n=== Best 5 cells (lowest min_dist) ===")
    for r in sorted_rows[:5]:
        print(f"  ep{r['ep']}: x={r['x_mm']:+.0f} y={r['y_mm']:+.0f} ang={r['angle_deg']:+.0f}°  min={r['min_dist']:.2f}mm")
    print("\n=== Worst 5 cells (highest min_dist) ===")
    for r in sorted_rows[-5:]:
        print(f"  ep{r['ep']}: x={r['x_mm']:+.0f} y={r['y_mm']:+.0f} ang={r['angle_deg']:+.0f}°  min={r['min_dist']:.2f}mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    args = ap.parse_args()
    analyze(args.eval_dir)
