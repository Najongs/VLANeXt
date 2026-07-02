"""Compare fine-align eval runs from their metrics_summary.csv files.

Usage:
    python -m scripts.compare_eval <shard_dir_1> <shard_dir_2> ...
        [--names label1 label2 ...]
        [--out report.md]
    python -m scripts.compare_eval --auto    # discover all known runs under checkpoints/

Each shard_dir must contain metrics_summary.csv (produced by sim_eval_align_only.py).

Prints a multi-axis comparison table comparing:
  - SR at distance thresholds {3, 5, 8, 10, 15, 20} mm (using strict success: dist<thr AND ang<10°)
  - SR at angle thresholds {5, 10, 15, 20}° (using strict: ang<thr AND dist<5mm)
  - Pos-only SR (dist<5mm, ignore angle) → isolates angle bottleneck
  - Angle-only SR (ang<10°, ignore dist) → isolates position bottleneck
  - Distributions: final_dist_mm / final_angle_deg / min_dist_mm  (mean / median / p25 / p75 / p95)
  - Per-cell SR pattern (sparse)
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_summary(shard_dir: str):
    path = os.path.join(shard_dir, "metrics_summary.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "success": int(row["success"]),
                    "final_dist_mm": float(row["final_dist_mm"]),
                    "final_angle_deg": float(row["final_angle_deg"]),
                    "min_dist_mm": float(row["min_dist_mm"]),
                    "steps": int(row["steps"]),
                    "cell_idx": int(row["cell_idx"]),
                    "initial_dist_mm": float(row["initial_dist_mm"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def dist_thr_sr(rows, dist_thr, angle_thr=10.0):
    if not rows: return float("nan")
    n_ok = sum(1 for r in rows if r["final_dist_mm"] < dist_thr and r["final_angle_deg"] < angle_thr)
    return 100.0 * n_ok / len(rows)


def angle_thr_sr(rows, angle_thr, dist_thr=5.0):
    if not rows: return float("nan")
    n_ok = sum(1 for r in rows if r["final_angle_deg"] < angle_thr and r["final_dist_mm"] < dist_thr)
    return 100.0 * n_ok / len(rows)


def pos_only_sr(rows, dist_thr=5.0):
    if not rows: return float("nan")
    return 100.0 * sum(1 for r in rows if r["final_dist_mm"] < dist_thr) / len(rows)


def angle_only_sr(rows, angle_thr=10.0):
    if not rows: return float("nan")
    return 100.0 * sum(1 for r in rows if r["final_angle_deg"] < angle_thr) / len(rows)


def close_once_sr(rows, dist_thr):
    """% of episodes where the model ever came within dist_thr (uses min_dist_mm).
    Compares to strict SR to isolate the "approached but couldn't hold" failure mode."""
    if not rows: return float("nan")
    return 100.0 * sum(1 for r in rows if r["min_dist_mm"] < dist_thr) / len(rows)


def retreat_stats(rows):
    """retreat = final_dist - min_dist. How much did the model drift away from
    its closest approach. Large value → hover/oscillation/no-hold."""
    xs = [r["final_dist_mm"] - r["min_dist_mm"] for r in rows]
    if not xs: return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {"mean": mean(xs), "median": median(xs), "p25": pct(xs, .25), "p75": pct(xs, .75), "p95": pct(xs, .95)}


def motion_stats(rows):
    """How much did the model actually move toward the goal.
    progress = initial_dist - min_dist. Large→approached. Small/zero→didn't move."""
    xs = [r["initial_dist_mm"] - r["min_dist_mm"] for r in rows]
    if not xs: return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {"mean": mean(xs), "median": median(xs), "p25": pct(xs, .25), "p75": pct(xs, .75), "p95": pct(xs, .95)}


def dist_stats(rows, key):
    xs = [r[key] for r in rows]
    if not xs: return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {"mean": mean(xs), "median": median(xs), "p25": pct(xs, .25), "p75": pct(xs, .75), "p95": pct(xs, .95)}


def per_cell_sr(rows):
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell_idx"]].append(r["success"])
    return {c: 100.0 * sum(v) / len(v) for c, v in by_cell.items()}


def fmt(x, w=6, prec=1):
    if x != x:  # nan
        return "  N/A "
    return f"{x:>{w}.{prec}f}"


def render_comparison(runs):
    """runs: list of (label, rows)."""
    labels = [lbl for lbl, _ in runs]
    out = []
    col_w = max(8, max(len(l) for l in labels) + 2)

    def header(title):
        out.append(f"\n## {title}")
        out.append("| Metric | " + " | ".join(f"{l:>{col_w}}" for l in labels) + " |")
        out.append("|" + "---|" * (len(labels) + 1))

    def row(name, values, w=col_w, prec=1):
        out.append(f"| {name} | " + " | ".join(fmt(v, w=w, prec=prec) for v in values) + " |")

    # Sample counts
    out.append("\n## Run summary")
    out.append("| Run | N episodes | Mean init dist (mm) |")
    out.append("|---|---|---|")
    for lbl, rows in runs:
        idist = mean(r["initial_dist_mm"] for r in rows) if rows else float("nan")
        out.append(f"| {lbl} | {len(rows)} | {fmt(idist)} |")

    # Strict SR (dist & angle both)
    header("Success rate (strict: dist<thr AND angle<10°)")
    for thr in [3, 5, 8, 10, 15, 20]:
        row(f"SR @ dist<{thr}mm", [dist_thr_sr(rows, thr) for _, rows in runs])

    header("Success rate (strict: angle<thr AND dist<5mm)")
    for thr in [5, 10, 15, 20]:
        row(f"SR @ angle<{thr}°", [angle_thr_sr(rows, thr) for _, rows in runs])

    header("Axis-only success (one axis ignored — reveals which is bottleneck)")
    row("Pos-only SR (dist<5mm)", [pos_only_sr(rows, 5.0) for _, rows in runs])
    row("Angle-only SR (ang<10°)", [angle_only_sr(rows, 10.0) for _, rows in runs])
    row("Pos-only SR (dist<8mm)", [pos_only_sr(rows, 8.0) for _, rows in runs])
    row("Angle-only SR (ang<15°)", [angle_only_sr(rows, 15.0) for _, rows in runs])

    header("Approach behavior (min_dist = closest reached, before retreat)")
    out.append("Close-once SR = % episodes that ever came within thr. Compare to strict SR:")
    out.append("- close-once ≫ strict → 'approached but couldn't hold/align' (hold-failure)")
    out.append("- close-once ≈ strict → 'never got close at all' (approach-failure)")
    for thr in [3, 5, 8, 10, 15]:
        row(f"Close-once <{thr}mm", [close_once_sr(rows, thr) for _, rows in runs])

    out.append(f"\n*Retreat = final_dist - min_dist* (large = drifted away from closest approach)")
    stats = [retreat_stats(rows) for _, rows in runs]
    for stat_name in ["mean", "median", "p75", "p95"]:
        row(f"retreat {stat_name} (mm)", [s[stat_name] for s in stats])

    out.append(f"\n*Progress = initial_dist - min_dist* (large = moved toward goal; ~0 = didn't move)")
    stats = [motion_stats(rows) for _, rows in runs]
    for stat_name in ["mean", "median", "p25", "p75"]:
        row(f"progress {stat_name} (mm)", [s[stat_name] for s in stats])

    # Final pose distribution
    header("Final pose distribution (mm / deg) — lower better")
    for key, name in [("final_dist_mm", "final_dist_mm"), ("final_angle_deg", "final_angle_deg"), ("min_dist_mm", "min_dist_mm (closest reached)")]:
        out.append(f"\n*{name}*")
        stats = [dist_stats(rows, key) for _, rows in runs]
        for stat_name in ["mean", "median", "p25", "p75", "p95"]:
            row(f"  {stat_name}", [s[stat_name] for s in stats])

    # Per-cell SR (compact)
    out.append("\n## Per-cell SR (cell_idx → SR %)")
    out.append("Pattern reveals systematic failure modes (e.g., consistent fail on extreme corner cells).")
    all_cells = set()
    cells_per_run = []
    for _, rows in runs:
        c = per_cell_sr(rows)
        cells_per_run.append(c)
        all_cells |= c.keys()
    out.append("| cell_idx | " + " | ".join(f"{l:>{col_w}}" for l in labels) + " |")
    out.append("|" + "---|" * (len(labels) + 1))
    for cell in sorted(all_cells):
        vals = [c.get(cell, float("nan")) for c in cells_per_run]
        row(str(cell), vals)

    return "\n".join(out)


def discover_runs():
    """Find all metrics_summary.csv under known checkpoint root."""
    found = []
    for p in sorted(glob.glob("/home/najo/NAS/VLANeXt/checkpoints/**/metrics_summary.csv", recursive=True)):
        sd = os.path.dirname(p)
        # label = parent ckpt dir name + shard subdir
        parts = sd.split("/")
        label = "/".join(parts[-3:])
        found.append((label, sd))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dirs", nargs="*", help="Directories containing metrics_summary.csv")
    ap.add_argument("--names", nargs="*", help="Labels for each run (default: shard dir basename)")
    ap.add_argument("--auto", action="store_true", help="Auto-discover all eval shards under checkpoints/")
    ap.add_argument("--out", type=str, default=None, help="Optional output .md path; otherwise stdout")
    args = ap.parse_args()

    if args.auto:
        discovered = discover_runs()
        print(f"# Auto-discovered {len(discovered)} eval runs:")
        for lbl, sd in discovered:
            print(f"  - {lbl} → {sd}")
        if not discovered:
            sys.exit("No eval shards found.")
        runs_specs = discovered
    else:
        if not args.shard_dirs:
            sys.exit("Provide shard_dirs or --auto")
        labels = args.names or [Path(p).name for p in args.shard_dirs]
        if len(labels) != len(args.shard_dirs):
            sys.exit("--names must match shard_dirs count")
        runs_specs = list(zip(labels, args.shard_dirs))

    runs = []
    for lbl, sd in runs_specs:
        rows = load_summary(sd)
        if rows is None:
            print(f"WARN: {sd} missing metrics_summary.csv, skipped")
            continue
        runs.append((lbl, rows))

    if len(runs) < 1:
        sys.exit("No usable runs")

    report = render_comparison(runs)
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# Eval comparison\n")
            f.write(report)
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
