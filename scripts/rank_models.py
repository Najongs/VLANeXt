"""Rank eval directories by a composite precision score (not just SR).

Why: post-retreat-2mm change, SR threshold (close_once_5mm) is loose for
fine-precision comparison. This script aggregates multiple "goal-reaching"
metrics into a single rank-sum score.

Usage:
    python -m scripts.rank_models <eval_dir1> [eval_dir2 ...] \
        [--labels lbl1 lbl2 ...] [--out /tmp/rank.md]

Metrics (all "higher = better" via sign-flips for distance metrics):
    +close_once_2mm_pct    # touched 2mm at some point
    +close_once_1mm_pct    # touched 1mm (sparse but informative)
    +time_near_2mm_median  # sustained near goal (hold quality)
    +handoff_ok_pct        # composite success criterion
    -min_dist_mean_mm      # avg closest approach (lower = better)
    -p90_dist_median_mm    # tail behavior
    -lateral_when_near_mm  # lateral error when close
    -angle_when_near_deg   # angle when close
    -retreat_median_mm     # how much it backs away after reaching

Composite: each model gets a per-metric rank (1 = best). Lower rank-sum wins.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_trajectory import analyze_episode, summarize


METRICS = [
    # (key, label, direction)  direction = +1 (higher better) or -1 (lower better)
    ("close_once_2mm_pct",        "2mm%",      +1),
    ("close_once_1mm_pct",        "1mm%",      +1),
    ("time_near_2mm_median",      "t≤2mm",     +1),
    ("handoff_ok_pct",            "handoff",   +1),
    ("min_dist_mean_mm",          "mean_min",  -1),
    ("p90_dist_median_mm",        "p90",       -1),
    ("lateral_when_near_mm",      "lat<5",     -1),
    ("angle_when_near_deg",       "ang<5",     -1),
    ("retreat_median_mm",         "retreat",   -1),
]


def load_eval_dir(path: Path) -> dict | None:
    rows = [r for r in (analyze_episode(f) for f in sorted(path.glob("traj_ep*.npz"))) if r is not None]
    if not rows:
        return None
    s = summarize(rows, label=path.name)
    # Fill missing keys with NaN
    for k, _, _ in METRICS:
        if k not in s:
            # angle/lat_when_near come from per-episode aggregation — recompute
            vals = np.array([r.get(k.replace("_when_near", "").replace("mm", "mm").replace("deg", "deg") if False else k.replace("_median_mm", "_when_near_mm").replace("_deg_median", "_when_near_deg"), float("nan")) for r in rows])
            if k == "lateral_when_near_mm":
                vals = np.array([r.get("lateral_when_near_mm", float("nan")) for r in rows])
                s[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
            elif k == "angle_when_near_deg":
                vals = np.array([r.get("angle_when_near_deg", float("nan")) for r in rows])
                s[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
            else:
                s[k] = float("nan")
    return s


def rank_models(summaries: list[dict]) -> list[dict]:
    """Return summaries with added `rank_sum` and per-metric ranks."""
    n = len(summaries)
    if n == 0:
        return []
    # Per-metric rank (1 = best). NaN sinks to worst.
    for key, _, direction in METRICS:
        vals = np.array([s.get(key, float("nan")) for s in summaries])
        # Replace NaN with worst value for ranking
        if direction > 0:
            vals_for_sort = np.where(np.isnan(vals), -np.inf, vals)
            order = np.argsort(-vals_for_sort)   # desc
        else:
            vals_for_sort = np.where(np.isnan(vals), np.inf, vals)
            order = np.argsort(vals_for_sort)    # asc
        ranks = np.empty(n, dtype=int)
        # Average rank for ties
        prev_val = None
        cur_rank = 1
        tied_indices = []
        for rank_pos, idx in enumerate(order, start=1):
            v = vals_for_sort[idx]
            if prev_val is None or v != prev_val:
                if tied_indices:
                    avg = sum(t[1] for t in tied_indices) / len(tied_indices)
                    for ti, _ in tied_indices:
                        ranks[ti] = round(avg)
                tied_indices = []
                prev_val = v
            tied_indices.append((idx, rank_pos))
        if tied_indices:
            avg = sum(t[1] for t in tied_indices) / len(tied_indices)
            for ti, _ in tied_indices:
                ranks[ti] = round(avg)
        for i, s in enumerate(summaries):
            s.setdefault("_ranks", {})[key] = int(ranks[i])

    for s in summaries:
        s["rank_sum"] = sum(s["_ranks"].values())
    summaries.sort(key=lambda s: s["rank_sum"])
    return summaries


def render_table(summaries: list[dict], labels: list[str]) -> str:
    if not summaries:
        return "(no data)"
    cols = ["label", "n"] + [m[1] for m in METRICS] + ["Σrank"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["----"] * len(cols)) + "|"]
    for s, lbl in zip(summaries, labels):
        row = [lbl, str(s["n_episodes"])]
        for key, _, direction in METRICS:
            v = s.get(key, float("nan"))
            if np.isnan(v):
                row.append("NaN")
            elif key.endswith("_pct"):
                row.append(f"{v:.1f}")
            else:
                row.append(f"{v:.2f}")
        row.append(str(s["rank_sum"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Custom labels (one per dir). Default = dir name.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write markdown table to file too.")
    args = ap.parse_args()

    summaries = []
    used_labels = []
    for i, d in enumerate(args.eval_dirs):
        if not d.is_dir():
            print(f"[rank] skip {d} (not a dir)", file=sys.stderr); continue
        s = load_eval_dir(d)
        if s is None:
            print(f"[rank] skip {d} (no episodes)", file=sys.stderr); continue
        summaries.append(s)
        used_labels.append((args.labels[i] if args.labels and i < len(args.labels) else d.name))

    if not summaries:
        print("No usable eval dirs found.", file=sys.stderr)
        sys.exit(1)

    rank_models(summaries)
    # Re-order labels by sorted summaries
    order_by_label = {s["label"]: i for i, s in enumerate(summaries)}
    # Note: summaries was sorted in-place by rank_models; labels must follow
    # but used_labels was built before sort, so realign:
    # The simplest: rebuild from current summaries' "label" field (= path.name)
    final_labels = [used_labels[order_by_label.get(s["label"], 0)] if s["label"] in order_by_label else s["label"]
                    for s in summaries]
    # Cleaner: just re-pair by sorting both together
    paired = list(zip(used_labels, summaries))
    paired_sorted = sorted(paired, key=lambda x: x[1]["rank_sum"])
    final_labels = [p[0] for p in paired_sorted]
    summaries_sorted = [p[1] for p in paired_sorted]

    out = render_table(summaries_sorted, final_labels)
    print()
    print("=" * 90)
    print(f"Ranked precision comparison ({len(summaries)} models). Lower Σrank = better.")
    print("=" * 90)
    print(out)
    print()
    print(f"🏆 Winner: {final_labels[0]} (Σrank={summaries_sorted[0]['rank_sum']})")
    if args.out:
        args.out.write_text(out + "\n")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
