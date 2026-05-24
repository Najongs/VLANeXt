#!/usr/bin/env python
"""Honest metric suite for fine-align eval data.

Designed for `--no-early-term` eval data where each episode runs full max_steps.
All metrics derived from per-step `lateral_mm` (axis-projected lateral distance).

Metrics:
  Reach@K       : fraction of eps where MIN lateral over trajectory < K (K=5/2/1 mm)
  TTA@K         : median step at which lateral first crosses < K (across reached eps)
  min_lat_med   : median per-ep min(lateral) — peak precision
  HoldSR@K_N    : fraction of eps with any N-step window where all lateral < K
                  (default K=2.5, N=20)
  Settled_lat   : median lateral over LAST 30 steps (per ep), then median across eps
  Settled_std   : std lateral over LAST 30 steps (per ep), then median across eps
  Safety_settled: p99 of per-ep settled_lat across eps (worst-case settled position)
  Settled_max_window : "last 30 step max < 2.5" SR — alternative settled-based hold

Usage:
  python scripts/honest_metrics.py <eval_dir_or_glob> [--compare <orig_dir>]
"""
import argparse
import sys
from pathlib import Path
import numpy as np


def per_episode_metrics(npz_dir, settled_window=30, hold_k=2.5, hold_n=20):
    eps_data = []
    for f in sorted(Path(npz_dir).glob("traj_ep*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            if "lateral_mm" not in d.files:
                continue
            lat = np.asarray(d["lateral_mm"], dtype=float)
            if lat.size == 0:
                continue

            # Time-aware metrics
            min_lat = float(np.min(lat))
            min_step = int(np.argmin(lat))

            # TTA@K: first step where lateral < K
            tta = {}
            for K in [5.0, 2.0, 1.0]:
                idx = np.where(lat < K)[0]
                tta[K] = int(idx[0]) if idx.size else None

            # HoldSR@K_N: any N-step window where all < K (using max in window)
            hold_ok = False
            if lat.size >= hold_n:
                run = 0
                for v in lat:
                    if v < hold_k:
                        run += 1
                        if run >= hold_n:
                            hold_ok = True
                            break
                    else:
                        run = 0

            # Settled metrics (last N steps)
            sw = min(settled_window, lat.size)
            settled_lat = float(np.median(lat[-sw:]))
            settled_std = float(np.std(lat[-sw:]))
            settled_max = float(np.max(lat[-sw:]))

            eps_data.append(dict(
                n_steps=lat.size,
                min_lat=min_lat,
                min_step=min_step,
                tta=tta,
                hold_ok=hold_ok,
                settled_lat=settled_lat,
                settled_std=settled_std,
                settled_max=settled_max,
            ))
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)
    return eps_data


def aggregate(eps_data):
    if not eps_data:
        return None
    n = len(eps_data)
    # Reach@K — any step lateral < K (i.e. min_lat < K)
    reach = {K: sum(1 for e in eps_data if e["min_lat"] < K) / n for K in [5.0, 2.0, 1.0]}

    # TTA@K — median first-reach step (only eps that reached)
    tta_med = {}
    for K in [5.0, 2.0, 1.0]:
        steps = [e["tta"][K] for e in eps_data if e["tta"][K] is not None]
        tta_med[K] = float(np.median(steps)) if steps else float("nan")

    min_lats = [e["min_lat"] for e in eps_data]
    settled_lats = [e["settled_lat"] for e in eps_data]
    settled_stds = [e["settled_std"] for e in eps_data]
    settled_maxs = [e["settled_max"] for e in eps_data]
    n_steps = [e["n_steps"] for e in eps_data]
    holds = [e["hold_ok"] for e in eps_data]

    settled_sub25 = sum(1 for s in settled_maxs if s < 2.5) / n   # last-window max < 2.5
    settled_sub2 = sum(1 for s in settled_lats if s < 2.0) / n    # settled median < 2.0

    return dict(
        n=n,
        n_steps_med=float(np.median(n_steps)),
        n_steps_mean=float(np.mean(n_steps)),
        reach5=reach[5.0],
        reach2=reach[2.0],
        reach1=reach[1.0],
        tta5_med=tta_med[5.0],
        tta2_med=tta_med[2.0],
        tta1_med=tta_med[1.0],
        min_lat_med=float(np.median(min_lats)),
        min_lat_p25=float(np.percentile(min_lats, 25)),
        holdSR=sum(holds) / n,
        settled_lat_med=float(np.median(settled_lats)),
        settled_std_med=float(np.median(settled_stds)),
        settled_max_p99=float(np.percentile(settled_maxs, 99)),
        settled_lat_p99=float(np.percentile(settled_lats, 99)),
        # Settled-window-based hold (no contig requirement, just window):
        settled_max_sub25=settled_sub25,   # last 30 step max < 2.5mm SR (cleaner than holdSR contig)
        settled_lat_sub2=settled_sub2,     # settled median < 2.0mm SR
    )


def fmt_row(tag, m):
    return (f"{tag:24} n={m['n']:>2} steps_med={m['n_steps_med']:>4.0f} | "
            f"R5={m['reach5']*100:>5.1f}% R2={m['reach2']*100:>5.1f}% R1={m['reach1']*100:>5.1f}% | "
            f"TTA2={m['tta2_med']:>4.0f}st | "
            f"mLat={m['min_lat_med']:>4.2f}mm p25={m['min_lat_p25']:>4.2f} | "
            f"Hold20={m['holdSR']*100:>5.1f}% | "
            f"Settled={m['settled_lat_med']:>4.2f}±{m['settled_std_med']:>4.2f} | "
            f"Max30<2.5={m['settled_max_sub25']*100:>5.1f}% | "
            f"SafeStl_p99={m['settled_lat_p99']:>4.2f}mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="eval directories with traj_ep*.npz")
    ap.add_argument("--settled-window", type=int, default=30)
    ap.add_argument("--hold-k", type=float, default=2.5)
    ap.add_argument("--hold-n", type=int, default=20)
    args = ap.parse_args()

    print(f"# Honest metrics — settled_window={args.settled_window}, hold K={args.hold_k}mm N={args.hold_n}step")
    print(f"# Columns: R5/R2/R1=Reach@K (any step lateral<K), TTA2=median time-to-2mm,")
    print(f"#   mLat=median per-ep min(lateral) [p25=25th%ile], Hold20=lateral<{args.hold_k} for {args.hold_n}+ contig steps,")
    print(f"#   Settled=median±std of last {args.settled_window} steps' lateral, Max30<2.5=last 30 step max<2.5 SR,")
    print(f"#   SafeStl_p99=p99 of settled_lat across eps (worst-case medical bound)")
    print()
    for d in args.dirs:
        eps = per_episode_metrics(d, args.settled_window, args.hold_k, args.hold_n)
        m = aggregate(eps)
        if m is None:
            print(f"{d}: EMPTY")
            continue
        tag = Path(d).parent.name + "/" + Path(d).name.replace("align_eval_", "")
        print(fmt_row(tag[:80], m))


if __name__ == "__main__":
    main()
