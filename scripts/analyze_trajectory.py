"""Trajectory-level analysis for hybrid handoff metrics.

For each episode (npz with per-step dist_mm, lateral_mm, angle_deg, ee_pose):
  * close_once_5mm:    min(dist) <= 5  → 한 번이라도 5mm 영역 도달 (handoff 가능)
  * time_near_5mm:     fraction of steps with dist <= 5  (sensor가 grid 찍을 시간)
  * time_near_8mm:     동일, ≤8mm
  * retreat_mm:        max(dist[t_min:]) - min(dist)  (도달 후 도망 거리)
  * angle_when_near:   mean angle_deg over steps with dist<=5  (NaN if never reached)
  * approach_cos:      mean cosine between step velocity and direction-to-target
                       (양수 = trocar 방향으로 움직임. 0 근처 = 랜덤 wander)

Episode-level handoff success (proposed):
  HANDOFF_OK = (close_once_5mm) AND (time_near_5mm >= 0.20) AND (retreat_mm <= 3)

Usage:
  python -m scripts.analyze_trajectory <eval_dir> [eval_dir ...]
  python -m scripts.analyze_trajectory <eval_dir1> --compare <eval_dir2>
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np


HANDOFF_NEAR_MM = 5.0
HANDOFF_TIME_FRAC = 0.20
HANDOFF_RETREAT_MM = 3.0


def analyze_episode(npz_path: Path) -> dict | None:
    z = np.load(npz_path)
    if "dist_mm" not in z.files:
        return None
    dist = z["dist_mm"].astype(np.float32)
    ang = z["angle_deg"].astype(np.float32) if "angle_deg" in z.files else None
    lat = z["lateral_mm"].astype(np.float32) if "lateral_mm" in z.files else None
    pos = z["ee_pose"].astype(np.float32) if "ee_pose" in z.files else None
    valid = ~np.isnan(dist)
    dist = dist[valid]
    if dist.size < 2:
        return None

    min_dist = float(np.min(dist))
    final_dist = float(dist[-1])
    t_min = int(np.argmin(dist))
    after_min = dist[t_min:]
    retreat = float(np.max(after_min) - min_dist) if after_min.size else 0.0

    near5 = dist <= 5.0
    near8 = dist <= 8.0
    time_near_5 = float(near5.mean())
    time_near_8 = float(near8.mean())

    angle_when_near = float(ang[valid][near5].mean()) if (ang is not None and near5.any()) else float("nan")
    lat_when_near = float(lat[valid][near5].mean()) if (lat is not None and near5.any()) else float("nan")

    # approach direction signal: cos(velocity, -gradient of dist)
    # Since we only have distance scalar, use proxy: sign(d_dist/dt) — negative means approaching.
    # approach_score = fraction of steps where dist is decreasing (or staying same when already near).
    diffs = np.diff(dist)
    approaching = (diffs < -0.05).mean()  # >0.05mm/step toward
    holding = ((np.abs(diffs) <= 0.05) & (dist[1:] <= 5.0)).mean()  # holding within 5mm
    fleeing = (diffs > 0.05).mean()
    approach_signal = float(approaching + holding - fleeing)  # ∈ [-1, 2], higher = better

    handoff_ok = bool(min_dist <= HANDOFF_NEAR_MM and time_near_5 >= HANDOFF_TIME_FRAC and retreat <= HANDOFF_RETREAT_MM)

    return dict(
        episode=npz_path.stem,
        success_marker="S" in npz_path.stem.split("_")[-1],
        min_dist_mm=min_dist,
        final_dist_mm=final_dist,
        retreat_mm=retreat,
        time_near_5mm=time_near_5,
        time_near_8mm=time_near_8,
        angle_when_near_deg=angle_when_near,
        lateral_when_near_mm=lat_when_near,
        approaching_frac=float(approaching),
        holding_frac=float(holding),
        fleeing_frac=float(fleeing),
        approach_signal=approach_signal,
        handoff_ok=handoff_ok,
        n_steps=int(dist.size),
    )


def summarize(rows: list[dict], label: str = "") -> dict:
    if not rows:
        return {}
    n = len(rows)
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    summary = {
        "label": label,
        "n_episodes": n,
        "close_once_5mm_pct": (col("min_dist_mm") <= 5).mean() * 100,
        "close_once_3mm_pct": (col("min_dist_mm") <= 3).mean() * 100,
        "close_once_8mm_pct": (col("min_dist_mm") <= 8).mean() * 100,
        "close_once_10mm_pct": (col("min_dist_mm") <= 10).mean() * 100,
        "handoff_ok_pct": (col("handoff_ok") > 0.5).mean() * 100,
        "min_dist_median_mm": float(np.median(col("min_dist_mm"))),
        "min_dist_mean_mm": float(np.mean(col("min_dist_mm"))),
        "final_dist_median_mm": float(np.median(col("final_dist_mm"))),
        "retreat_median_mm": float(np.median(col("retreat_mm"))),
        "time_near_5mm_median": float(np.median(col("time_near_5mm"))),
        "time_near_8mm_median": float(np.median(col("time_near_8mm"))),
        "approach_signal_median": float(np.median(col("approach_signal"))),
        "approaching_frac_mean": float(np.mean(col("approaching_frac"))),
        "fleeing_frac_mean": float(np.mean(col("fleeing_frac"))),
        "sr_pct": (col("success_marker") > 0.5).mean() * 100,
    }
    return summary


def print_summary(s: dict):
    print(f"\n=== {s.get('label','')}  ({s['n_episodes']} ep) ===")
    print(f"  SR (strict):              {s['sr_pct']:5.1f}%")
    print(f"  Handoff OK (5mm+20%+r≤3): {s['handoff_ok_pct']:5.1f}%   ← VLA handoff readiness")
    print(f"  close_once 3 / 5 / 8 / 10 mm: {s['close_once_3mm_pct']:.1f} / {s['close_once_5mm_pct']:.1f} / {s['close_once_8mm_pct']:.1f} / {s['close_once_10mm_pct']:.1f} %")
    print(f"  min_dist  median / mean:  {s['min_dist_median_mm']:.2f} / {s['min_dist_mean_mm']:.2f} mm")
    print(f"  final_dist median:        {s['final_dist_median_mm']:.2f} mm")
    print(f"  retreat   median:         {s['retreat_median_mm']:.2f} mm  (lower = stays near target)")
    print(f"  time_near 5mm / 8mm med:  {s['time_near_5mm_median']:.2%} / {s['time_near_8mm_median']:.2%}")
    print(f"  approach_signal median:   {s['approach_signal_median']:+.3f}  (+ = drifts toward, − = flees)")
    print(f"    approaching/fleeing:    {s['approaching_frac_mean']:.2%}  /  {s['fleeing_frac_mean']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+", help="One or more eval directories (containing traj_ep*.npz)")
    ap.add_argument("--label", default=None, help="Optional label override (single dir only)")
    args = ap.parse_args()

    for d in args.eval_dirs:
        d = Path(d)
        npz_files = sorted(d.glob("traj_ep*.npz"))
        if not npz_files:
            print(f"[{d}] no traj_ep*.npz files", file=sys.stderr)
            continue
        rows = []
        skipped_old = 0
        for f in npz_files:
            r = analyze_episode(f)
            if r is None:
                skipped_old += 1
                continue
            rows.append(r)
        label = args.label if args.label and len(args.eval_dirs) == 1 else str(d)
        s = summarize(rows, label=label)
        print_summary(s)
        if skipped_old:
            print(f"  ({skipped_old} old-format npz skipped — no per-step dist)")


if __name__ == "__main__":
    main()
