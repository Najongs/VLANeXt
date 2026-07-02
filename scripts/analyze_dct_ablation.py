#!/usr/bin/env python
"""DCT ablation 결과 정리 — 6 ckpts (dct_off/on × ck500/1000/1500) 표 형식 출력.

각 eval dir의 metrics_summary.csv를 로드해서:
- SR_old (final dist < 5mm)
- minLat (median min_dist)
- final_lat_med
- final_ang_med (success cells only)
- safety_bound (worst final_lat, p99)
- per-region SR (perturb_y_mm = -25 / 0 / +25)
산출. 동일 spec 비교라 paired diff 출력.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def aggregate(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    n = len(rows)

    fdist = [float(r["final_dist_mm"]) for r in rows]
    flat = [float(r["final_lateral_mm"]) for r in rows]
    fang = [float(r["final_angle_deg"]) for r in rows]
    minD = [float(r["min_dist_mm"]) for r in rows]
    success = [int(r["success"]) for r in rows]

    sr_old = sum(1 for d in fdist if d < 5.0) / n

    # close_2mm SR (lateral basis)
    close_2 = sum(1 for v in flat if v < 2.0) / n
    close_5 = sum(1 for v in flat if v < 5.0) / n

    minLat_med = statistics.median(minD)
    flat_med = statistics.median(flat)
    # ang on near-goal cells only (final_dist < 5mm)
    near_ang = [a for a, d in zip(fang, fdist) if d < 5.0]
    ang_med = statistics.median(near_ang) if near_ang else float("nan")
    safety = sorted(flat)[int(0.99 * n) - 1] if n >= 10 else max(flat)

    # per y region SR_old
    region = {-25: [], 0: [], 25: []}
    for r, d in zip(rows, fdist):
        py = round(float(r["perturb_y_mm"]))
        if py in region:
            region[py].append(d < 5.0)
    region_sr = {k: (sum(v) / len(v) if v else None, len(v)) for k, v in region.items()}

    return {
        "n": n,
        "sr_old": sr_old,
        "close_5": close_5,
        "close_2": close_2,
        "minLat_med": minLat_med,
        "flat_med": flat_med,
        "ang_med": ang_med,
        "safety": safety,
        "region": region_sr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/najo/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_NEARGOAL")
    args = ap.parse_args()

    root = Path(args.root)
    results = {}
    for variant in ["off", "on"]:
        ckpt_dir = root / f"dct_{variant}_v1"
        for step in [500, 1000, 1500]:
            pattern = f"align_eval_step{step}_exec2_diff10_SR*"
            candidates = sorted(ckpt_dir.glob(pattern))
            if not candidates:
                # also accept non-SR suffix
                no_sr = sorted(ckpt_dir.glob(f"align_eval_step{step}_exec2_diff10"))
                candidates = no_sr
            if not candidates:
                print(f"  MISSING: {variant} ck{step}")
                continue
            ed = candidates[-1]
            csv_p = ed / "metrics_summary.csv"
            if not csv_p.exists():
                print(f"  MISSING CSV: {csv_p}")
                continue
            agg = aggregate(csv_p)
            if agg:
                results[f"dct_{variant}_ck{step}"] = agg
                print(f"  loaded: {ed.name}")

    if not results:
        print("No results found.")
        return

    print("\n=== DCT ablation summary (27-cell, retreat=2, exec=2, diff=10) ===")
    print(f"{'variant':<22} | {'n':>3} | {'SR_old':>7} | {'close5':>7} | {'close2':>7} | {'minLat':>7} | {'finLat':>7} | {'ang°':>5} | {'safety':>7} | y=-25 | y=0   | y=+25")
    print("-" * 145)
    for k, v in results.items():
        r = v["region"]
        rstr = lambda key: f"{r[key][0]*100:.0f}/{r[key][1]}" if r[key][0] is not None else "  ?  "
        print(f"{k:<22} | {v['n']:>3} | {v['sr_old']*100:>6.1f}% | {v['close_5']*100:>6.1f}% | {v['close_2']*100:>6.1f}% | {v['minLat_med']:>6.2f}mm | {v['flat_med']:>6.2f}mm | {v['ang_med']:>4.2f}° | {v['safety']:>6.2f}mm | {rstr(-25):>5} | {rstr(0):>5} | {rstr(25):>5}")

    print("\n=== Paired diff (DCT_on - DCT_off) ===")
    print(f"{'step':<6} | {'ΔSR_old':>9} | {'Δclose5':>9} | {'Δclose2':>9} | {'ΔminLat':>9} | {'ΔfinLat':>9} | {'Δang':>7} | {'Δsafety':>9}")
    print("-" * 90)
    for step in [500, 1000, 1500]:
        off = results.get(f"dct_off_ck{step}")
        on = results.get(f"dct_on_ck{step}")
        if not (off and on):
            continue
        d = lambda key, scale=1: (on[key] - off[key]) * scale
        print(f"ck{step:<4} | {d('sr_old',100):+8.1f}pp | {d('close_5',100):+8.1f}pp | {d('close_2',100):+8.1f}pp | {d('minLat_med'):+7.2f}mm | {d('flat_med'):+7.2f}mm | {d('ang_med'):+6.2f}° | {d('safety'):+7.2f}mm")

    out = Path("/home/najo/NAS/VLANeXt/logs/dct_ablation/dct_ablation_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {kk: ({str(rk): rv for rk, rv in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in results.items()}, indent=2, default=str))
    print(f"\nJSON saved: {out}")


if __name__ == "__main__":
    main()
