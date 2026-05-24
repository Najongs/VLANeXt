#!/usr/bin/env python
"""Baseline matrix analyzer — ACT / DP / ConvNeXt / SigLIP2 / DINOv3 + champion.

Per-baseline metrics combining CSV summary + per-step npz trajectories:
- SR_old (final 3D < 5mm)
- close_5 / close_2 (final_lateral < 5/2mm)
- holdSR (lateral < 2.5mm for ≥20 contiguous steps — minLat champion convention)
- min_lat_med (per-ep min lateral, median across grid)
- min_dist3D_med (per-ep min 3D distance, median)
- final_lat_med
- ang_at_near (median angle when final_3D<5mm)
- safety (p99 final_lateral — worst-case end state)
- per-region SR_old (y=-25 / 0 / +25)
"""
import csv
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path("/data/public/NAS/VLANeXt/checkpoints")

TARGETS = [
    ("ACT (ResNet18+CVAE+T)",        ROOT / "ACT_baseline_align/align_eval_step30000_exec1_diff10", 1, "retreat=2, 30k step, exec=1"),
    ("DP  (ResNet18+CondUnet1D)",    ROOT / "DP_baseline_align/align_eval_step30000_exec1_diff10",  1, "retreat=2, 30k step, exec=1"),
    ("ConvNeXt fresh 1500",          ROOT / "VLANeXt_ConvNeXt_unfreeze/v5b/align_eval_step1500_exec2_diff10_shard0_retreat2", 2, "1500 step fresh"),
    ("SigLIP2-so400m fresh 1500",    ROOT / "VLANeXt_SigLIP2_baseline/v1/align_eval_step1500_exec2_diff10", 2, "1500 step fresh"),
    ("DINOv3-ViT-L/16 fresh 1500",   ROOT / "VLANeXt_DINOv3_baseline/v1/align_eval_step1500_exec2_diff10",  2, "1500 step fresh"),
    ("ConvNeXt fresh 5000",          ROOT / "VLANeXt_ConvNeXt_long5k/v1/align_eval_step5000_exec2_diff10", 2, "5000 step fresh, lr 5e-6"),
    ("DINOv3-ViT-L/16 fresh 5000",   ROOT / "VLANeXt_DINOv3_long5k/v1/align_eval_step5000_exec2_diff10",  2, "5000 step fresh, lr 5e-6"),
    ("ConvNeXt fresh 20000",         ROOT / "VLANeXt_ConvNeXt_long20k/v1/align_eval_step20000_exec2_diff10", 2, "20000 step fresh, lr 5e-6 (matched SigLIP2 chain budget)"),
    ("DINOv3-ViT-L/16 fresh 20000",  ROOT / "VLANeXt_DINOv3_long20k/v1/align_eval_step20000_exec2_diff10",  2, "20000 step fresh, lr 5e-6 (matched SigLIP2 chain budget)"),
    ("SigLIP2 + dist only (no hold L/D)", ROOT / "VLANeXt_SigLIP2_NEARGOAL/v2_dual_lr1e6/align_eval_step1000_exec2_diff10_SR74.07", 2, "SAME arch, NO aux_hold/lateral, NO yneg_hold data"),
    ("SigLIP2 champion (full chain)", ROOT / "VLANeXt_SigLIP2_NEARGOAL/lat_hold_v4_yneg_hold/align_eval_step1000_exec2_diff10_SR70.37", 2, "production champ: + aux_hold + aux_lat + yneg_hold/perfect_strict"),
    ("Ours reach_recover v1 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v1/align_eval_step1000_exec2_diff10", 2, "v1: champion + yneg_v1 + ypos_v1, lr 5e-7"),
    ("Ours reach_recover v1 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v1/align_eval_step1500_exec2_diff10", 2, "v1: lr 5e-7"),
    ("Ours reach_recover v2 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v2_aggressive/align_eval_step1500_exec2_diff10", 2, "v2 aggressive: lr 1e-6 (2x)"),
    ("Ours reach_recover v2 ck2000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v2_aggressive/align_eval_step2000_exec2_diff10", 2, "v2 aggressive: lr 1e-6 (2x)"),
    ("Ours reach_recover v2 ck3000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v2_aggressive/align_eval_step3000_exec2_diff10", 2, "v2 aggressive: lr 1e-6 (2x)"),
    ("Ours reach_recover v3 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v3_softhold/align_eval_step1000_exec2_diff10", 2, "v3 softhold: hold weights 0.15/0.25 (half)"),
    ("Ours reach_recover v3 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v3_softhold/align_eval_step1500_exec2_diff10", 2, "v3 softhold: hold weights half"),
    ("Ours reach_recover v4 ck3000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v4_longer/align_eval_step3000_exec2_diff10", 2, "v4 longer: lr 1e-6, 5000 step (eval mid)"),
    ("Ours reach_recover v4 ck4000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v4_longer/align_eval_step4000_exec2_diff10", 2, "v4 longer"),
    ("Ours reach_recover v4 ck5000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v4_longer/align_eval_step5000_exec2_diff10", 2, "v4 longer: full 5000 step"),
    ("Ours reach_recover v5 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step1500_exec2_diff10", 2, "v5 combo: lr 1e-6 + softhold half"),
    ("Ours reach_recover v5 ck2000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step2000_exec2_diff10", 2, "v5 combo"),
    ("Ours reach_recover v5 ck2500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step2500_exec2_diff10", 2, "v5 combo: ck2500 (curve fill)"),
    ("Ours reach_recover v5 ck3000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step3000_exec2_diff10", 2, "v5 combo: full"),
    ("Ours reach_recover v6 ck500",  ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v6_v5consol/align_eval_step500_exec2_diff10", 2, "v6: v5 ck2000 base + lr 5e-7 + 1000 step consol"),
    ("Ours reach_recover v6 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v6_v5consol/align_eval_step1000_exec2_diff10", 2, "v6 consol full"),
    ("Ours reach_recover v7 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v7_pushlr/align_eval_step1000_exec2_diff10", 2, "v7 pushlr: lr 2e-6 (4x)"),
    ("Ours reach_recover v7 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v7_pushlr/align_eval_step1500_exec2_diff10", 2, "v7 pushlr"),
    ("Ours reach_recover v7 ck2000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v7_pushlr/align_eval_step2000_exec2_diff10", 2, "v7 pushlr: full"),
    ("Ours reach_recover v8 ck500",  ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v8_gentle/align_eval_step500_exec2_diff10", 2, "v8: v5 ck2000 base + lr 2.5e-7 + 500 step (gentle consol)"),
    ("Ours reach_recover v9 ck500",  ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v9_v5push/align_eval_step500_exec2_diff10",  2, "v9: v5 ck2000 base + lr 1.5e-6 (mid)"),
    ("Ours reach_recover v9 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v9_v5push/align_eval_step1000_exec2_diff10", 2, "v9 mid"),
    ("Ours reach_recover v9 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v9_v5push/align_eval_step1500_exec2_diff10", 2, "v9 mid: full"),
    ("Ours v5 ck2000 exec=1", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step2000_exec1_diff10", 1, "v5 ck2000 inference: exec=1 (single-step)"),
    ("Ours v5 ck2000 exec=4", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/align_eval_step2000_exec4_diff10", 4, "v5 ck2000 inference: exec=4 (long chunk)"),
    ("Ours v10 ck500",  ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v10_yneg25/align_eval_step500_exec2_diff10", 2, "v10: v5 ck2000 base + yneg25_strict_v1 added"),
    ("Ours v10 ck1000", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v10_yneg25/align_eval_step1000_exec2_diff10", 2, "v10"),
    ("Ours v10 ck1500", ROOT / "VLANeXt_SigLIP2_NEARGOAL/reach_recover_v10_yneg25/align_eval_step1500_exec2_diff10", 2, "v10 full"),
    ("Qwen3.5-2B (with LM) 20k fresh", ROOT / "output_dir_v2_dual_finetune_qwen_20000step/align_eval_step20000_exec2_diff10", 2, "Qwen3.5-2B hybrid VL (linear+full attn) + ours diff head, fresh 20k step (with-LM ablation)"),
]


def per_ep_min_lat(eval_dir: Path):
    """For each traj_ep*.npz return (min_lateral, hold_ok). hold_ok = lateral<2.5 for 20+ contig steps."""
    out = []
    for f in sorted(eval_dir.glob("traj_ep*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            lat = np.asarray(d["lateral_mm"], dtype=float) if "lateral_mm" in d.files else np.asarray([])
            dist3 = np.asarray(d["dist_mm"], dtype=float) if "dist_mm" in d.files else np.asarray([])
            if lat.size == 0:
                continue
            min_lat = float(np.min(lat))
            min_d3 = float(np.min(dist3)) if dist3.size else float("nan")
            # 20-step contig hold check
            hold_ok = False
            run = 0
            for v in lat:
                if v < 2.5:
                    run += 1
                    if run >= 20:
                        hold_ok = True
                        break
                else:
                    run = 0
            out.append((min_lat, min_d3, hold_ok))
        except Exception as e:
            print(f"    skip {f.name}: {e}")
    return out


def aggregate(eval_dir: Path):
    csv_path = eval_dir / "metrics_summary.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    n = len(rows)

    fdist = [float(r["final_dist_mm"]) for r in rows]
    flat = [float(r["final_lateral_mm"]) for r in rows]
    fang = [float(r["final_angle_deg"]) for r in rows]

    sr_old = sum(1 for d in fdist if d < 5.0) / n
    close_5 = sum(1 for v in flat if v < 5.0) / n
    close_2 = sum(1 for v in flat if v < 2.0) / n
    flat_med = statistics.median(flat)
    near_ang = [a for a, d in zip(fang, fdist) if d < 5.0]
    ang_med = statistics.median(near_ang) if near_ang else float("nan")
    safety = sorted(flat)[int(0.99 * n) - 1] if n >= 10 else max(flat)

    # per-step from npz: min_lat per ep + hold success
    npz_data = per_ep_min_lat(eval_dir)
    if npz_data:
        min_lats = [x[0] for x in npz_data]
        min_d3s = [x[1] for x in npz_data if not np.isnan(x[1])]
        hold_ok = [x[2] for x in npz_data]
        min_lat_med = statistics.median(min_lats)
        min_d3_med = statistics.median(min_d3s) if min_d3s else float("nan")
        hold_sr = sum(hold_ok) / len(hold_ok)
    else:
        min_lat_med = min_d3_med = float("nan")
        hold_sr = float("nan")

    region = {-25: [], 0: [], 25: []}
    for r, d in zip(rows, fdist):
        try:
            py = round(float(r["perturb_y_mm"]))
        except (KeyError, ValueError):
            continue
        if py in region:
            region[py].append(d < 5.0)
    region_sr = {k: (sum(v) / len(v) if v else None, len(v)) for k, v in region.items()}

    return {
        "n": n, "sr_old": sr_old, "close_5": close_5, "close_2": close_2,
        "hold_sr": hold_sr, "min_lat_med": min_lat_med, "min_d3_med": min_d3_med,
        "flat_med": flat_med, "ang_med": ang_med, "safety": safety, "region": region_sr,
    }


def fmt_region(r, key):
    v, c = r[key]
    return f"{v*100:.0f}/{c}" if v is not None else f"-/{c}"


def main():
    rows_out = []
    for label, eval_dir, _exec, note in TARGETS:
        if not eval_dir.exists():
            print(f"  MISSING: {label} ({eval_dir})")
            continue
        agg = aggregate(eval_dir)
        if not agg:
            print(f"  EMPTY: {label}")
            continue
        rows_out.append((label, agg, note))
        print(f"  loaded ({agg['n']}): {label}")

    print("\n=== Baseline matrix (retreat=2, multi-metric, 27-cell grid) ===")
    hdr = (f"{'baseline':<32} | {'n':>3} | {'SR_old':>7} | {'close5':>7} | {'close2':>7} | "
           f"{'holdSR':>7} | {'min_lat':>8} | {'min_3D':>8} | {'finLat':>8} | {'ang°':>5} | "
           f"{'safety':>8} | y=-25 | y=0   | y=+25")
    print(hdr); print("-" * len(hdr))
    for label, v, note in rows_out:
        r = v["region"]
        print(f"{label:<32} | {v['n']:>3} | {v['sr_old']*100:>6.1f}% | {v['close_5']*100:>6.1f}% | {v['close_2']*100:>6.1f}% | "
              f"{v['hold_sr']*100:>6.1f}% | {v['min_lat_med']:>6.2f}mm | {v['min_d3_med']:>6.2f}mm | {v['flat_med']:>6.2f}mm | {v['ang_med']:>4.2f}° | "
              f"{v['safety']:>6.2f}mm | {fmt_region(r,-25):>5} | {fmt_region(r,0):>5} | {fmt_region(r,25):>5}")
    print("\nProtocol notes:")
    for label, _v, note in rows_out:
        print(f"  {label}: {note}")

    out_json = Path("/data/public/NAS/VLANeXt/logs/baseline_matrix/baseline_matrix_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([
        {"label": l, "note": n, **{k: ({str(rk): rv for rk, rv in vv.items()} if isinstance(vv, dict) else vv) for k, vv in agg.items()}}
        for l, agg, n in rows_out
    ], indent=2, default=str))
    print(f"\nJSON saved: {out_json}")


if __name__ == "__main__":
    main()
