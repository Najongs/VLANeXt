#!/usr/bin/env python
"""Compare honest eval (--no-early-term) vs original early-term eval.

Reveals what was hidden by early termination on success.

Usage:
  python scripts/compare_honest_vs_origterm.py
"""
import sys
from pathlib import Path
import numpy as np

# Import metric computation from honest_metrics
sys.path.insert(0, str(Path(__file__).parent))
from honest_metrics import per_episode_metrics, aggregate


TARGETS = [
    ("qwen v2 ck1500",         "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v2_aggressive",     1500, 2),
    ("qwen v5 ck500 (hold)",   "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v5_holddata",        500, 2),
    ("qwen v8 ck500 (prec)",   "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v8_aux_extreme",     500, 2),
    ("qwen v8 ck1000 (safe)",  "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v8_aux_extreme",    1000, 2),
    ("qwen v9 ck500 (hold★)",  "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v9_gentle_hold",     500, 2),
    ("qwen v11 ck1500 (SOTA)", "VLANeXt_Qwen35_NEARGOAL", "reach_recover_v11_submm_tight",   1500, 2),
    ("chain lathold ck1000",   "VLANeXt_SigLIP2_NEARGOAL","lat_hold_v4_yneg_hold",           1000, 2),
    ("chain v5combo ck2000 e4","VLANeXt_SigLIP2_NEARGOAL","reach_recover_v5_combo",          2000, 4),
]

BASE = Path("/data/public/NAS/VLANeXt/checkpoints")

def get_dirs(arch, variant, step, execn):
    parent = BASE / arch / variant
    origterm = parent / f"align_eval_step{step}_exec{execn}_diff10_origterm"
    noET     = parent / f"align_eval_step{step}_exec{execn}_diff10_noET"
    # Fallback to non-suffixed
    if not origterm.exists():
        origterm = parent / f"align_eval_step{step}_exec{execn}_diff10"
    return origterm, noET


def fmt(v, fmt_str):
    if v is None or (isinstance(v, float) and (v != v)):
        return "  -  "
    return fmt_str.format(v)


def main():
    print("# Honest (--no-early-term) vs Original (early-term) — same ckpt, same seed")
    print("# Δ = honest − orig.  Positive Δ on Reach/Hold/Settled-pass = MORE measured. Negative Δ on Settled_lat/Safety = WORSE actual settle")
    print()
    fmt_hdr = f"{'ckpt':25} {'mode':6} {'n_steps':>7} | {'R5':>5} {'R2':>5} {'R1':>5} | {'mLat':>5} {'p25':>5} | {'Hold20':>6} {'Max30<2.5':>10} | {'Settled':>11} | {'Sf_p99':>6}"
    print(fmt_hdr)
    print("-" * 140)
    for label, arch, variant, step, execn in TARGETS:
        orig_dir, noET_dir = get_dirs(arch, variant, step, execn)
        orig_eps = per_episode_metrics(orig_dir) if orig_dir.exists() else []
        noET_eps = per_episode_metrics(noET_dir) if noET_dir.exists() else []
        orig_m = aggregate(orig_eps) if orig_eps else None
        noET_m = aggregate(noET_eps) if noET_eps else None

        for mode, m in [("orig", orig_m), ("noET", noET_m)]:
            if m is None:
                print(f"{label:25} {mode:6} (no data)")
                continue
            line = (f"{label:25} {mode:6} "
                    f"{fmt(m['n_steps_med'], '{:5.0f}'):>7} | "
                    f"{fmt(m['reach5']*100,'{:>4.1f}%'):>5} "
                    f"{fmt(m['reach2']*100,'{:>4.1f}%'):>5} "
                    f"{fmt(m['reach1']*100,'{:>4.1f}%'):>5} | "
                    f"{fmt(m['min_lat_med'],'{:>4.2f}'):>5} "
                    f"{fmt(m['min_lat_p25'],'{:>4.2f}'):>5} | "
                    f"{fmt(m['holdSR']*100,'{:>4.1f}%'):>6} "
                    f"{fmt(m['settled_max_sub25']*100,'{:>8.1f}%'):>10} | "
                    f"{fmt(m['settled_lat_med'],'{:>4.2f}'):>5}±{fmt(m['settled_std_med'],'{:>4.2f}'):>5} | "
                    f"{fmt(m['settled_lat_p99'],'{:>5.2f}'):>6}")
            print(line)

        # Delta row
        if orig_m and noET_m:
            def d(k):
                return noET_m[k] - orig_m[k]
            line = (f"{'  Δ (honest − orig)':25} {'delta':6} "
                    f"{d('n_steps_med'):>+6.0f}  | "
                    f"{d('reach5')*100:>+4.1f} "
                    f"{d('reach2')*100:>+4.1f} "
                    f"{d('reach1')*100:>+4.1f}  | "
                    f"{d('min_lat_med'):>+4.2f} "
                    f"{d('min_lat_p25'):>+4.2f}  | "
                    f"{d('holdSR')*100:>+4.1f}% "
                    f"{d('settled_max_sub25')*100:>+8.1f}%   | "
                    f"{d('settled_lat_med'):>+4.2f}±{d('settled_std_med'):>+4.2f}  | "
                    f"{d('settled_lat_p99'):>+5.2f}")
            print(line)
        print()


if __name__ == "__main__":
    main()
