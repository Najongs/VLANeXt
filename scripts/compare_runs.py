"""Compare multiple eval logs side by side as a markdown table.

Usage:
    python -m scripts.compare_runs \
        "champion: outputs/eval_<champion>/sim_eval_align_only.log" \
        "v2/3000: outputs/eval_<v2_3k>/sim_eval_align_only.log" \
        "v2/5000: outputs/eval_<v2_5k>/sim_eval_align_only.log" \
        "ACT/15k: outputs/eval_<act_15k>/sim_eval_align_only.log"

Outputs a markdown block ready to paste into EXPERIMENTS_fine_align.md.
"""
import re
import sys
from pathlib import Path

PAT = re.compile(
    r"Episode\s+(\d+)/(\d+).*?\|\s+(\S+)\s+\|.*?dist=([-\d.]+)mm\s+lateral=([-\d.]+)mm\s+"
    r"angle=([-\d.]+)deg\s+min_dist=([-\d.]+)mm\s+sensor=([-\d.]+)"
)


def parse(path):
    eps = []
    with open(path) as f:
        for ln in f:
            m = PAT.search(ln)
            if not m:
                continue
            _, _, status, dist, lat, ang, mind, sen = m.groups()
            eps.append({
                "status": status,
                "dist_mm": float(dist),
                "lat_mm": float(lat),
                "angle_deg": float(ang),
                "min_dist_mm": float(mind),
                "true_align_err_mm": abs(float(mind) - 10.0),  # min_dist 기대값 10 (retreat)
                "sensor": float(sen),
            })
    return eps


def metrics(eps):
    if not eps:
        return {}
    n = len(eps)
    sr = 100.0 * sum(1 for e in eps if e["status"].startswith("SUCCESS")) / n
    avg = lambda key: sum(e[key] for e in eps) / n
    rate_lt = lambda key, thr: 100.0 * sum(1 for e in eps if e[key] < thr) / n
    return {
        "N": n,
        "SR%": sr,
        "lat<5%": rate_lt("lat_mm", 5),
        "lat<3%": rate_lt("lat_mm", 3),
        "min_d<5%": rate_lt("min_dist_mm", 5),
        "true_err_mm": avg("true_align_err_mm"),
        "mean_min_d_mm": avg("min_dist_mm"),
        "mean_|ang|_deg": sum(abs(e["angle_deg"]) for e in eps) / n,
    }


def main():
    rows = []
    for arg in sys.argv[1:]:
        if ":" in arg:
            label, path = arg.split(":", 1)
            label = label.strip()
            path = path.strip()
        else:
            label = Path(arg).parent.name
            path = arg
        eps = parse(path)
        m = metrics(eps)
        if not m:
            print(f"# WARN empty: {path}", file=sys.stderr)
            continue
        rows.append((label, m))

    if not rows:
        print("No data parsed.", file=sys.stderr)
        sys.exit(1)

    keys = ["N", "SR%", "lat<5%", "lat<3%", "min_d<5%", "true_err_mm", "mean_min_d_mm", "mean_|ang|_deg"]
    print("| Run | " + " | ".join(keys) + " |")
    print("|---" + "|---:" * len(keys) + "|")
    for label, m in rows:
        cells = []
        for k in keys:
            v = m[k]
            if k == "N":
                cells.append(f"{int(v)}")
            elif "%" in k:
                cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
