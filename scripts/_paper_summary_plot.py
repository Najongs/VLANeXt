"""Generate paper-ready summary plots for the champion model.
Usage:  python -m scripts._paper_summary_plot
Output: docs/paper_summary.png + docs/paper_summary.md
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("/data/public/NAS/VLANeXt/logs")
OUT_DIR = Path("/data/public/NAS/VLANeXt/docs")
OUT_DIR.mkdir(exist_ok=True)


def parse_log(path):
    """Returns list of dicts per episode: dist, lateral, angle, min_dist, sensor, success."""
    if not Path(path).exists():
        return []
    eps = []
    pat = re.compile(
        r"Episode\s+(\d+)/(\d+)\s+\|\s+(SUCCESS\[\w+\]|FAIL)\s+\|\s+Steps:\s+(\d+)\s+\|\s+"
        r"SR:\s+[\d.]+%\s+\([\d/]+\)\s+\|\s+dist=([\d.]+)mm\s+lateral=([\d.]+)mm\s+"
        r"angle=([\d.]+)deg\s+min_dist=([\d.]+)mm\s+sensor=(-?[\d.]+)mm"
    )
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        eps.append({
            "ep": int(m.group(1)),
            "total": int(m.group(2)),
            "status": m.group(3),
            "steps": int(m.group(4)),
            "dist": float(m.group(5)),
            "lateral": float(m.group(6)),
            "angle": float(m.group(7)),
            "min_dist": float(m.group(8)),
            "sensor": float(m.group(9)),
        })
    return eps


LOGS = {
    "12ep (max=250)": "eval_realistic12_HARD_cotrain_1k.log",
    "12ep (max=400)": "eval_realistic12_HARD_cotrain_1k_SCALEoff_max400.log",
    "24ep (max=250)": "eval_realistic24_HARD_cotrain_1k.log",
    "90ep (max=250)": "eval_realistic30_HARD_cotrain_1k.log",
}

results = {}
for label, fname in LOGS.items():
    eps = parse_log(LOG_DIR / fname)
    if not eps:
        print(f"[WARN] missing {fname}")
        continue
    n_succ = sum(1 for e in eps if e["status"].startswith("SUCCESS"))
    n_tot = len(eps)
    lat_succ = [e["lateral"] for e in eps if e["status"].startswith("SUCCESS")]
    md_all = [e["min_dist"] for e in eps]
    lat_all = [e["lateral"] for e in eps]
    ang_succ = [e["angle"] for e in eps if e["status"].startswith("SUCCESS")]
    results[label] = {
        "n_succ": n_succ, "n_tot": n_tot, "sr": 100*n_succ/n_tot,
        "lat_succ": lat_succ, "md_all": md_all, "lat_all": lat_all,
        "ang_succ": ang_succ, "eps": eps,
    }
    print(f"{label}: SR={100*n_succ/n_tot:.1f}% ({n_succ}/{n_tot})  "
          f"lat_med(succ)={np.median(lat_succ) if lat_succ else 0:.2f}mm  "
          f"md_min={min(md_all):.2f}mm")

# === FIGURE ===
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# Plot 1: SR bar by grid
ax1 = fig.add_subplot(gs[0, 0])
labels = list(results.keys())
srs = [results[k]["sr"] for k in labels]
colors = ["#5b9bd5", "#5bd575", "#ed7d31", "#7030a0"]
bars = ax1.bar(range(len(labels)), srs, color=colors)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("Success Rate (%)")
ax1.set_ylim(0, 100)
ax1.set_title("Champion (HARD_cotrain/1k) — SR across grids", fontsize=11, fontweight="bold")
for bar, sr in zip(bars, srs):
    ax1.text(bar.get_x()+bar.get_width()/2, sr+1.5, f"{sr:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax1.axhline(50, ls=":", color="gray", alpha=0.5)
ax1.grid(axis="y", alpha=0.3)

# Plot 2: lateral histogram (90ep all episodes)
ax2 = fig.add_subplot(gs[0, 1])
if "90ep (max=250)" in results:
    eps90 = results["90ep (max=250)"]["eps"]
    lat_succ = [e["lateral"] for e in eps90 if e["status"].startswith("SUCCESS")]
    lat_fail = [e["lateral"] for e in eps90 if e["status"] == "FAIL"]
    bins = np.linspace(0, 30, 31)
    ax2.hist([lat_succ, lat_fail], bins=bins, stacked=True,
             label=[f"SUCCESS (n={len(lat_succ)})", f"FAIL (n={len(lat_fail)})"],
             color=["#70ad47", "#c00000"], alpha=0.85)
    ax2.axvline(5, color="black", ls="--", lw=1, label="success thr (5mm)")
    ax2.axvline(3, color="gray", ls=":", lw=1, label="strict (3mm)")
    ax2.set_xlabel("Final lateral error (mm)")
    ax2.set_ylabel("# Episodes")
    ax2.set_title("90ep — Lateral error distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

# Plot 3: min_dist histogram (90ep)
ax3 = fig.add_subplot(gs[0, 2])
if "90ep (max=250)" in results:
    md_succ = [e["min_dist"] for e in eps90 if e["status"].startswith("SUCCESS")]
    md_fail = [e["min_dist"] for e in eps90 if e["status"] == "FAIL"]
    bins = np.linspace(0, 30, 31)
    ax3.hist([md_succ, md_fail], bins=bins, stacked=True,
             label=[f"SUCCESS (n={len(md_succ)})", f"FAIL (n={len(md_fail)})"],
             color=["#70ad47", "#c00000"], alpha=0.85)
    ax3.axvline(5, color="black", ls="--", lw=1, label="md<5mm")
    ax3.axvline(1, color="purple", ls=":", lw=1.5, label="sub-mm")
    ax3.set_xlabel("Min distance reached (mm)")
    ax3.set_ylabel("# Episodes")
    ax3.set_title("90ep — Closest approach (min_dist)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

# Plot 4: angle distribution (success only)
ax4 = fig.add_subplot(gs[1, 0])
if "90ep (max=250)" in results:
    ang_succ_90 = [e["angle"] for e in eps90 if e["status"].startswith("SUCCESS")]
    ax4.hist(ang_succ_90, bins=np.linspace(0, 20, 21), color="#5b9bd5", alpha=0.85,
             edgecolor="black", lw=0.5)
    ax4.axvline(10, color="black", ls="--", lw=1, label="success thr (10°)")
    ax4.axvline(np.median(ang_succ_90), color="red", ls=":", lw=1.5,
                label=f"median={np.median(ang_succ_90):.1f}°")
    ax4.set_xlabel("Final angle error (deg)")
    ax4.set_ylabel("# Successful episodes")
    ax4.set_title("90ep — Angle error (SUCCESS only)", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

# Plot 5: training progression baseline → HARD → step-bump
ax5 = fig.add_subplot(gs[1, 1])
prog_x = ["HOLDv2\n(baseline)", "HARD_cotrain\n(data)", "+ step\nbump"]
prog_y = [58.3, 66.7, 75.0]
prog_y_24 = [54.0, 75.0, 75.0]  # 24ep numbers (step bump same as base for 24ep)
ax5.bar(np.arange(3) - 0.18, prog_y, width=0.35, label="12ep",
        color="#5b9bd5", alpha=0.9)
ax5.bar(np.arange(3) + 0.18, prog_y_24, width=0.35, label="24ep",
        color="#ed7d31", alpha=0.9)
ax5.set_xticks(range(3))
ax5.set_xticklabels(prog_x, fontsize=9)
ax5.set_ylabel("SR (%)")
ax5.set_ylim(0, 100)
ax5.set_title("Progression toward champion", fontsize=11, fontweight="bold")
for i, (a, b) in enumerate(zip(prog_y, prog_y_24)):
    ax5.text(i-0.18, a+1.5, f"{a:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax5.text(i+0.18, b+1.5, f"{b:.0f}", ha="center", fontsize=9, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(axis="y", alpha=0.3)

# Plot 6: precision metric matrix
ax6 = fig.add_subplot(gs[1, 2])
metric_labels = ["SR", "lat<5", "lat<3", "ang<10°", "ang<5°", "md<8"]
# 12ep baseline:
values = []
for label in ["12ep (max=250)", "12ep (max=400)"]:
    if label not in results:
        continue
    eps = results[label]["eps"]
    n = len(eps)
    sr = 100 * sum(1 for e in eps if e["status"].startswith("SUCCESS")) / n
    lat5 = 100 * sum(1 for e in eps if e["lateral"] < 5) / n
    lat3 = 100 * sum(1 for e in eps if e["lateral"] < 3) / n
    ang10 = 100 * sum(1 for e in eps if e["angle"] < 10) / n
    ang5 = 100 * sum(1 for e in eps if e["angle"] < 5) / n
    md8 = 100 * sum(1 for e in eps if e["min_dist"] < 8) / n
    values.append([sr, lat5, lat3, ang10, ang5, md8])
if len(values) == 2:
    x = np.arange(len(metric_labels))
    ax6.bar(x - 0.18, values[0], width=0.35, label="max=250", color="#5b9bd5", alpha=0.9)
    ax6.bar(x + 0.18, values[1], width=0.35, label="max=400", color="#70ad47", alpha=0.9)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metric_labels, rotation=15, fontsize=9)
    ax6.set_ylabel("%")
    ax6.set_ylim(0, 100)
    ax6.set_title("12ep — Precision metric matrix", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)

fig.suptitle("HARD_cotrain/1k Champion — Performance Summary",
             fontsize=14, fontweight="bold", y=0.995)
out_png = OUT_DIR / "paper_summary.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_png}")

# === MARKDOWN summary ===
md_lines = ["# Champion Model — Performance Summary",
            "",
            "**ckpt**: `b24_ft10mm_HARD_cotrain/checkpoint_1000.pt`",
            "**Pipeline**: VLA + KP servo×3 + sensor sweep + polish",
            "",
            "## SR across grids", ""]
md_lines.append("| Grid | n | SR | lat_med(succ) | min(min_dist) |")
md_lines.append("|---|---|---|---|---|")
for label, r in results.items():
    lat_med = np.median(r["lat_succ"]) if r["lat_succ"] else 0
    md_min = min(r["md_all"]) if r["md_all"] else 0
    md_lines.append(f"| {label} | {r['n_tot']} | **{r['sr']:.1f}%** | "
                    f"{lat_med:.2f}mm | {md_min:.2f}mm |")
(OUT_DIR / "paper_summary.md").write_text("\n".join(md_lines))
print(f"Saved: {OUT_DIR / 'paper_summary.md'}")
