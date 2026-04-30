#!/usr/bin/env python3
"""
Compare real-replay align dataset with sim align dataset.

Quantifies:
  • action_real vs action_sim (within-episode jitter)
  • action distributions (real vs sim)
  • ee_pose trajectory shapes (overlay)
  • episode length distribution

Usage:
    python -m scripts.compare_replay_vs_sim \
        --real-dir dataset/collected_data_real_replay \
        --sim-dir dataset/fine_align/fine_align_04/collected_data_merged \
        --out-dir dataset/compare_replay_vs_sim \
        --sim-sample 200

python -m scripts.compare_replay_vs_sim \
    --real-dir dataset/collected_data_real \
    --sim-dir dataset/collected_data_real \
    --out-dir dataset/compare_replay_vs_sim \
    --sim-sample 14
    
"""
import argparse
import glob
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_episodes(pattern, limit=None):
    paths = sorted(glob.glob(str(pattern)))
    if limit is not None:
        rng = np.random.default_rng(0)
        if len(paths) > limit:
            paths = list(rng.choice(paths, size=limit, replace=False))
    eps = []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                ep = {
                    "path": p,
                    "action": f["action"][:],
                    "ee_pose": f["observations/ee_pose"][:],
                    "qpos": f["observations/qpos"][:],
                    "T": f["action"].shape[0],
                }
                if "action_sim" in f:
                    ep["action_sim"] = f["action_sim"][:]
                if "metadata/perturb_xyz_mm" in f:
                    ep["perturb_xyz_mm"] = f["metadata/perturb_xyz_mm"][:]
                eps.append(ep)
        except Exception as e:
            print(f"  skip {p}: {e}")
    return eps


def _stack_actions(eps, key="action"):
    arrs = [e[key] for e in eps if key in e]
    return np.concatenate(arrs, axis=0) if arrs else np.empty((0, 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--out-dir", default="dataset/compare_replay_vs_sim")
    ap.add_argument("--sim-sample", type=int, default=200,
                    help="Random sample size from sim (default 200)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_eps = load_episodes(Path(args.real_dir) / "*.h5")
    sim_eps = load_episodes(Path(args.sim_dir) / "*.h5", limit=args.sim_sample)
    print(f"Real episodes: {len(real_eps)}")
    print(f"Sim  episodes: {len(sim_eps)}")
    if not real_eps or not sim_eps:
        raise SystemExit("Need both real and sim episodes.")

    # ── 1. Action distribution: real vs sim (per-axis hist) ─────────────────
    real_act = _stack_actions(real_eps, "action")
    sim_act = _stack_actions(sim_eps, "action")
    labels = ["dx_mm", "dy_mm", "dz_mm", "drx_rad", "dry_rad", "drz_rad", "grip"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, ax in enumerate(axes.ravel()[:7]):
        r = real_act[:, i]
        s = sim_act[:, i]
        lo = min(r.min(), s.min()); hi = max(r.max(), s.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(s, bins=bins, alpha=0.5, label=f"sim (n={len(s)})", color="steelblue", density=True)
        ax.hist(r, bins=bins, alpha=0.5, label=f"real (n={len(r)})", color="orange", density=True)
        ax.set_title(f"{labels[i]}\nreal μ={r.mean():.3f} σ={r.std():.3f}\n"
                     f"sim  μ={s.mean():.3f} σ={s.std():.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes.ravel()[7].axis("off")
    fig.suptitle("Action distribution — real vs sim (per axis)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "action_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out/'action_dist.png'}")

    # ── 2. Within-episode: action vs action_sim (jitter measure) ─────────────
    if any("action_sim" in e for e in real_eps):
        diffs = []
        for e in real_eps:
            if "action_sim" not in e:
                continue
            a = e["action"][:, :6]; b = e["action_sim"][:, :6]
            mask = ~np.isnan(b).any(axis=-1)
            diffs.append((a[mask] - b[mask]))
        if diffs:
            d = np.concatenate(diffs, axis=0)
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            for i, ax in enumerate(axes.ravel()):
                ax.hist(d[:, i], bins=60, color="purple", alpha=0.7)
                ax.set_title(f"{labels[i]} : real - sim_planned\n"
                             f"μ={d[:,i].mean():.3f}  σ={d[:,i].std():.3f}  "
                             f"|max|={np.abs(d[:,i]).max():.3f}", fontsize=9)
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                ax.grid(alpha=0.3)
            fig.suptitle("Within real episode: action - action_sim "
                         "(non-zero = real didn't follow plan exactly)",
                         fontsize=13)
            plt.tight_layout()
            plt.savefig(out / "real_minus_sim_action.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → {out/'real_minus_sim_action.png'}")

            # Plot one episode timeseries
            e0 = next((e for e in real_eps if "action_sim" in e), None)
            if e0 is not None:
                fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
                t = np.arange(e0["T"])
                for i, ax in enumerate(axes.ravel()):
                    ax.plot(t, e0["action"][:, i], label="real", color="orange")
                    ax.plot(t, e0["action_sim"][:, i], label="sim_planned",
                            color="steelblue", linestyle="--")
                    ax.set_title(labels[i], fontsize=9)
                    ax.legend(fontsize=7); ax.grid(alpha=0.3)
                axes[-1, 0].set_xlabel("step")
                axes[-1, 1].set_xlabel("step")
                fig.suptitle(f"Per-step action timeseries (1 real episode)\n{Path(e0['path']).name}",
                             fontsize=11)
                plt.tight_layout()
                plt.savefig(out / "ep0_action_timeseries.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  → {out/'ep0_action_timeseries.png'}")

    # ── 3. EE-pose trajectories overlay ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    proj = [(0, 1, "X (mm)", "Y (mm)", "XY"),
            (0, 2, "X (mm)", "Z (mm)", "XZ"),
            (1, 2, "Y (mm)", "Z (mm)", "YZ")]
    for ax, (a, b, xl, yl, ttl) in zip(axes, proj):
        for e in sim_eps[:50]:
            ax.plot(e["ee_pose"][:, a], e["ee_pose"][:, b],
                    color="steelblue", alpha=0.15, linewidth=0.7)
        for e in real_eps:
            ax.plot(e["ee_pose"][:, a], e["ee_pose"][:, b],
                    color="orange", alpha=0.9, linewidth=1.4)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"EE trajectory — {ttl}\nsim(blue, n=50) vs real(orange, n={len(real_eps)})",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out / "ee_traj_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out/'ee_traj_overlay.png'}")

    # ── 4. Episode length distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sim_T = np.array([e["T"] for e in sim_eps])
    real_T = np.array([e["T"] for e in real_eps])
    bins = np.linspace(min(sim_T.min(), real_T.min()),
                       max(sim_T.max(), real_T.max()), 30)
    ax.hist(sim_T, bins=bins, alpha=0.5, label=f"sim  μ={sim_T.mean():.1f}",
            color="steelblue", density=True)
    ax.hist(real_T, bins=bins, alpha=0.5, label=f"real μ={real_T.mean():.1f}",
            color="orange", density=True)
    ax.set_xlabel("episode length (frames)")
    ax.set_ylabel("density")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Episode length distribution")
    plt.tight_layout()
    plt.savefig(out / "episode_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out/'episode_length.png'}")

    # ── 5. Numeric summary ──────────────────────────────────────────────────
    summary = []
    summary.append(f"Real episodes: {len(real_eps)}  (T: μ={real_T.mean():.1f} ± {real_T.std():.1f})")
    summary.append(f"Sim  episodes: {len(sim_eps)}  (T: μ={sim_T.mean():.1f} ± {sim_T.std():.1f})")
    summary.append("")
    summary.append("Per-axis action stats (μ ± σ, |max|):")
    summary.append(f"{'axis':<10}{'sim':<35}{'real':<35}")
    for i, lb in enumerate(labels):
        s, r = sim_act[:, i], real_act[:, i]
        summary.append(
            f"  {lb:<8}"
            f"{s.mean():+.3f} ± {s.std():.3f} (|max|={np.abs(s).max():.3f})    "
            f"{r.mean():+.3f} ± {r.std():.3f} (|max|={np.abs(r).max():.3f})"
        )
    if any("action_sim" in e for e in real_eps):
        diffs = []
        for e in real_eps:
            if "action_sim" not in e: continue
            a = e["action"][:, :6]; b = e["action_sim"][:, :6]
            mask = ~np.isnan(b).any(axis=-1)
            diffs.append(a[mask] - b[mask])
        d = np.concatenate(diffs, axis=0)
        summary.append("")
        summary.append("Real - sim_planned (within-episode tracking error):")
        for i, lb in enumerate(labels[:6]):
            summary.append(f"  {lb:<8} μ={d[:,i].mean():+.4f}  σ={d[:,i].std():.4f}  "
                           f"RMS={np.sqrt((d[:,i]**2).mean()):.4f}  "
                           f"|max|={np.abs(d[:,i]).max():.4f}")
    text = "\n".join(summary)
    print()
    print(text)
    with open(out / "summary.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
