"""Robot perturbation visualization — 4 separate PNGs (high res).

Outputs:
  vqa_samples/robot_perturb_X_views.png (1 row × 7 col)
  vqa_samples/robot_perturb_Y_views.png (1 row × 7 col)
  vqa_samples/robot_perturb_Z_views.png (1 row × 7 col)
  vqa_samples/robot_perturb_angle_views.png (2 rows × 7 col: Track A 5° + Track B 15°)

각 panel = NEARGOAL HDF5 에피소드의 첫 frame (post phantom-align + robot-perturb state).
"""
from __future__ import annotations
import glob
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DATA_A = "/data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_eval_match_v2/collected_data_merged"
DATA_B = "/data/public/NAS/VLANeXt/dataset/fine_align/NEARGOAL_angle_only_v2/collected_data_merged"
OUT_DIR = Path("/data/public/NAS/VLANeXt/vqa_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_meta(data_dir, n_samples=1500):
    files = sorted(glob.glob(f"{data_dir}/*.h5"))[:n_samples]
    metas = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                metas.append({
                    "x": float(h["metadata/perturb_xyz_mm"][0]),
                    "y": float(h["metadata/perturb_xyz_mm"][1]),
                    "z": float(h["metadata/perturb_xyz_mm"][2]),
                    "angle": float(h["metadata/perturb_angle_deg"][()]),
                    "path": f,
                })
        except OSError:
            continue
    return metas


def find_closest(metas, key, target, near_keys=None, used_paths=None):
    candidates = metas
    if near_keys:
        for k, tol in near_keys.items():
            candidates = [m for m in candidates if abs(m[k]) < tol]
    if used_paths:
        candidates = [m for m in candidates if m["path"] not in used_paths]
    if not candidates:
        return None
    return min(candidates, key=lambda m: abs(m[key] - target))


def extract_first_frame(h5_path, cam_name="tool_camera"):
    with h5py.File(h5_path, "r") as h:
        jpeg_bytes = bytes(h[f"observations/images/{cam_name}"][0])
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_single_row_png(metas, key, bins, near_keys, title, out_path, unit="mm"):
    """1 row × len(bins) PNG, 각 panel 큰 사이즈."""
    n_cols = len(bins)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]
    used = set()
    for col_i, target in enumerate(bins):
        m = find_closest(metas, key, target, near_keys, used)
        ax = axes[col_i]
        if m is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
            ax.set_title(f"target {key}={target:+}{unit}", fontsize=11)
        else:
            used.add(m["path"])
            img = extract_first_frame(m["path"])
            ax.imshow(img)
            t = f"{key}={m[key]:+.1f}{unit}"
            # Other dims info (only if non-trivial)
            others = []
            for k in ["x", "y", "z", "angle"]:
                if k != key and abs(m[k]) > 0.5:
                    u_other = "°" if k == "angle" else "mm"
                    others.append(f"{k}={m[k]:+.1f}{u_other}")
            if others:
                t += "  (" + ", ".join(others) + ")"
            ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_angle_png(metas_A, metas_B, bins_small, bins_large, out_path):
    """2 rows × 7 cols: Track A small + Track B large."""
    n_cols = max(len(bins_small), len(bins_large))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

    for row_i, (metas, bins, label, near) in enumerate([
        (metas_A, bins_small, "Angle perturb small (Track A, ±5°)", {"x": 1.5, "y": 1.5, "z": 1.5}),
        (metas_B, bins_large, "Angle perturb large (Track B, ±15°, XY=0)", None),
    ]):
        used = set()
        for col_i, target in enumerate(bins):
            m = find_closest(metas, "angle", target, near, used)
            ax = axes[row_i, col_i]
            if m is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
                ax.set_title(f"target angle={target:+}°", fontsize=11)
            else:
                used.add(m["path"])
                img = extract_first_frame(m["path"])
                ax.imshow(img)
                t = f"angle={m['angle']:+.1f}°"
                others = []
                for k in ["x", "y", "z"]:
                    if abs(m[k]) > 0.5:
                        others.append(f"{k}={m[k]:+.1f}mm")
                if others:
                    t += "  (" + ", ".join(others) + ")"
                ax.set_title(t, fontsize=11)
            ax.axis("off")
        # Row label
        axes[row_i, 0].text(
            -0.18, 0.5, label,
            rotation=90, transform=axes[row_i, 0].transAxes,
            va="center", ha="center", fontsize=12, fontweight="bold",
        )

    fig.suptitle("Robot Angle Perturbation Views", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Collecting Track A metadata...")
    metas_A = collect_meta(DATA_A, n_samples=1500)
    print(f"  {len(metas_A)} eps from Track A")
    print("Collecting Track B metadata...")
    metas_B = collect_meta(DATA_B, n_samples=800)
    print(f"  {len(metas_B)} eps from Track B")

    bins_5mm = [-5, -3, -1.5, 0, 1.5, 3, 5]
    bins_angle_small = [-5, -3, -1.5, 0, 1.5, 3, 5]
    bins_angle_large = [-15, -10, -5, 0, 5, 10, 15]

    make_single_row_png(
        metas_A, "x", bins_5mm,
        near_keys={"y": 1.5, "z": 1.5, "angle": 1.5},
        title="Robot X perturbation (Track A, near y=z=angle=0)",
        out_path=OUT_DIR / "robot_perturb_X_views.png",
    )
    make_single_row_png(
        metas_A, "y", bins_5mm,
        near_keys={"x": 1.5, "z": 1.5, "angle": 1.5},
        title="Robot Y perturbation (Track A, near x=z=angle=0)",
        out_path=OUT_DIR / "robot_perturb_Y_views.png",
    )
    make_single_row_png(
        metas_A, "z", bins_5mm,
        near_keys={"x": 1.5, "y": 1.5, "angle": 1.5},
        title="Robot Z perturbation (Track A, near x=y=angle=0)",
        out_path=OUT_DIR / "robot_perturb_Z_views.png",
    )
    make_angle_png(
        metas_A, metas_B, bins_angle_small, bins_angle_large,
        out_path=OUT_DIR / "robot_perturb_angle_views.png",
    )


if __name__ == "__main__":
    main()
