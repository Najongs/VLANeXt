"""Sanity check: approach phase center crop이 phantom/trocar 보여주는가?

목적: 옵션 C (approach+align 모두 cropped로 cotrain) 결정 전에
approach data 여러 episodes의 다양한 거리에서 center crop 결과 확인.

판정:
  - 거리별로 phantom/trocar 가시성 측정 (어디서 사라지나)
  - 빈 화면 (empty floor) 비율 quantify
"""
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import glob

CROP_WIN = 256
CROP_TGT = 512
OUT = Path("/data/public/NAS/VLANeXt/vqa_samples/approach_crop_sanity.png")


def center_crop(img, win=CROP_WIN, tgt=CROP_TGT):
    H, W = img.shape[:2]
    x0 = (W - win) // 2
    y0 = (H - win) // 2
    patch = img[y0:y0 + win, x0:x0 + win, :]
    return cv2.resize(patch, (tgt, tgt), interpolation=cv2.INTER_LINEAR)


def decode_jpeg(jpeg_data):
    buf = np.array(jpeg_data).flatten().astype(np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Sample 5 random approach_00 episodes
approach_paths = sorted(glob.glob("/data/public/NAS/VLANeXt/dataset/approach/approach_00/**/*.h5", recursive=True))
print(f"Total approach_00 eps: {len(approach_paths)}")
rng = np.random.RandomState(42)
sample_eps = rng.choice(approach_paths, 5, replace=False).tolist()

# Distance bins to show
DIST_BINS_MM = [50, 30, 20, 15, 10, 5]
LABEL_BINS = [f"~{d}mm" for d in DIST_BINS_MM]

# 5 episodes × 6 distance bins
fig, axes = plt.subplots(5, 6, figsize=(20, 18))
for ep_i, ep_path in enumerate(sample_eps):
    with h5py.File(ep_path, "r") as f:
        # Compute tip-to-trocar distance per frame (positions stored in mm already)
        tip = f["observations"]["needle_tip_pos"][:].astype(np.float32)
        entry = f["observations"]["trocar_entry_pos"][:].astype(np.float32)
        dists_mm = np.linalg.norm(tip - entry, axis=-1)
        # For each distance bin, find closest frame
        for bin_i, target_d in enumerate(DIST_BINS_MM):
            # find frame with dist closest to target
            idx = int(np.argmin(np.abs(dists_mm - target_d)))
            actual_d = dists_mm[idx]
            img = decode_jpeg(f["observations"]["images"]["tool_camera"][idx])
            cropped = center_crop(img)
            ax = axes[ep_i, bin_i]
            ax.imshow(cropped)
            color = "green" if abs(actual_d - target_d) < 3 else "red"
            ax.set_title(f"target {target_d}mm | actual {actual_d:.1f}mm",
                         fontsize=10, color=color)
            ax.axis('off')
            if bin_i == 0:
                ep_name = Path(ep_path).stem[:25]
                ax.text(-0.12, 0.5, f"ep{ep_i}\n{ep_name}", rotation=90,
                        transform=ax.transAxes, va='center', fontsize=9)

fig.suptitle("approach_00 5 episodes × distance bins — center cropped views\n"
             "(red title = bin not actually covered by this episode)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT}")

# Also print stats: for ALL approach episodes, distance distribution at frame 0
print("\n=== approach_00 first-frame distance distribution ===")
sample_for_stats = rng.choice(approach_paths, 50, replace=False).tolist()
first_dists = []
for p in sample_for_stats:
    try:
        with h5py.File(p, "r") as f:
            tip0 = f["observations"]["needle_tip_pos"][0]
            entry0 = f["observations"]["trocar_entry_pos"][0]
            d = np.linalg.norm(tip0 - entry0)
            first_dists.append(d)
    except Exception:
        pass
first_dists = np.array(first_dists)
print(f"  n={len(first_dists)}, mean={first_dists.mean():.1f}mm, std={first_dists.std():.1f}")
print(f"  min={first_dists.min():.1f}, p25={np.percentile(first_dists, 25):.1f}, "
      f"median={np.median(first_dists):.1f}, p75={np.percentile(first_dists, 75):.1f}, "
      f"max={first_dists.max():.1f}")
print(f"  >30mm: {(first_dists > 30).mean():.1%}, "
      f">20mm: {(first_dists > 20).mean():.1%}, "
      f">10mm: {(first_dists > 10).mean():.1%}")
