"""Visualize keypoint dataset: overlay tip_uv (red) + trocar_uv (green) markers
on the saved frames. Save as a grid PNG.

Usage:
    python -m scripts.keypoint_viz \\
        --h5 dataset/keypoint/val_500.h5 --n 20 --out vqa_samples/keypoint_viz.png
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=5)
    args = p.parse_args()

    f = h5py.File(args.h5, "r")
    N = f["image"].shape[0]
    n = min(args.n, N)
    rng = np.random.default_rng(2026)
    idxs = sorted(rng.choice(N, size=n, replace=False).tolist())

    cell_w = cell_h = 256
    cols = args.cols
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows + 24 * rows), (32, 32, 32))
    draw_grid = ImageDraw.Draw(grid)

    for i, idx in enumerate(idxs):
        img = f["image"][idx]
        tip_u, tip_v = f["tip_uv"][idx]
        troc_u, troc_v = f["trocar_uv"][idx]
        lat = float(f["lateral_mm"][idx]) if "lateral_mm" in f else -1.0
        ang = float(f["angle_deg"][idx]) if "angle_deg" in f else -1.0

        pil = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(pil)
        # tip (red)
        tx, ty = int(tip_u * 256), int(tip_v * 256)
        draw.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], outline="red", width=2)
        # trocar (green) — only if in-frame; otherwise draw a small arrow at frame edge
        if 0 <= troc_u <= 1 and 0 <= troc_v <= 1:
            cx, cy = int(troc_u * 256), int(troc_v * 256)
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline="lime", width=2)
            draw.line([tx, ty, cx, cy], fill="yellow", width=1)
        else:
            # off-frame: clamp + arrow
            cx = int(min(max(troc_u * 256, 8), 248))
            cy = int(min(max(troc_v * 256, 8), 248))
            draw.line([tx, ty, cx, cy], fill="orange", width=2)
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline="orange", width=2)

        col = i % cols
        row = i // cols
        grid.paste(pil, (col * cell_w, row * (cell_h + 24)))
        label = f"#{idx} lat={lat:.1f}mm ang={ang:.1f}deg uv=({troc_u:.2f},{troc_v:.2f})"
        draw_grid.text((col * cell_w + 4, row * (cell_h + 24) + cell_h + 4), label, fill="white")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
