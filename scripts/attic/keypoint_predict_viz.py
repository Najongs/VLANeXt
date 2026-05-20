"""Visualize keypoint head predictions vs GT on val frames.

For each sampled frame:
  - Red circle: tip GT (constant in tool-cam)
  - Green circle: trocar GT
  - Cyan circle: trocar PREDICTED
  - Yellow line: GT tip→trocar
  - Magenta line: GT trocar ↔ predicted trocar (= error vector)

Usage:
    python -m scripts.keypoint_predict_viz \
        --ckpt checkpoints/keypoint_trocar/cotrain_v1/head_best.pt \
        --h5 dataset/keypoint/real_val_2k.h5 \
        --n 20 --out vqa_samples/kp_pred_cotrain.png
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import SiglipVisionModel, SiglipImageProcessor


VISION_MODEL = "google/siglip2-so400m-patch16-512"


class KeypointHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, patch_tokens):
        feat = patch_tokens.mean(dim=1)
        out = self.mlp(feat)
        uv = torch.sigmoid(out[..., :2])
        vis_logit = out[..., 2]
        return uv, vis_logit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bucket", choices=["all", "near", "mid", "far"], default="all",
                   help="filter by lateral_mm bucket")
    p.add_argument("--offset-px", type=float, nargs=2, default=[0.0, 0.0],
                   help="Apply (du, dv) pixel offset to BOTH tip and trocar markers (e.g. -3 3 shifts markers down-left)")
    args = p.parse_args()

    device = "cuda"
    image_processor = SiglipImageProcessor.from_pretrained(VISION_MODEL)
    vision = SiglipVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16).to(device).eval()
    head = KeypointHead(vision.config.hidden_size).to(device).to(torch.bfloat16).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ckpt["head_state"])
    print(f"Loaded ckpt step={ckpt.get('step','?')} metrics={ckpt.get('metrics',{})}")

    f = h5py.File(args.h5, "r")
    N = f["image"].shape[0]
    lat = f["lateral_mm"][:]

    if args.bucket == "near":
        pool = np.where(lat < 5)[0]
    elif args.bucket == "mid":
        pool = np.where((lat >= 5) & (lat < 15))[0]
    elif args.bucket == "far":
        pool = np.where(lat >= 15)[0]
    else:
        pool = np.arange(N)

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(pool))
    idxs = sorted(rng.choice(pool, size=n, replace=False).tolist())

    # Batch predict.
    images = np.stack([f["image"][i] for i in idxs])
    pils = [Image.fromarray(im) for im in images]
    proc = image_processor(images=pils, return_tensors="pt")
    pv = proc["pixel_values"].to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        feats = vision(pixel_values=pv).last_hidden_state
        uv_pred, vis_logit = head(feats)
    uv_pred = uv_pred.float().cpu().numpy()
    vis_prob = torch.sigmoid(vis_logit).float().cpu().numpy()

    # Auto-detect image size from h5 (256 legacy, 512 current default)
    img_w = int(f["image"].shape[2])
    img_h = int(f["image"].shape[1])
    cell_w, cell_h = img_w, img_h
    cols = args.cols
    rows = (n + cols - 1) // cols
    label_h = 28
    grid = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows), (32, 32, 32))
    draw_grid = ImageDraw.Draw(grid)

    px_errs = []
    for i, idx in enumerate(idxs):
        img = images[i]
        tip_u, tip_v = f["tip_uv"][idx]
        troc_u_gt, troc_v_gt = f["trocar_uv"][idx]
        troc_u_pr, troc_v_pr = uv_pred[i]
        lateral = float(f["lateral_mm"][idx])
        vis_p = float(vis_prob[i])

        pil = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(pil)
        du, dv = args.offset_px
        tx, ty = int(tip_u * img_w + du), int(tip_v * img_h + dv)
        gx, gy = int(np.clip(troc_u_gt, 0, 1) * img_w + du), int(np.clip(troc_v_gt, 0, 1) * img_h + dv)
        px, py = int(np.clip(troc_u_pr, 0, 1) * img_w + du), int(np.clip(troc_v_pr, 0, 1) * img_h + dv)

        # tip (red) — crosshair for precise center
        draw.line([tx-8, ty, tx+8, ty], fill="red", width=1)
        draw.line([tx, ty-8, tx, ty+8], fill="red", width=1)
        # GT troc (green) — crosshair
        draw.line([gx-10, gy, gx+10, gy], fill="lime", width=1)
        draw.line([gx, gy-10, gx, gy+10], fill="lime", width=1)
        # pred troc (cyan) — crosshair (slightly smaller, offset diagonal)
        draw.line([px-7, py-7, px+7, py+7], fill="cyan", width=1)
        draw.line([px-7, py+7, px+7, py-7], fill="cyan", width=1)
        # error vector (magenta)
        draw.line([gx, gy, px, py], fill="magenta", width=1)

        err_px = float(np.sqrt((troc_u_pr - troc_u_gt)**2 + (troc_v_pr - troc_v_gt)**2) * img_w)
        px_errs.append(err_px)

        col = i % cols
        row = i // cols
        grid.paste(pil, (col * cell_w, row * (cell_h + label_h)))
        label = f"#{idx} lat={lateral:.1f}mm err={err_px:.1f}px vis={vis_p:.2f}"
        draw_grid.text((col * cell_w + 4, row * (cell_h + label_h) + cell_h + 4), label, fill="white")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    px_errs = np.array(px_errs)
    print(f"Saved -> {out_path}")
    print(f"This sample px_err: median={np.median(px_errs):.1f} mean={px_errs.mean():.1f} "
          f"p90={np.percentile(px_errs,90):.1f} max={px_errs.max():.1f}")


if __name__ == "__main__":
    main()
