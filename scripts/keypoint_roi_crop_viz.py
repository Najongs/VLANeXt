"""ROI crop visualization: use current keypoint head as detector, crop around prediction.

Output grid (3 columns per sample):
  | full 256 with GT/pred | 128x128 crop centered on pred | 128x128 crop centered on GT |

Usage:
    python -m scripts.keypoint_roi_crop_viz \
        --ckpt checkpoints/keypoint_trocar/cotrain_v1/head_best.pt \
        --h5 dataset/keypoint/real_val_2k.h5 \
        --bucket far --n 12 --crop 128 \
        --out vqa_samples/roi_crop_far.png
"""
import argparse
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import SiglipVisionModel, SiglipImageProcessor

VISION_MODEL = "google/siglip2-so400m-patch16-512"


class Head(nn.Module):
    def __init__(self, h, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h,512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512,256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256,3),
        )
    def forward(self, x):
        f = x.mean(1); o = self.mlp(f)
        return torch.sigmoid(o[..., :2]), o[..., 2]


def crop_around(img, cu, cv, crop_size, img_size=256):
    """Crop crop_size square centered at (cu, cv) normalized coords, with pad if out-of-bounds."""
    half = crop_size // 2
    cx, cy = int(cu * img_size), int(cv * img_size)
    x0 = cx - half; y0 = cy - half
    # Pad image to handle out-of-bounds
    padded = np.zeros((img_size + crop_size, img_size + crop_size, 3), dtype=np.uint8)
    padded[half:half+img_size, half:half+img_size] = img
    # Adjust coords for padded image
    x0p = x0 + half; y0p = y0 + half
    crop = padded[y0p:y0p+crop_size, x0p:x0p+crop_size]
    return crop, (x0, y0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--bucket", choices=["all", "near", "mid", "far"], default="all")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    dev = "cuda"
    proc = SiglipImageProcessor.from_pretrained(VISION_MODEL)
    vm = SiglipVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16).to(dev).eval()
    head = Head(vm.config.hidden_size).to(dev).to(torch.bfloat16).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    head.load_state_dict(ck["head_state"])

    f = h5py.File(args.h5, "r")
    lat = f["lateral_mm"][:]
    if args.bucket == "near": pool = np.where(lat < 5)[0]
    elif args.bucket == "mid": pool = np.where((lat >= 5) & (lat < 15))[0]
    elif args.bucket == "far": pool = np.where(lat >= 15)[0]
    else: pool = np.arange(len(lat))

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(pool))
    idxs = sorted(rng.choice(pool, size=n, replace=False).tolist())

    # Predict
    images = np.stack([f["image"][i] for i in idxs])
    pils = [Image.fromarray(im) for im in images]
    proc_out = proc(images=pils, return_tensors="pt")
    pv = proc_out["pixel_values"].to(dev, dtype=torch.bfloat16)
    with torch.no_grad():
        feats = vm(pixel_values=pv).last_hidden_state
        uv_pred, _ = head(feats)
    uv_pred = uv_pred.float().cpu().numpy()

    # Build grid: 3 cols (full / pred-crop / GT-crop), n rows
    cell = 256
    cw = args.crop
    pad = 6
    label_h = 24
    cols_w = [cell, cw, cw]
    grid_w = sum(cols_w) + pad * 4
    row_h = cell + label_h + pad
    rows = n
    grid = Image.new("RGB", (grid_w, row_h * rows), (32, 32, 32))
    dg = ImageDraw.Draw(grid)

    for i, idx in enumerate(idxs):
        img = images[i]
        tip_uv = f["tip_uv"][idx]
        gt_uv = f["trocar_uv"][idx]
        pr_u, pr_v = uv_pred[i]
        latv = float(lat[idx])
        err_px = float(np.linalg.norm((uv_pred[i] - gt_uv) * 256))

        # Col 1: full image with overlays
        pil = Image.fromarray(img).convert("RGB")
        d = ImageDraw.Draw(pil)
        tx, ty = int(tip_uv[0]*256), int(tip_uv[1]*256)
        gx, gy = int(np.clip(gt_uv[0],0,1)*256), int(np.clip(gt_uv[1],0,1)*256)
        px, py = int(np.clip(pr_u,0,1)*256), int(np.clip(pr_v,0,1)*256)
        d.ellipse([tx-4,ty-4,tx+4,ty+4], outline="red", width=2)
        d.ellipse([gx-7,gy-7,gx+7,gy+7], outline="lime", width=2)
        d.ellipse([px-5,py-5,px+5,py+5], outline="cyan", width=2)
        # Pred-centered crop box
        half = args.crop // 2
        d.rectangle([px-half, py-half, px+half, py+half], outline="cyan", width=2)
        # GT-centered crop box
        d.rectangle([gx-half, gy-half, gx+half, gy+half], outline="lime", width=1)

        # Col 2: crop centered on prediction
        crop_pr, _ = crop_around(img, pr_u, pr_v, args.crop)
        pil_pr = Image.fromarray(crop_pr).convert("RGB")
        d2 = ImageDraw.Draw(pil_pr)
        # mark center
        d2.line([cw//2-4, cw//2, cw//2+4, cw//2], fill="cyan", width=1)
        d2.line([cw//2, cw//2-4, cw//2, cw//2+4], fill="cyan", width=1)
        # Mark GT relative position in this crop
        gt_rel_x = (gt_uv[0]*256 - (pr_u*256 - args.crop/2))
        gt_rel_y = (gt_uv[1]*256 - (pr_v*256 - args.crop/2))
        if 0 <= gt_rel_x < cw and 0 <= gt_rel_y < cw:
            d2.ellipse([gt_rel_x-5, gt_rel_y-5, gt_rel_x+5, gt_rel_y+5], outline="lime", width=2)

        # Col 3: crop centered on GT (oracle reference)
        crop_gt, _ = crop_around(img, gt_uv[0], gt_uv[1], args.crop)
        pil_gt = Image.fromarray(crop_gt).convert("RGB")
        d3 = ImageDraw.Draw(pil_gt)
        d3.ellipse([cw//2-5, cw//2-5, cw//2+5, cw//2+5], outline="lime", width=2)

        # Place into grid
        y = i * row_h
        grid.paste(pil, (pad, y))
        grid.paste(pil_pr, (pad*2 + cell, y + (cell - cw)//2))
        grid.paste(pil_gt, (pad*3 + cell + cw, y + (cell - cw)//2))
        dg.text((pad, y + cell + 2),
                f"#{idx} lat={latv:.1f}mm err={err_px:.1f}px | pred-crop center ↔ GT (lime in mid col)",
                fill="white")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
