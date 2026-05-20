"""Train a frozen-SigLIP2 + MLP head to predict trocar_uv from tool_camera image.

Input  : RGB HxW (read from h5; legacy 256, current default 512)
Output : (trocar_u, trocar_v) in normalized [0,1] image coords + visibility logit

Loss   : MSE on (u,v) weighted by visibility mask
         + BCE on in-frame visibility flag (allows model to abstain when trocar is off-screen)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u -m scripts.train_keypoint \\
        --train-h5 dataset/keypoint/train_3k.h5 \\
        --val-h5   dataset/keypoint/val_500.h5 \\
        --out checkpoints/keypoint_trocar/run01 \\
        --steps 5000 --batch-size 64 --lr 1e-4
"""
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import SiglipVisionModel, SiglipImageProcessor


VISION_MODEL = "google/siglip2-so400m-patch16-512"  # match VLA backbone


class KeypointDataset(Dataset):
    def __init__(self, h5_path, image_processor, training: bool = True):
        self.h5_path = h5_path
        self.processor = image_processor
        self.training = training
        with h5py.File(h5_path, "r") as f:
            self.n = f["image"].shape[0]
            self.img_h = int(f["image"].shape[1])
            self.img_w = int(f["image"].shape[2])
        self._f = None  # lazy open per worker

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_open()
        img = self._f["image"][idx]  # (H,W,3) uint8
        uv = self._f["trocar_uv"][idx]  # (2,) float32 normalized
        lat = float(self._f["lateral_mm"][idx])

        # In-frame visibility flag: 1 if uv is within [0,1] with small margin.
        margin = 0.02
        visible = float(margin < uv[0] < 1 - margin and margin < uv[1] < 1 - margin)
        # Clamp uv to [0,1] for stable regression even when off-frame.
        uv_clamped = np.clip(uv, 0.0, 1.0)

        # Augmentation: small color jitter only (no spatial — would break uv labels).
        pil = Image.fromarray(img)
        if self.training:
            arr = np.array(pil, dtype=np.float32) / 255.0
            # Color jitter: per-channel scale 0.85-1.15, brightness shift -0.1..+0.1
            scale = np.random.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32)
            shift = np.random.uniform(-0.1, 0.1, size=(1, 1, 3)).astype(np.float32)
            arr = np.clip(arr * scale + shift, 0.0, 1.0)
            pil = Image.fromarray((arr * 255).astype(np.uint8))

        proc = self.processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "uv_target": torch.tensor(uv_clamped, dtype=torch.float32),
            "visible": torch.tensor(visible, dtype=torch.float32),
            "lateral_mm": torch.tensor(lat, dtype=torch.float32),
        }


class KeypointHead(nn.Module):
    """SigLIP2 frozen → mean-pool patch tokens → MLP → (u, v, visibility logit)."""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 3),  # u, v, vis_logit
        )

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, D)
        feat = patch_tokens.mean(dim=1)  # global avg pool
        out = self.mlp(feat)
        uv = torch.sigmoid(out[..., :2])
        vis_logit = out[..., 2]
        return uv, vis_logit


def evaluate(model_v, head, loader, device, px_scale=None):
    model_v.eval(); head.eval()
    n = 0
    sq_sum = 0.0
    px_errs = []
    vis_correct = 0
    vis_total = 0
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            uv_gt = batch["uv_target"].to(device)
            vis_gt = batch["visible"].to(device)
            out = model_v(pixel_values=pv)
            patches = out.last_hidden_state
            uv, vis_logit = head(patches)
            sq_sum += F.mse_loss(uv, uv_gt, reduction="sum").item()
            n += uv.shape[0]
            # Pixel error in source image-space (auto-detected from h5 image_size)
            scale = float(px_scale) if px_scale is not None else 256.0
            err = ((uv - uv_gt) * scale).pow(2).sum(dim=-1).sqrt()
            px_errs.extend(err.cpu().numpy().tolist())
            vis_pred = (torch.sigmoid(vis_logit) > 0.5).float()
            vis_correct += (vis_pred == vis_gt).sum().item()
            vis_total += vis_gt.shape[0]
    px = np.array(px_errs)
    return {
        "uv_mse": sq_sum / max(n, 1),
        "px_err_median": float(np.median(px)),
        "px_err_mean": float(px.mean()),
        "px_err_p90": float(np.percentile(px, 90)),
        "vis_acc": vis_correct / max(vis_total, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vision-model", default=VISION_MODEL)
    p.add_argument("--vis-loss-weight", type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    device = "cuda"
    log(f"Loading {args.vision_model} ...")
    image_processor = SiglipImageProcessor.from_pretrained(args.vision_model)
    vision = SiglipVisionModel.from_pretrained(args.vision_model, dtype=torch.bfloat16).to(device)
    vision.eval()
    for p_ in vision.parameters():
        p_.requires_grad_(False)
    hidden = vision.config.hidden_size
    log(f"Vision hidden size: {hidden}")

    head = KeypointHead(hidden).to(device).to(torch.bfloat16)
    log(f"Head params: {sum(p.numel() for p in head.parameters())}")

    train_ds = KeypointDataset(args.train_h5, image_processor, training=True)
    val_ds = KeypointDataset(args.val_h5, image_processor, training=False)
    px_scale = float(val_ds.img_w)  # report errors in source image-space
    log(f"train: {len(train_ds)}  val: {len(val_ds)}  img_size={val_ds.img_w}x{val_ds.img_h}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=False)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    train_iter = iter(train_loader)
    best_med = 1e9
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        uv_gt = batch["uv_target"].to(device)
        vis_gt = batch["visible"].to(device)

        with torch.no_grad():
            patches = vision(pixel_values=pv).last_hidden_state

        uv, vis_logit = head(patches)
        uv_f = uv.float(); vis_logit_f = vis_logit.float()
        # Only penalize uv where visible (1.0) — weight 1; otherwise 0.1.
        w = vis_gt + 0.1 * (1 - vis_gt)
        uv_err = ((uv_f - uv_gt) ** 2).sum(dim=-1)  # (B,)
        uv_loss = (uv_err * w).mean()
        vis_loss = F.binary_cross_entropy_with_logits(vis_logit_f, vis_gt)
        loss = uv_loss + args.vis_loss_weight * vis_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            log(f"[step {step:5d}] loss={loss.item():.5f} uv={uv_loss.item():.5f} "
                f"vis={vis_loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")

        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(vision, head, val_loader, device, px_scale=px_scale)
            log(f"  [val step {step}] uv_mse={metrics['uv_mse']:.5f} "
                f"px_err median/mean/p90={metrics['px_err_median']:.1f}/"
                f"{metrics['px_err_mean']:.1f}/{metrics['px_err_p90']:.1f}  "
                f"vis_acc={metrics['vis_acc']*100:.1f}%")
            ckpt = {
                "head_state": head.state_dict(),
                "step": step,
                "metrics": metrics,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"head_step{step}.pt")
            if metrics["px_err_median"] < best_med:
                best_med = metrics["px_err_median"]
                torch.save(ckpt, out_dir / "head_best.pt")
                log(f"  ★ new best median {best_med:.1f}px → saved head_best.pt")

    log(f"\nDone in {time.time()-t0:.0f}s. Best median px err = {best_med:.1f}")
    log_f.close()


if __name__ == "__main__":
    main()
