"""
probe_keypoint.py

Question: does the frozen SigLIP2-SO400m vision encoder actually localize
the needle tip and trocar entry in the wrist-mounted camera image?

Approach:
  frozen SigLIP2 → mean-pool patch tokens → 2-layer MLP → regress one of:
    - kp2_pixel       : 2-d trocar pixel coord in [0,1] (kp index 1)
    - trocar_world    : 3-d trocar entry world position (mm)
    - tip_to_trocar   : 3-d offset (trocar - needle_tip) in world (mm)

If R² is high / pixel error is small, the encoder IS spatially aware →
the bottleneck is the head/policy, not the encoder. If R² is low, the
encoder doesn't see it → input resolution / encoder swap / 3D info needed.

Run:
    python -m scripts.probe_keypoint --target kp2_pixel  --steps 4000
    python -m scripts.probe_keypoint --target trocar_world --steps 4000
    python -m scripts.probe_keypoint --target tip_to_trocar --steps 4000
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import argparse
import glob
import io
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipVisionModel, SiglipImageProcessor


# ─── data ─────────────────────────────────────────────────────────────────
class ProbeDataset(Dataset):
    def __init__(self, files, target, processor, frame_stride=2):
        self.target = target
        self.processor = processor
        # build (file, frame_idx) index
        self.index = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                n = f["action"].shape[0]
            for i in range(0, n, frame_stride):
                self.index.append((fp, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fp, i = self.index[idx]
        with h5py.File(fp, "r") as f:
            jpeg = bytes(f["observations/images/tool_camera"][i])
            kp_w = f["observations/keypoints_wrist"][i]            # (4,) [u1,v1,u2,v2]
            vis = f["observations/keypoints_visibility"][i]        # (2,)
            tip = f["observations/needle_tip_pos"][i]              # (3,) mm
            trc = f["observations/trocar_entry_pos"][i]            # (3,) mm

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        # SigLIP2 processor handles resize+normalize. forces 512x512.
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]

        if self.target == "kp2_pixel":
            target = torch.tensor(kp_w[2:4], dtype=torch.float32)
            mask = torch.tensor(float(vis[1]), dtype=torch.float32)
        elif self.target == "trocar_world":
            target = torch.tensor(trc, dtype=torch.float32)
            mask = torch.tensor(1.0)
        elif self.target == "tip_to_trocar":
            target = torch.tensor(trc - tip, dtype=torch.float32)
            mask = torch.tensor(1.0)
        else:
            raise ValueError(self.target)
        return pixel, target, mask


# ─── model ────────────────────────────────────────────────────────────────
class ProbeHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True,
                   choices=["kp2_pixel", "trocar_world", "tip_to_trocar"])
    p.add_argument("--encoder", default="google/siglip2-so400m-patch16-512")
    p.add_argument("--data_root",
                   default="/data/public/NAS/VLANeXt/dataset/approach/approach_00/collected_data_merged")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--frame_stride", type=int, default=4)
    p.add_argument("--n_train_ep", type=int, default=300)   # episodes for train
    p.add_argument("--n_val_ep", type=int, default=50)      # held-out episodes
    p.add_argument("--pool", default="mean", choices=["mean", "cls"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out_dir", default="/data/public/NAS/VLANeXt/logs/probe")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"probe_{args.target}_{int(time.time())}"
    log_path = os.path.join(args.out_dir, f"{tag}.log")
    print(f"writing log → {log_path}")
    logf = open(log_path, "w")
    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    log(f"args: {vars(args)}")

    # files
    all_files = sorted(glob.glob(os.path.join(args.data_root, "*.h5")))
    random.Random(args.seed).shuffle(all_files)
    train_files = all_files[:args.n_train_ep]
    val_files = all_files[args.n_train_ep:args.n_train_ep + args.n_val_ep]
    log(f"train ep={len(train_files)}  val ep={len(val_files)}")

    # encoder
    processor = SiglipImageProcessor.from_pretrained(args.encoder)
    encoder = SiglipVisionModel.from_pretrained(args.encoder, dtype=torch.bfloat16).to(args.device)
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    in_dim = encoder.config.hidden_size
    log(f"encoder hidden_size={in_dim}")

    # data
    tr_ds = ProbeDataset(train_files, args.target, processor, frame_stride=args.frame_stride)
    va_ds = ProbeDataset(val_files, args.target, processor, frame_stride=args.frame_stride)
    log(f"train frames={len(tr_ds)}  val frames={len(va_ds)}")
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    # compute target mean/std on train (for standardization)
    Ys = []
    for fp in train_files[:50]:
        with h5py.File(fp, "r") as f:
            if args.target == "kp2_pixel":
                Ys.append(f["observations/keypoints_wrist"][:, 2:4])
            elif args.target == "trocar_world":
                Ys.append(f["observations/trocar_entry_pos"][:])
            elif args.target == "tip_to_trocar":
                Ys.append(f["observations/trocar_entry_pos"][:] - f["observations/needle_tip_pos"][:])
    Y = np.concatenate(Ys, 0).astype(np.float32)
    Y_mean = torch.tensor(Y.mean(0), device=args.device)
    Y_std  = torch.tensor(Y.std(0) + 1e-6, device=args.device)
    out_dim = int(Y_mean.numel())
    log(f"target mean={Y_mean.tolist()}  std={Y_std.tolist()}")

    head = ProbeHead(in_dim, out_dim).to(args.device).to(torch.bfloat16)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    # ─── train loop ─────────────────────────────────────────────────────
    step = 0
    t0 = time.time()
    tr_iter = iter(tr_loader)
    while step < args.steps:
        try:
            pixel, target, mask = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_loader)
            pixel, target, mask = next(tr_iter)
        pixel = pixel.to(args.device, dtype=torch.bfloat16, non_blocking=True)
        target = target.to(args.device, dtype=torch.float32, non_blocking=True)
        mask = mask.to(args.device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            feat = encoder(pixel).last_hidden_state  # (B, T, D)
            if args.pool == "mean":
                feat = feat.mean(dim=1)
            else:  # cls
                feat = feat[:, 0]

        pred_norm = head(feat).to(torch.float32)
        target_norm = (target - Y_mean) / Y_std
        # masked MSE (per-sample)
        err = ((pred_norm - target_norm) ** 2).mean(dim=-1)
        loss = (err * mask).sum() / (mask.sum() + 1e-6)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 100 == 0 or step == 1:
            log(f"step {step:5d} | loss_norm {loss.item():.4f} | {step/(time.time()-t0):.2f} it/s")

        if step % 500 == 0 or step == args.steps:
            head.eval()
            errs = []
            with torch.no_grad():
                for pixel, target, mask in va_loader:
                    pixel = pixel.to(args.device, dtype=torch.bfloat16)
                    target = target.to(args.device, dtype=torch.float32)
                    mask = mask.to(args.device, dtype=torch.float32)
                    feat = encoder(pixel).last_hidden_state
                    feat = feat.mean(dim=1) if args.pool == "mean" else feat[:, 0]
                    pred_norm = head(feat).to(torch.float32)
                    pred = pred_norm * Y_std + Y_mean
                    e = (pred - target).cpu().numpy()
                    m = mask.cpu().numpy()
                    errs.append((e, m))
            E = np.concatenate([e for e, _ in errs], 0)  # (N, D)
            M = np.concatenate([m for _, m in errs], 0)  # (N,)
            sel = M > 0.5
            E = E[sel]
            l2 = np.linalg.norm(E, axis=-1)
            mae_per = np.abs(E).mean(0)
            log(f"  VAL step {step}: n={len(E)} | L2 mean {l2.mean():.4f} median {np.median(l2):.4f}"
                f" | per-dim MAE {mae_per.tolist()}")
            head.train()

    logf.close()


if __name__ == "__main__":
    main()
