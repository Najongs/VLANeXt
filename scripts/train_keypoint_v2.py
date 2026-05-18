"""Train multi-output head: trocar_uv + dist_mm (3D) + visibility.

Diagnostic version: measure which signals SigLIP2-frozen can predict accurately.

Outputs:
  - uv (2): trocar_u, trocar_v in [0,1]
  - dist_norm (1): tip-trocar 3D distance normalized by 50mm
  - vis_logit (1): trocar visibility

Loss:
  MSE(uv) × vis_weight + 0.5 × MSE(dist_norm) + 0.1 × BCE(vis)
"""
import argparse, os, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor


VISION_MODEL = "google/siglip2-so400m-patch16-512"
DIST_NORM = 50.0  # mm


class KPDataset(Dataset):
    def __init__(self, h5_path, processor, training=True):
        self.path = h5_path; self.proc = processor; self.training = training
        with h5py.File(h5_path, "r") as f:
            self.n = f["image"].shape[0]
        self._f = None

    def _open(self):
        if self._f is None: self._f = h5py.File(self.path, "r")

    def __len__(self): return self.n

    def __getitem__(self, idx):
        self._open()
        img = self._f["image"][idx]
        uv = self._f["trocar_uv"][idx]
        dist = float(self._f["lateral_mm"][idx])  # stored 3D dist
        margin = 0.02
        vis = float(margin < uv[0] < 1-margin and margin < uv[1] < 1-margin)
        uv_c = np.clip(uv, 0.0, 1.0)
        pil = Image.fromarray(img)
        if self.training:
            arr = np.array(pil, dtype=np.float32) / 255.0
            scale = np.random.uniform(0.85, 1.15, size=(1,1,3)).astype(np.float32)
            shift = np.random.uniform(-0.1, 0.1, size=(1,1,3)).astype(np.float32)
            arr = np.clip(arr*scale + shift, 0, 1)
            pil = Image.fromarray((arr*255).astype(np.uint8))
        proc = self.proc(images=pil, return_tensors="pt")
        return {
            "pixel_values": proc["pixel_values"].squeeze(0),
            "uv": torch.tensor(uv_c, dtype=torch.float32),
            "dist_norm": torch.tensor(min(dist / DIST_NORM, 2.0), dtype=torch.float32),
            "vis": torch.tensor(vis, dtype=torch.float32),
            "dist_mm": torch.tensor(dist, dtype=torch.float32),
        }


class Head(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 4),  # u, v, dist_norm, vis_logit
        )

    def forward(self, x):
        feat = x.mean(dim=1)
        out = self.mlp(feat)
        return torch.sigmoid(out[..., :2]), out[..., 2], out[..., 3]


def evaluate(vis_model, head, loader, device):
    vis_model.eval(); head.eval()
    px_errs, dist_errs, vis_correct, n = [], [], 0, 0
    with torch.no_grad():
        for b in loader:
            pv = b["pixel_values"].to(device)
            uv_gt = b["uv"].to(device)
            dist_norm_gt = b["dist_norm"].to(device)
            dist_mm = b["dist_mm"].to(device)
            vis_gt = b["vis"].to(device)
            patches = vis_model(pixel_values=pv).last_hidden_state
            uv, dist_pred, vis_lg = head(patches)
            err_px = ((uv - uv_gt) * 256).pow(2).sum(-1).sqrt()
            err_dist = (dist_pred * DIST_NORM - dist_mm).abs()
            px_errs.extend(err_px.cpu().numpy().tolist())
            dist_errs.extend(err_dist.cpu().numpy().tolist())
            vis_correct += ((torch.sigmoid(vis_lg) > 0.5).float() == vis_gt).sum().item()
            n += uv.shape[0]
    px = np.array(px_errs); de = np.array(dist_errs)
    return {
        "px_median": float(np.median(px)), "px_mean": float(px.mean()), "px_p90": float(np.percentile(px,90)),
        "dist_median_mm": float(np.median(de)), "dist_mean_mm": float(de.mean()), "dist_p90_mm": float(np.percentile(de,90)),
        "vis_acc": vis_correct / max(n,1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vis-weight", type=float, default=0.1)
    p.add_argument("--dist-weight", type=float, default=0.5)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "w")
    def log(m):
        print(m); log_f.write(m+"\n"); log_f.flush()

    device = "cuda"
    proc = SiglipImageProcessor.from_pretrained(VISION_MODEL)
    vis_model = SiglipVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16).to(device).eval()
    for p_ in vis_model.parameters(): p_.requires_grad_(False)
    hidden = vis_model.config.hidden_size
    head = Head(hidden).to(device).to(torch.bfloat16)
    log(f"hidden={hidden}, head params={sum(p.numel() for p in head.parameters())}")

    train_ds = KPDataset(args.train_h5, proc, training=True)
    val_ds = KPDataset(args.val_h5, proc, training=False)
    log(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    it = iter(train_loader)
    best_dist = 1e9
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try: b = next(it)
        except StopIteration:
            it = iter(train_loader); b = next(it)
        pv = b["pixel_values"].to(device, dtype=torch.bfloat16)
        uv_gt = b["uv"].to(device); dist_norm_gt = b["dist_norm"].to(device); vis_gt = b["vis"].to(device)
        with torch.no_grad():
            patches = vis_model(pixel_values=pv).last_hidden_state
        uv, dist_pred, vis_lg = head(patches)
        uv_f = uv.float(); dist_f = dist_pred.float(); vis_f = vis_lg.float()
        w = vis_gt + 0.1 * (1 - vis_gt)
        uv_loss = (((uv_f - uv_gt) ** 2).sum(-1) * w).mean()
        dist_loss = F.mse_loss(dist_f, dist_norm_gt)
        vis_loss = F.binary_cross_entropy_with_logits(vis_f, vis_gt)
        loss = uv_loss + args.dist_weight * dist_loss + args.vis_weight * vis_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            log(f"[{step:5d}] L={loss.item():.4f} uv={uv_loss.item():.4f} dist={dist_loss.item():.4f} vis={vis_loss.item():.3f} lr={sched.get_last_lr()[0]:.1e}")
        if step % args.eval_interval == 0 or step == args.steps:
            m = evaluate(vis_model, head, val_loader, device)
            log(f"  [val {step}] px med/mean/p90={m['px_median']:.1f}/{m['px_mean']:.1f}/{m['px_p90']:.1f}  "
                f"dist med/mean/p90={m['dist_median_mm']:.2f}/{m['dist_mean_mm']:.2f}/{m['dist_p90_mm']:.2f}mm  "
                f"vis_acc={m['vis_acc']*100:.1f}%")
            ck = {"head_state": head.state_dict(), "step": step, "metrics": m, "args": vars(args)}
            torch.save(ck, out_dir / f"head_step{step}.pt")
            if m["dist_median_mm"] < best_dist:
                best_dist = m["dist_median_mm"]
                torch.save(ck, out_dir / "head_best.pt")
                log(f"  ★ new best dist median {best_dist:.2f}mm")
    log(f"\nDone in {time.time()-t0:.0f}s. Best dist median = {best_dist:.2f}mm")
    log_f.close()


if __name__ == "__main__":
    main()
