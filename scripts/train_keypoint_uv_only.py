"""Single-task: predict trocar_uv ONLY. Isolated learnability test (mirror of dist_only)."""
import argparse, time
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor

VISION_MODEL = "google/siglip2-so400m-patch16-512"


class DS(Dataset):
    def __init__(self, h5, proc, training=True):
        self.path = h5; self.proc = proc; self.training = training
        with h5py.File(h5, "r") as f:
            self.n = f["image"].shape[0]
            self.img_w = int(f["image"].shape[2])
        self._f = None
    def _o(self):
        if self._f is None: self._f = h5py.File(self.path, "r")
    def __len__(self): return self.n
    def __getitem__(self, i):
        self._o()
        img = self._f["image"][i]
        uv = self._f["trocar_uv"][i]
        uv_c = np.clip(uv, 0.0, 1.0)
        pil = Image.fromarray(img)
        if self.training:
            a = np.array(pil, dtype=np.float32) / 255.0
            s = np.random.uniform(0.85, 1.15, size=(1,1,3)).astype(np.float32)
            sh = np.random.uniform(-0.1, 0.1, size=(1,1,3)).astype(np.float32)
            pil = Image.fromarray((np.clip(a*s+sh, 0, 1)*255).astype(np.uint8))
        return {
            "pv": self.proc(images=pil, return_tensors="pt")["pixel_values"].squeeze(0),
            "uv": torch.tensor(uv_c, dtype=torch.float32),
        }


class Head(nn.Module):
    def __init__(self, h, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        return torch.sigmoid(self.mlp(x.mean(dim=1)))


def evaluate(vm, head, loader, device, px_scale=256):
    vm.eval(); head.eval()
    errs = []
    with torch.no_grad():
        for b in loader:
            pv = b["pv"].to(device, dtype=torch.bfloat16)
            uv_gt = b["uv"].to(device)
            feats = vm(pixel_values=pv).last_hidden_state
            uv = head(feats).float()
            err_px = ((uv - uv_gt) * float(px_scale)).pow(2).sum(-1).sqrt()
            errs.extend(err_px.cpu().numpy().tolist())
    e = np.array(errs)
    return {"med": float(np.median(e)), "mean": float(e.mean()), "p90": float(np.percentile(e,90)), "p99": float(np.percentile(e,99))}


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
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out/"train.log", "w")
    def lg(m): print(m); log_f.write(m+"\n"); log_f.flush()

    dev = "cuda"
    proc = SiglipImageProcessor.from_pretrained(VISION_MODEL)
    vm = SiglipVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16).to(dev).eval()
    for x in vm.parameters(): x.requires_grad_(False)
    head = Head(vm.config.hidden_size).to(dev).to(torch.bfloat16)
    lg(f"head params={sum(p.numel() for p in head.parameters())}")

    tr = DS(args.train_h5, proc, True); vl = DS(args.val_h5, proc, False)
    px_scale = float(vl.img_w)
    lg(f"train={len(tr)} val={len(vl)} img_w={vl.img_w}")
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    vll = DataLoader(vl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    it = iter(trl); best = 1e9; t0 = time.time()
    for step in range(1, args.steps+1):
        try: b = next(it)
        except StopIteration: it = iter(trl); b = next(it)
        pv = b["pv"].to(dev, dtype=torch.bfloat16)
        uv_gt = b["uv"].to(dev)
        with torch.no_grad():
            feats = vm(pixel_values=pv).last_hidden_state
        uv = head(feats).float()
        loss = F.mse_loss(uv, uv_gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 50 == 0:
            lg(f"[{step:5d}] L={loss.item():.5f} lr={sch.get_last_lr()[0]:.1e}")
        if step % args.eval_interval == 0 or step == args.steps:
            m = evaluate(vm, head, vll, dev, px_scale=px_scale)
            lg(f"  [val {step}] px med/mean/p90/p99 = {m['med']:.1f}/{m['mean']:.1f}/{m['p90']:.1f}/{m['p99']:.1f}")
            ck = {"head_state": head.state_dict(), "step": step, "metrics": m}
            torch.save(ck, out/f"head_step{step}.pt")
            if m["med"] < best:
                best = m["med"]; torch.save(ck, out/"head_best.pt")
                lg(f"  ★ best {best:.1f}px")
    lg(f"Done {time.time()-t0:.0f}s, best med={best:.1f}px"); log_f.close()


if __name__ == "__main__":
    main()
