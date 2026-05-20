"""Extract keypoint (image, trocar_uv) samples from real_align HDF5 episodes.

Real episodes already contain pre-computed `keypoints_wrist` = [tip_u, tip_v, trocar_u, trocar_v]
in [0,1] normalized image coords, plus `keypoints_visibility`. No re-rendering needed.

Subsample N frames per episode (skipping initial frames where action hasn't started
moving and avoiding consecutive-frame redundancy).

Usage:
    python -u -m scripts.keypoint_data_gen_real \\
        --data-roots dataset/real_align/collected_data_real_0430 \\
                     dataset/real_align/collected_data_real_0501 ... \\
        --frames-per-ep 5 --out dataset/keypoint/real_3k.h5
"""
import argparse
import glob
import io
import os
import time

import h5py
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-roots", nargs="+", required=True,
                   help="Directories containing episode_*.h5 files.")
    p.add_argument("--frames-per-ep", type=int, default=5)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip-first", type=int, default=10, help="Skip first N frames per ep (settling).")
    p.add_argument("--image-size", type=int, default=512,
                   help="Output image resolution (square). 2026-05-18 upgrade: 256→512.")
    args = p.parse_args()

    # Collect episode files.
    episodes = []
    for root in args.data_roots:
        # Accept both `episode_*.h5` (real) and `w*_episode_*.h5` (sim worker-prefixed)
        episodes.extend(sorted(glob.glob(os.path.join(root, "episode_*.h5")) +
                               glob.glob(os.path.join(root, "w*_episode_*.h5"))))
    print(f"Found {len(episodes)} episodes across {len(args.data_roots)} roots.")
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    H = W = int(args.image_size)
    f = h5py.File(args.out, "w")
    chunk_n = min(64, args.n_samples)
    img_ds = f.create_dataset("image", (args.n_samples, H, W, 3), dtype="u1", chunks=(chunk_n, H, W, 3))
    tip_ds = f.create_dataset("tip_uv", (args.n_samples, 2), dtype="f4")
    troc_ds = f.create_dataset("trocar_uv", (args.n_samples, 2), dtype="f4")
    vis_ds = f.create_dataset("trocar_visible", (args.n_samples,), dtype="f4")
    lat_ds = f.create_dataset("lateral_mm", (args.n_samples,), dtype="f4")  # proxy: tip-trocar distance

    saved = 0
    t0 = time.time()
    for ep_idx, ep_path in enumerate(episodes):
        if saved >= args.n_samples:
            break
        try:
            with h5py.File(ep_path, "r") as ef:
                imgs = ef["observations/images/tool_camera"][:]
                kw = ef["observations/keypoints_wrist"][:]  # (T, 4)
                kv = ef["observations/keypoints_visibility"][:]  # (T, 2)
                tip_pos = ef["observations/needle_tip_pos"][:]  # (T, 3) mm
                troc_pos = ef["observations/trocar_entry_pos"][:]  # (T, 3) mm
        except Exception as e:
            print(f"  skip {os.path.basename(ep_path)}: {e}")
            continue

        T = len(imgs)
        if T <= args.skip_first + args.frames_per_ep:
            continue

        # Weighted stratified sample across tip-trocar distance buckets.
        # Down-weight too-close (≤3mm: weak signal), emphasize mid-range, include some off-frame.
        dist_mm = np.linalg.norm(tip_pos - troc_pos, axis=1)  # (T,)
        valid = np.arange(args.skip_first, T)
        d_valid = dist_mm[valid]
        buckets_weights = [
            ((-0.1, 3.0), 0.10),
            ((3.0, 7.0), 0.30),
            ((7.0, 15.0), 0.30),
            ((15.0, 30.0), 0.25),
            ((30.0, 1e9), 0.05),
        ]
        sel = []
        for (lo, hi), w in buckets_weights:
            mask = (d_valid >= lo) & (d_valid < hi)
            candidates = valid[mask]
            if len(candidates) > 0:
                # weighted count, at least 1 if weight > 0
                take = max(1, int(round(args.frames_per_ep * w)))
                take = min(take, len(candidates))
                sel.extend(rng.choice(candidates, size=take, replace=False).tolist())
        # If under-filled, top up with random.
        remaining = args.frames_per_ep - len(sel)
        if remaining > 0 and len(valid) > 0:
            extras = rng.choice(valid, size=min(remaining, len(valid)), replace=False).tolist()
            sel.extend([x for x in extras if x not in sel])
        sel = sorted(set(sel))[: args.frames_per_ep]

        for idx in sel:
            if saved >= args.n_samples:
                break
            jpeg = bytes(imgs[idx])
            try:
                pil = Image.open(io.BytesIO(jpeg)).convert("RGB")
            except Exception:
                continue
            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.LANCZOS)
            arr = np.array(pil)

            tip_uv = kw[idx, :2]
            troc_uv = kw[idx, 2:4]
            troc_vis = float(kv[idx, 1])  # second slot = trocar visibility

            img_ds[saved] = arr
            tip_ds[saved] = tip_uv
            troc_ds[saved] = troc_uv
            vis_ds[saved] = troc_vis
            lat_ds[saved] = float(dist_mm[idx])
            saved += 1

        if (ep_idx + 1) % 50 == 0 or saved >= args.n_samples:
            elapsed = time.time() - t0
            rate = saved / max(elapsed, 1e-3)
            eta = (args.n_samples - saved) / max(rate, 1e-3)
            print(f"  saved {saved}/{args.n_samples} | {rate:.0f} samples/s | ETA {eta:.0f}s "
                  f"({ep_idx+1}/{len(episodes)} eps)")

    f.close()
    print(f"\nDone: {saved} samples in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
