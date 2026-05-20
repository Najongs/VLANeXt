"""Merge multiple keypoint H5 files into one (concatenate along sample axis)."""
import argparse
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    sizes = []
    for path in args.inputs:
        with h5py.File(path, "r") as f:
            sizes.append(f["image"].shape[0])
    total = sum(sizes)
    print(f"Merging {len(args.inputs)} files: {sizes} → total {total}")

    with h5py.File(args.inputs[0], "r") as f0:
        H, W, C = f0["image"].shape[1:]
        keys = list(f0.keys())

    f_out = h5py.File(args.out, "w")
    img_ds = f_out.create_dataset("image", (total, H, W, C), dtype="u1", chunks=(min(64, total), H, W, C))
    other_dsets = {}
    with h5py.File(args.inputs[0], "r") as f0:
        for k in keys:
            if k == "image":
                continue
            shape = (total,) + f0[k].shape[1:]
            other_dsets[k] = f_out.create_dataset(k, shape, dtype=f0[k].dtype)

    offset = 0
    for path, n in zip(args.inputs, sizes):
        with h5py.File(path, "r") as f:
            img_ds[offset:offset+n] = f["image"][:]
            for k in keys:
                if k == "image":
                    continue
                if k in f:
                    other_dsets[k][offset:offset+n] = f[k][:]
        print(f"  copied {n} from {path}")
        offset += n

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(total)
        print(f"Shuffling {total} samples...")
        img_data = img_ds[:][perm]
        img_ds[:] = img_data
        for k, ds in other_dsets.items():
            data = ds[:][perm]
            ds[:] = data

    f_out.close()
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
