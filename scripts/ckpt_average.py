"""Simple model soup: average model_state_dict of two checkpoints with same architecture.

Usage:
    python scripts/ckpt_average.py --ckpts A.pt B.pt --output averaged.pt [--weights 0.5 0.5]
"""
import argparse
import torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="2+ checkpoint paths to average")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-ckpt weight (must sum to 1.0). Default: equal weights.")
    p.add_argument("--output", required=True, help="Output ckpt path")
    args = p.parse_args()

    n = len(args.ckpts)
    weights = args.weights if args.weights else [1.0 / n] * n
    assert len(weights) == n, f"weights ({len(weights)}) != ckpts ({n})"
    assert abs(sum(weights) - 1.0) < 1e-5, f"weights sum {sum(weights)} != 1.0"
    print(f"Averaging {n} checkpoints with weights {weights}")

    # Load first
    base = torch.load(args.ckpts[0], map_location='cpu', weights_only=False)
    avg_sd = {k: v.float() * weights[0] for k, v in base['model_state_dict'].items()
              if torch.is_tensor(v)}
    # Add others
    for i, ck_path in enumerate(args.ckpts[1:], start=1):
        print(f"  Adding {Path(ck_path).name} (weight {weights[i]})")
        other = torch.load(ck_path, map_location='cpu', weights_only=False)
        other_sd = other['model_state_dict']
        for k in avg_sd.keys():
            if k not in other_sd:
                raise KeyError(f"Key {k} missing in {ck_path}")
            avg_sd[k] = avg_sd[k] + other_sd[k].float() * weights[i]

    # Cast back to original dtypes
    for k in avg_sd.keys():
        orig_dtype = base['model_state_dict'][k].dtype
        avg_sd[k] = avg_sd[k].to(orig_dtype)

    # Save as new ckpt (keep config from base for eval compatibility)
    out = {
        'step': base['step'],
        'model_state_dict': avg_sd,
        'config': base['config'],
        # Skip optimizer/scheduler — not needed for eval
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved averaged ckpt: {args.output}")
    print(f"  step={out['step']}, {len(avg_sd)} params")


if __name__ == "__main__":
    main()
