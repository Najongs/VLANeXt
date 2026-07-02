#!/usr/bin/env python
"""Convert a DeepSpeed mp_rank model-state file into VLANeXt eval format."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="DeepSpeed mp_rank_00_model_states.pt")
    parser.add_argument("--output", required=True, help="Output checkpoint_*.pt")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if "module" not in checkpoint:
        raise KeyError(f"{src} does not contain a DeepSpeed 'module' state dict")

    out = {
        "step": checkpoint.get("global_steps", checkpoint.get("step", 0)),
        "config": checkpoint.get("config", {}),
        "model_state_dict": checkpoint["module"],
    }
    if not out["config"]:
        raise KeyError(f"{src} does not contain embedded training config")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"saved {dst}")
    print(f"  step={out['step']}, params={len(out['model_state_dict'])}")
    print(f"  first_key={next(iter(out['model_state_dict']))}")


if __name__ == "__main__":
    main()
