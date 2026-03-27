#!/usr/bin/env python
"""
Convert DeepSpeed ZeRO-3 checkpoint (or zero_to_fp32 output) to a single .pt file
compatible with VLANeXt's pretrained_checkpoint loading.

Usage:
  # From ZeRO-3 sharded checkpoint directory
  python scripts/convert_zero3_to_pt.py \
      --input /path/to/checkpoints/VLANeXt_finetune_sim/first_zero3 \
      --output /path/to/checkpoints/VLANeXt_droid.pt \
      --tag final

  # From already-converted zero_to_fp32 output directory (pytorch_model*.bin)
  python scripts/convert_zero3_to_pt.py \
      --input /path/to/Saved_models/output_step1000.pt \
      --output /path/to/checkpoints/VLANeXt_droid.pt
"""

import argparse
import os
import sys
import json
import torch
from collections import OrderedDict


def load_from_bin_shards(input_dir):
    """Load from zero_to_fp32 output (pytorch_model-*.bin + index.json)."""
    index_path = os.path.join(input_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index file found at {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Collect unique shard files
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Loading from {len(shard_files)} shard(s)...")

    state_dict = OrderedDict()
    for shard_file in shard_files:
        shard_path = os.path.join(input_dir, shard_file)
        print(f"  Loading {shard_file}...")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        state_dict.update(shard)
        del shard

    return state_dict


def load_from_zero3_checkpoint(input_dir, tag):
    """Load from raw DeepSpeed ZeRO-3 checkpoint using zero_to_fp32.py."""
    zero_to_fp32_path = os.path.join(input_dir, "zero_to_fp32.py")
    if not os.path.exists(zero_to_fp32_path):
        raise FileNotFoundError(f"No zero_to_fp32.py found at {input_dir}")

    tag_dir = os.path.join(input_dir, tag)
    if not os.path.exists(tag_dir):
        raise FileNotFoundError(f"Tag directory not found: {tag_dir}")

    # Import zero_to_fp32 from the checkpoint directory
    sys.path.insert(0, input_dir)
    from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    sys.path.pop(0)

    print(f"Extracting FP32 state dict from ZeRO-3 checkpoint (tag={tag})...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(input_dir, tag=tag)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed ZeRO-3 checkpoint to single .pt file")
    parser.add_argument("--input", required=True, help="Path to ZeRO-3 checkpoint dir or zero_to_fp32 output dir")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--tag", default="final", help="Checkpoint tag for ZeRO-3 (default: final)")
    args = parser.parse_args()

    # Detect input type
    index_path = os.path.join(args.input, "pytorch_model.bin.index.json")
    zero_to_fp32_path = os.path.join(args.input, "zero_to_fp32.py")

    if os.path.exists(index_path):
        # Already converted by zero_to_fp32 → shard files
        print(f"Detected zero_to_fp32 output (bin shards): {args.input}")
        state_dict = load_from_bin_shards(args.input)
    elif os.path.exists(zero_to_fp32_path):
        # Raw ZeRO-3 checkpoint
        print(f"Detected raw ZeRO-3 checkpoint: {args.input}")
        state_dict = load_from_zero3_checkpoint(args.input, args.tag)
    else:
        raise FileNotFoundError(
            f"Cannot detect checkpoint type at {args.input}. "
            f"Expected pytorch_model.bin.index.json or zero_to_fp32.py"
        )

    # Strip 'module.' prefix if present (from DDP wrapping)
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[key] = v

    print(f"Total parameters: {len(cleaned)}")
    total_size = sum(v.numel() * v.element_size() for v in cleaned.values())
    print(f"Total size: {total_size / 1e9:.2f} GB")

    # Save in pretrained_checkpoint format
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Saving to {args.output}...")
    torch.save({"model_state_dict": cleaned}, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
