"""Convert state-dict-only ckpt to our format with vision_encoder.vision_model. prefix.

The new b100 baseline ckpt has keys like `vision_encoder.embeddings.X`,
but our model expects `vision_encoder.vision_model.embeddings.X`.
"""
import argparse
import torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input pytorch_model.bin path")
    p.add_argument("--output", required=True, help="Output .pt path")
    p.add_argument("--add-prefix", default="vision_model.",
                   help="Prefix to add inside vision_encoder.* keys")
    args = p.parse_args()

    state_dict = torch.load(args.input, map_location='cpu', weights_only=False)
    print(f"Loaded {len(state_dict)} keys")

    # Check if already has vision_model
    sample_keys = [k for k in state_dict.keys() if k.startswith("vision_encoder.")][:3]
    print(f"Sample vision_encoder keys before: {sample_keys}")
    needs_rename = sample_keys and not sample_keys[0].startswith("vision_encoder.vision_model.")

    if needs_rename:
        renamed = {}
        n_renamed = 0
        for k, v in state_dict.items():
            if k.startswith("vision_encoder.") and not k.startswith("vision_encoder.vision_model."):
                new_k = "vision_encoder." + args.add_prefix + k[len("vision_encoder."):]
                renamed[new_k] = v
                n_renamed += 1
            else:
                renamed[k] = v
        print(f"Renamed {n_renamed} vision_encoder.* keys")
        state_dict = renamed

    # Wrap in dict (eval script expects single tensor file OR wrapped dict)
    out = {
        'step': 50000,
        'model_state_dict': state_dict,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved: {args.output}")
    print(f"  step={out['step']}, {len(state_dict)} params")
    print(f"  Sample keys after: {list(state_dict.keys())[:3]}")


if __name__ == "__main__":
    main()
