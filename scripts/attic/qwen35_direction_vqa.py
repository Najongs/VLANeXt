"""Run Qwen3.5-VL-9B as a direction oracle and grade against ground truth.

Loads Qwen3.5-9B via AutoModelForImageTextToText + AutoProcessor (bf16),
queries each sample frame with a strict JSON prompt, parses the answer, and
compares to ground-truth direction/magnitude labels.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen35_direction_vqa \\
        --samples-dir vqa_samples/run01 \\
        --model Qwen/Qwen3.5-9B \\
        --out vqa_samples/run01/predictions.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM_PROMPT = (
    "You are a precision needle-trocar alignment assistant. "
    "You see a single image from a tool-mounted camera looking at a long thin needle "
    "and a small trocar (a dark circular hole on a flat phantom surface).\n\n"
    "Your job: estimate the 2D offset from the needle tip to the trocar hole center "
    "in the image plane, and judge whether the needle is aimed into the hole.\n\n"
    "Respond with STRICT JSON only, no prose, matching this schema:\n"
    "{\n"
    '  "direction": "<one of: up, down, left, right, up-left, up-right, down-left, down-right, centered>",\n'
    '  "magnitude": "<one of: tiny, small, medium, large>",\n'
    '  "confidence": "<low | medium | high>",\n'
    '  "trocar_visible": <true | false>,\n'
    '  "needle_pointing_into_hole": <true | false>\n'
    "}\n\n"
    "Rules:\n"
    "- The image is 256x256. 'right' = needle should move toward image +x; 'down' = toward image +y.\n"
    "- 'centered' = the tip is within ~2mm (~8 pixels) of the hole center.\n"
    "- 'needle_pointing_into_hole' is true ONLY if the needle long-axis is aimed at the dark hole interior, "
    "not at the surrounding solid phantom.\n"
    "- If you see only the trocar's outer rim (not the dark inside), set needle_pointing_into_hole=false.\n"
    "- Be conservative. If unsure, pick lower confidence rather than guessing."
)

USER_TEXT = (
    "Estimate the offset from the needle tip to the trocar hole center. "
    "Output the JSON only."
)


JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.S)


def parse_response(text):
    text = text.strip()
    # Strip <think>...</think> blocks first (Qwen3 thinking mode).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Strip code fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try every JSON-looking block, prefer the LAST (final answer after reasoning).
    matches = JSON_BLOCK_RE.findall(text)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


def direction_distance(pred, gt):
    """Circular distance in 45-degree sectors. centered is its own bucket."""
    order = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    if pred == gt:
        return 0
    if pred == "centered" or gt == "centered":
        return 4  # treat as worst case unless exact match
    if pred not in order or gt not in order:
        return 4
    i, j = order.index(pred), order.index(gt)
    return min(abs(i - j), 8 - abs(i - j))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--out", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Let model do CoT before JSON (default off — direct JSON)")
    args = p.parse_args()

    samples_dir = Path(args.samples_dir)
    gt = json.load(open(samples_dir / "ground_truth.json"))
    records = gt["samples"]
    if args.limit:
        records = records[: args.limit]

    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda"
    )
    model.eval()

    results = []
    for rec in records:
        img_path = samples_dir / rec["frame"]
        image = Image.open(img_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_TEXT},
            ]},
        ]
        chat_kwargs = dict(add_generation_prompt=True, tokenize=False)
        try:
            prompt = processor.apply_chat_template(
                messages, enable_thinking=args.enable_thinking, **chat_kwargs
            )
        except TypeError:
            prompt = processor.apply_chat_template(messages, **chat_kwargs)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=0.95,
            )
        gen = out[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        parsed = parse_response(text)

        pred_dir = (parsed or {}).get("direction")
        pred_mag = (parsed or {}).get("magnitude")
        gt_dir = rec["gt_direction"]
        gt_mag = rec["gt_magnitude"]
        d_dist = direction_distance(pred_dir, gt_dir) if pred_dir else None

        results.append({
            "ep": rec["ep"],
            "lateral_mm": rec["lateral_mm"],
            "angle_deg": rec["angle_deg"],
            "gt_direction": gt_dir, "gt_magnitude": gt_mag,
            "pred_raw": text,
            "pred": parsed,
            "dir_match": pred_dir == gt_dir,
            "dir_within_1_sector": d_dist is not None and d_dist <= 1,
            "mag_match": pred_mag == gt_mag,
        })
        print(f"[ep{rec['ep']:03d}] lat={rec['lateral_mm']:5.2f}mm  "
              f"GT={gt_dir}/{gt_mag}  PRED={pred_dir}/{pred_mag}  "
              f"{'OK' if pred_dir == gt_dir else ('~' if d_dist is not None and d_dist <= 1 else 'X')}")

    n = len(results)
    n_dir = sum(r["dir_match"] for r in results)
    n_dir1 = sum(r["dir_within_1_sector"] for r in results)
    n_mag = sum(r["mag_match"] for r in results)
    n_parsed = sum(r["pred"] is not None for r in results)

    print("\n=== Summary ===")
    print(f"Samples:        {n}")
    print(f"JSON parsed:    {n_parsed}/{n} ({100*n_parsed/max(n,1):.1f}%)")
    print(f"Dir exact:      {n_dir}/{n} ({100*n_dir/max(n,1):.1f}%)")
    print(f"Dir within 1:   {n_dir1}/{n} ({100*n_dir1/max(n,1):.1f}%)")
    print(f"Mag exact:      {n_mag}/{n} ({100*n_mag/max(n,1):.1f}%)")
    print("\nConfusion (GT -> PRED counts):")
    confusion = Counter((r["gt_direction"], (r["pred"] or {}).get("direction")) for r in results)
    for (g, p), c in sorted(confusion.items()):
        print(f"  {g:>12s} -> {str(p):<12s}: {c}")

    out_path = Path(args.out) if args.out else samples_dir / "predictions.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "summary": {
            "n": n, "parsed": n_parsed, "dir_exact": n_dir,
            "dir_within_1": n_dir1, "mag_exact": n_mag,
        }}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
