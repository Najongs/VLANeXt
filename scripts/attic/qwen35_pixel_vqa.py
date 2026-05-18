"""Variant VQA: ask Qwen3.5-VL to output PIXEL COORDINATES of needle tip and
trocar hole center directly. Avoids the 'centered' bias of the discrete-direction prompt.

The image is upscaled 2× (256→512) so pixel-level reasoning has more pixels to attend to.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen35_pixel_vqa \\
        --samples-dir vqa_samples/run02 \\
        --model /data/public/98_model/models--Qwen--Qwen3.5-9B/snapshots/<hash>
"""
import argparse, json, re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM_PROMPT = (
    "You analyze surgical alignment images. The image (512x512) shows:\n"
    "- A long thin gray rod (the needle) usually entering from the bottom of the image, pointing roughly upward.\n"
    "- A yellow/cream donut-shaped trocar with a dark circular hole in the center, sitting on a white phantom plate.\n\n"
    "Your job: report the pixel coordinates of TWO points.\n"
    "1) NEEDLE_TIP: the very top of the gray rod (its sharpest end).\n"
    "2) TROCAR_HOLE_CENTER: the center of the dark hole inside the yellow donut.\n\n"
    "Coordinate system: origin (0,0) at TOP-LEFT, x to the right, y downward. Range [0, 511].\n\n"
    "Respond with STRICT JSON only:\n"
    "{\n"
    '  "needle_tip": {"x": <int>, "y": <int>},\n'
    '  "trocar_hole_center": {"x": <int>, "y": <int>},\n'
    '  "trocar_visible": <true|false>,\n'
    '  "needle_visible": <true|false>\n'
    "}\n\n"
    "Be precise. The trocar hole center and the needle tip are usually 0-60 pixels apart. "
    "Look carefully — they are RARELY at exactly the same pixel."
)

USER_TEXT = "Report the pixel coordinates as JSON."

JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_resp(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--upscale", type=int, default=2, help="Upscale factor (512 final if input 256)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    samples_dir = Path(args.samples_dir)
    gt = json.load(open(samples_dir / "ground_truth.json"))
    recs = gt["samples"]
    if args.limit:
        recs = recs[: args.limit]

    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda"
    )
    model.eval()

    results = []
    for r in recs:
        img = Image.open(samples_dir / r["frame"]).convert("RGB")
        W, H = img.size
        if args.upscale > 1:
            img = img.resize((W * args.upscale, H * args.upscale), Image.LANCZOS)
        W2, H2 = img.size

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_TEXT},
            ]},
        ]
        chat_kwargs = dict(add_generation_prompt=True, tokenize=False)
        try:
            prompt = processor.apply_chat_template(messages, enable_thinking=False, **chat_kwargs)
        except TypeError:
            prompt = processor.apply_chat_template(messages, **chat_kwargs)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        gen = out[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        parsed = parse_resp(text)

        # Convert pred to 256-space for fair compare with GT.
        pred_tip = pred_troc = None
        if parsed:
            try:
                t = parsed["needle_tip"]; h = parsed["trocar_hole_center"]
                pred_tip = [float(t["x"]) / args.upscale, float(t["y"]) / args.upscale]
                pred_troc = [float(h["x"]) / args.upscale, float(h["y"]) / args.upscale]
            except Exception:
                pass

        # GT in 256 space (we built partial GT from log earlier — may lack uv coords)
        gt_tip = r.get("tip_uv"); gt_troc = r.get("trocar_uv")

        # Compute prediction quality: pixel error for tip and trocar; delta_uv direction match.
        err = {}
        if pred_tip and gt_tip:
            err["tip_err_px"] = float(((pred_tip[0]-gt_tip[0])**2 + (pred_tip[1]-gt_tip[1])**2) ** 0.5)
        if pred_troc and gt_troc:
            err["trocar_err_px"] = float(((pred_troc[0]-gt_troc[0])**2 + (pred_troc[1]-gt_troc[1])**2) ** 0.5)

        # Direction check: predicted delta vs GT direction
        dir_match = None
        if pred_tip and pred_troc:
            pdu = pred_troc[0] - pred_tip[0]
            pdv = pred_troc[1] - pred_tip[1]
            ang = (((180.0 / 3.14159) * (3.14159 / 2 - 0)) if False else 0)  # placeholder
            import math
            pa = (math.degrees(math.atan2(-pdv, pdu)) + 360.0) % 360.0
            sectors = [("right",0),("up-right",45),("up",90),("up-left",135),
                       ("left",180),("down-left",225),("down",270),("down-right",315)]
            pred_dir = min(sectors, key=lambda s: min(abs(pa-s[1]), 360-abs(pa-s[1])))[0]
            mag = (pdu**2 + pdv**2) ** 0.5
            if mag < 8:
                pred_dir = "centered"
            dir_match = pred_dir == r["gt_direction"]
            # within-1 sector
            order = [s[0] for s in sectors]
            within1 = False
            if pred_dir == r["gt_direction"]:
                within1 = True
            elif pred_dir != "centered" and r["gt_direction"] != "centered":
                i = order.index(pred_dir); j = order.index(r["gt_direction"])
                within1 = min(abs(i-j), 8-abs(i-j)) <= 1
        else:
            pred_dir = None
            within1 = False

        results.append({
            "ep": r["ep"], "frame": r["frame"],
            "gt_direction": r["gt_direction"], "gt_magnitude": r["gt_magnitude"],
            "lateral_mm": r["lateral_mm"], "angle_deg": r["angle_deg"],
            "pred": parsed, "pred_tip_256": pred_tip, "pred_troc_256": pred_troc,
            "pred_direction": pred_dir, "dir_match": dir_match,
            "within_1_sector": within1, "errors": err,
            "pred_raw": text,
        })
        print(f"[ep{r['ep']:03d}] GT={r['gt_direction']:<11s} lat={r['lateral_mm']:5.2f} "
              f"PRED tip={pred_tip} troc={pred_troc} -> {pred_dir}  "
              f"{'OK' if dir_match else ('~' if within1 else 'X')}")

    n = len(results)
    n_exact = sum(1 for x in results if x["dir_match"])
    n_w1 = sum(1 for x in results if x["within_1_sector"])
    print(f"\n=== Summary ===")
    print(f"N: {n}")
    print(f"Dir exact: {n_exact}/{n} ({100*n_exact/max(n,1):.0f}%)")
    print(f"Dir within 1: {n_w1}/{n} ({100*n_w1/max(n,1):.0f}%)")

    out = Path(args.out) if args.out else samples_dir / "predictions_pixel.json"
    with open(out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
