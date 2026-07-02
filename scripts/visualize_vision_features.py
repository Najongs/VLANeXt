"""
Vision encoder visualization (backbone-agnostic).

학습된 VLANeXt SigLIP2 / pretrained DINOv3 / 다른 ViT을 같은 인터페이스로 비교.
출력: 프레임별 [원본 | PCA-RGB | attention-rollout] 가로 concat PNG.

Examples:
    # 학습된 ckpt 사용 (SigLIP2 backbone 추출)
    python -m scripts.visualize_vision_features \
        --backbone google/siglip2-so400m-patch16-512 \
        --ckpt /home/najo/NAS/VLANeXt/checkpoints/.../step_5000.pt \
        --episode /home/najo/NAS/VLANeXt/dataset/approach/approach_00/collected_data_merged/w0_episode_20260507_174548.h5 \
        --frames 0 25 50 75 100 \
        --out viz/siglip2_step5000

    # Pretrained 비교 (학습 X)
    python -m scripts.visualize_vision_features --backbone facebook/dinov2-large \
        --episode <h5> --frames 0 25 50 --out viz/dinov2_pretrained

    python -m scripts.visualize_vision_features --backbone google/siglip2-so400m-patch16-512 \
        --episode <h5> --frames 0 25 50 --out viz/siglip2_pretrained
"""
import argparse
import io
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoImageProcessor


def load_backbone(backbone_path: str, device: str, dtype=torch.float32):
    """Backbone-agnostic loader. trust_remote_code on for DINOv3 류.
    attn_implementation='eager' 강제 — sdpa/flash는 output_attentions=True를 지원 안 함."""
    try:
        model = AutoModel.from_pretrained(
            backbone_path, trust_remote_code=True, attn_implementation="eager"
        ).to(device, dtype=dtype).eval()
    except (TypeError, ValueError):
        # 구버전 transformers나 attn_implementation 인자 미지원 모델 fallback
        model = AutoModel.from_pretrained(backbone_path, trust_remote_code=True).to(device, dtype=dtype).eval()
    # SigLIP/SigLIP2의 경우 image_text 모델이라 vision_model attribute 사용.
    if hasattr(model, "vision_model"):
        vision = model.vision_model
    else:
        vision = model
    proc = AutoImageProcessor.from_pretrained(backbone_path)
    return vision, proc


def load_finetuned_vision_weights(vision_module, ckpt_path: str):
    """train.py가 저장한 state_dict에서 vision_encoder.* 만 골라 로드.
    train ckpt 형태: dict with 'model_state_dict' (또는 'model').
    키 prefix: 'vision_encoder.vision_model.*' (SiglipVisionModel wrapper 때문).
    vision_module이 SiglipVisionTransformer면 'vision_model.' 까지 한 번 더 strip 필요."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model_state_dict", "model", "state_dict"):
            if k in sd:
                sd = sd[k]
                break

    prefix = "vision_encoder."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        print(f"[warn] no vision_encoder.* keys found in {ckpt_path} — using pretrained weights")
        return

    # vision_module이 SiglipVisionTransformer(혹은 inner)면 추가 'vision_model.' 제거 필요.
    # module의 첫 키 prefix와 sub의 첫 키 prefix를 비교해서 어느 형태인지 결정.
    module_keys = list(vision_module.state_dict().keys())
    module_starts_with_vision_model = any(k.startswith("vision_model.") for k in module_keys)
    sub_starts_with_vision_model = all(k.startswith("vision_model.") for k in sub)
    if sub_starts_with_vision_model and not module_starts_with_vision_model:
        sub = {k[len("vision_model."):]: v for k, v in sub.items()}
    missing, unexpected = vision_module.load_state_dict(sub, strict=False)
    loaded = len(sub) - len(unexpected)
    print(f"loaded {loaded}/{len(sub)} keys, missing={len(missing)}, unexpected={len(unexpected)}")


def decode_h5_frame(h5_path: str, cam_key: str, frame_idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        jpeg_bytes = f["observations"]["images"][cam_key][frame_idx]
    img = Image.open(io.BytesIO(jpeg_bytes.tobytes() if hasattr(jpeg_bytes, "tobytes") else bytes(jpeg_bytes)))
    return np.array(img.convert("RGB"))


@torch.no_grad()
def extract_patch_tokens_and_attn(vision, proc, image_np, device, dtype, input_size=None):
    """returns: patch_tokens (1, N, D), attentions list[(1, H, T, T)], grid_h, grid_w
    input_size: 지정 시 processor size를 override (DINOv3 등 dynamic res 지원 모델용)."""
    pil = Image.fromarray(image_np)
    proc_kwargs = {"images": pil, "return_tensors": "pt"}
    if input_size is not None:
        # 일반적인 HF processor는 size={"height", "width"} 또는 size=int
        proc_kwargs["size"] = {"height": input_size, "width": input_size}
    pixel = proc(**proc_kwargs)["pixel_values"].to(device, dtype=dtype)

    # interpolate_pos_encoding=True: native보다 큰 해상도일 때 pos embed 보간 (DINOv3/v2)
    forward_kwargs = dict(output_attentions=True, output_hidden_states=True)
    if input_size is not None:
        forward_kwargs["interpolate_pos_encoding"] = True
    try:
        out = vision(pixel_values=pixel, **forward_kwargs)
    except TypeError:
        forward_kwargs.pop("interpolate_pos_encoding", None)
        out = vision(pixel_values=pixel, **forward_kwargs)
    last = out.last_hidden_state  # (1, T, D) — T = (CLS?) + patches
    assert last is not None, "vision encoder didn't return last_hidden_state"
    attns = list(out.attentions) if out.attentions is not None else []

    # patch grid 추정: image_size / patch_size
    img_size = pixel.shape[-1]  # square
    # SigLIP/SigLIP2/DINO: model.config or embeddings.patch_size
    patch_size = getattr(vision.config, "patch_size", None) or getattr(getattr(vision, "embeddings", None), "patch_size", 16)
    grid = img_size // patch_size
    expected_patches = grid * grid

    if last.shape[1] == expected_patches:
        patch_tokens = last  # SigLIP은 CLS 없음
        cls_offset = 0
    elif last.shape[1] == expected_patches + 1:
        patch_tokens = last[:, 1:, :]  # DINO/CLIP은 CLS 있음
        cls_offset = 1
    else:
        # register token이 더 붙은 경우 (DINOv2-with-registers)
        cls_offset = last.shape[1] - expected_patches
        patch_tokens = last[:, cls_offset:, :]

    return patch_tokens, attns, grid, grid, cls_offset


def pca_rgb(patch_tokens_stack: torch.Tensor, grid_per_frame):
    """patch_tokens_stack: (F*N, D). PCA(3) → 프레임별 grid RGB.
    Returns: (uint8 grids (F, g, g, 3), explained_variance_ratio (3,))."""
    X = patch_tokens_stack.float().cpu().numpy()
    pca = PCA(n_components=3)
    proj = pca.fit_transform(X)  # (F*N, 3)
    # robust normalize per channel (2nd–98th percentile)
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1)

    g = grid_per_frame
    rgbs = proj.reshape(-1, g, g, 3)
    return (rgbs * 255).astype(np.uint8), pca.explained_variance_ratio_


def attention_rollout(attns, cls_offset, grid_h, grid_w, target_size, mode="last"):
    """attns: list of (1, H, T, T). Returns heatmap (target_size, target_size) in [0,1].
    mode='rollout': 전 layer 곱 (Abnar & Zuidema 2020) — 깊은 모델일수록 smearing
    mode='last':   마지막 layer attention만 — depth-fair, 가장 task-relevant attention
    CNN 백본(ConvNeXt 류)처럼 attention이 없으면 균일 회색 반환."""
    if not attns:
        return np.full((target_size, target_size), 0.5, dtype=np.float32)

    if mode == "last":
        A = attns[-1].mean(dim=1).float()  # (1, T, T)
        if cls_offset >= 1:
            attn_to_patches = A[0, 0, cls_offset:]
        else:
            attn_to_patches = A[0, :, :].mean(dim=0)
    else:
        rollout = None
        for A in attns:
            A = A.mean(dim=1).float()
            I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
            A = 0.5 * A + 0.5 * I
            A = A / A.sum(dim=-1, keepdim=True)
            rollout = A if rollout is None else torch.bmm(rollout, A)
        if cls_offset >= 1:
            attn_to_patches = rollout[0, 0, cls_offset:]
        else:
            attn_to_patches = rollout[0, :, :].mean(dim=0)
    heat = attn_to_patches.reshape(1, 1, grid_h, grid_w)
    heat = F.interpolate(heat, size=(target_size, target_size), mode="bilinear", align_corners=False)
    heat = heat.squeeze().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return heat


def overlay_heatmap(image_np, heat, alpha=0.45):
    """jet colormap overlay."""
    import matplotlib.cm as cm  # type: ignore[import-not-found]
    cmap = cm.get_cmap("jet")
    colored_rgba = np.asarray(cmap(heat))      # (H, W, 4) float
    colored = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    if colored.shape[:2] != image_np.shape[:2]:
        colored = np.array(Image.fromarray(colored).resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR))
    return ((1 - alpha) * image_np + alpha * colored).astype(np.uint8)


def upsample_grid(grid_rgb, target_size):
    return np.array(Image.fromarray(grid_rgb).resize((target_size, target_size), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="HF id or path (e.g., google/siglip2-so400m-patch16-512)")
    ap.add_argument("--ckpt", default=None, help="VLANeXt checkpoint .pt to load vision_encoder.* weights from")
    ap.add_argument("--episode", required=True, help="HDF5 episode path")
    ap.add_argument("--cam", default="tool_camera")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--attn-mode", default="last", choices=["last", "rollout"],
                    help="last: 마지막 layer (depth-fair). rollout: 전 layer 곱 (smearing 위험).")
    ap.add_argument("--input-size", type=int, default=None,
                    help="강제 input res (예: 512). 미지정시 processor default. DINOv3/v2는 보간 지원.")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    os.makedirs(args.out, exist_ok=True)

    print(f"loading backbone: {args.backbone}")
    vision, proc = load_backbone(args.backbone, args.device, dtype)
    if args.ckpt:
        print(f"loading fine-tuned vision weights from: {args.ckpt}")
        load_finetuned_vision_weights(vision, args.ckpt)
        vision = vision.to(args.device, dtype=dtype).eval()

    # 프레임별로 patch token + attention 수집
    images = [decode_h5_frame(args.episode, args.cam, i) for i in args.frames]
    tokens_per_frame, attns_per_frame, grids = [], [], []
    for img in images:
        pt, attns, gh, gw, cls = extract_patch_tokens_and_attn(vision, proc, img, args.device, dtype, input_size=args.input_size)
        tokens_per_frame.append(pt[0])  # (N, D)
        attns_per_frame.append((attns, cls, gh, gw))
        grids.append((gh, gw))
    print(f"grid: {grids[0]}, patches/frame: {tokens_per_frame[0].shape[0]}, dim: {tokens_per_frame[0].shape[1]}")

    # PCA: 모든 프레임을 한 번에 fit → 프레임 간 색이 의미 일관
    g = grids[0][0]
    stacked = torch.cat(tokens_per_frame, dim=0)  # (F*N, D)
    pca_rgbs, evr = pca_rgb(stacked, g)  # (F, g, g, 3), (3,)
    print(f"PCA explained variance ratio (top-3): {evr.tolist()} (sum={evr.sum():.3f})")

    # 출력
    H_target = images[0].shape[0]
    for i, (img, (attns, cls, gh, gw)) in enumerate(zip(images, attns_per_frame)):
        # PCA 패널: nearest upsample (patch 경계 보존)
        pca_img = upsample_grid(pca_rgbs[i], H_target)
        # Attention rollout overlay
        heat = attention_rollout(attns, cls, gh, gw, H_target, mode=args.attn_mode)
        attn_img = overlay_heatmap(img, heat)
        # concat
        h = img.shape[0]
        panels = [img, pca_img, attn_img]
        # 가로 정렬 위해 width 통일
        panels = [np.array(Image.fromarray(p).resize((h, h), Image.BILINEAR)) for p in panels]
        canvas = np.concatenate(panels, axis=1)
        out_path = os.path.join(args.out, f"frame_{args.frames[i]:04d}.png")
        Image.fromarray(canvas).save(out_path)
        print(f"saved {out_path}")

    # PCA basis 정보 저장 (다른 backbone 결과와 변동성 비교용)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"backbone={args.backbone}\nckpt={args.ckpt}\nepisode={args.episode}\n")
        f.write(f"frames={args.frames}\ngrid={g}x{g}\n")
        f.write(f"pca_explained_variance_ratio_top3={evr.tolist()}\n")
        f.write(f"pca_explained_variance_ratio_sum={float(evr.sum()):.4f}\n")


if __name__ == "__main__":
    main()
