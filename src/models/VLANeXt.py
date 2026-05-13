import sys
import os
from contextlib import nullcontext

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModel, AutoImageProcessor,
    SiglipVisionModel, SiglipImageProcessor,
    Siglip2VisionModel, Siglip2ImageProcessor,
    LlamaForCausalLM,
    PaliGemmaForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
from peft import LoraConfig, get_peft_model
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from .policies import (
    ActionDiffusionTransformerMetaquery, ActionDiffusionTransformerMoE,
    ActionRegressionTransformerMetaquery, ActionRegressionTransformerMoE,
    ActionClassificationTransformerMetaquery, ActionClassificationTransformerMoE, ActionVQVAE
)
from .generator import ImageGeneratorTransformer
from .encoder import ActionTransformerProjector
from .connector import ConnectorTransformer


def _build_2d_sincos_pos_embed(h, w, dim):
    """ACT/DETR-style 2D sinusoidal positional encoding. Returns (h*w, dim).
    dim는 4의 배수여야 함 (h,w 각각 sin/cos 절반씩)."""
    assert dim % 4 == 0, f"2D sincos pos embed needs dim%4==0, got {dim}"
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (h, w) each
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)
    half = dim // 4
    omega = torch.arange(half, dtype=torch.float32) / float(half)
    omega = 1.0 / (10000.0 ** omega)
    eh = gh[:, None] * omega[None, :]  # (h*w, half)
    ew = gw[:, None] * omega[None, :]
    return torch.cat([torch.sin(ew), torch.cos(ew), torch.sin(eh), torch.cos(eh)], dim=-1)  # (h*w, dim)

try:
    from .Emu3_5_VisionTokenizer.modeling_emu3p5visionvq import Emu3p5VisionVQModel
except ImportError:
    # Fallback for directory with dot in name (Emu3.5_VisionTokenizer) which is not a valid package name
    sys.path.append(os.path.join(os.path.dirname(__file__), "Emu3.5_VisionTokenizer"))
    from modeling_emu3p5visionvq import Emu3p5VisionVQModel



class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

class SpatialCrossAttentionHead(nn.Module):
    """Cross-attention based spatial head.

    Spatial queries attend to wrist camera tokens for visibility prediction.
    Global queries (dist/phase) attend to ALL image tokens across views.

    Output layout (4D):
        [0:2] visibility: tip_visible, trocar_visible (logits)
        [2]   dist: normalized 3D distance
        [3]   phase: align/insert (logit)
    """

    def __init__(self, hidden_size, num_layers_to_use=4, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers_to_use = num_layers_to_use
        self.num_heads = num_heads

        # Learnable spatial queries: tip + trocar (visibility, wrist-only)
        self.spatial_queries = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        # Learnable global queries: dist + phase (all views)
        self.global_queries = nn.Parameter(torch.randn(2, hidden_size) * 0.02)

        # Layer projection: fuse selected layers into one
        self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use) / num_layers_to_use)

        # Cross-attention (shared projections)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

        # Per-query output: visibility logit only (1 per query → 2 total)
        self.visibility_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),  # visibility_logit
        )

        # Global output from both queries: dist + phase = 2
        self.global_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # dist_norm, phase_logit
        )

    def _cross_attend(self, queries, kv, attn_mask, B):
        """Shared cross-attention logic."""
        num_q = queries.shape[1]
        head_dim = self.hidden_size // self.num_heads

        Q = self.q_proj(queries).view(B, num_q, self.num_heads, head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, -1, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, num_q, self.hidden_size)
        return self.o_proj(attn_out)

    def _extract_and_pad(self, fused, mask, B):
        """Extract masked tokens and pad to same length."""
        patch_list = []
        max_patches = 0
        for b in range(B):
            patches = fused[b][mask[b]]
            patch_list.append(patches)
            max_patches = max(max_patches, patches.shape[0])

        if max_patches == 0:
            return self.norm_kv(fused), None

        padded = torch.zeros(B, max_patches, self.hidden_size,
                             device=fused.device, dtype=fused.dtype)
        attn_mask = torch.zeros(B, max_patches, device=fused.device, dtype=torch.bool)
        for b, patches in enumerate(patch_list):
            n = patches.shape[0]
            padded[b, :n] = patches
            attn_mask[b, :n] = True
        return self.norm_kv(padded), attn_mask

    def forward(self, hidden_states, image_token_mask, image_grid_thw=None, wrist_image_index=1):
        """
        Args:
            hidden_states: tuple of (B, seq_len, H) from all VLM layers
            image_token_mask: (B, seq_len) bool, True for image patch tokens
            image_grid_thw: (num_images, 3) tensor — per-image (t, h, w) grid info.
                            Used to identify wrist camera token range.
            wrist_image_index: index of wrist camera in image sequence (default: 1,
                               i.e. [side=0, wrist=1, top=2])
        Returns:
            spatial_pred: (B, 8)
        """
        # Select last N layers and compute weighted sum
        num_layers = len(hidden_states)
        selected_indices = list(range(
            max(0, num_layers - self.num_layers_to_use), num_layers
        ))
        w = F.softmax(self.layer_weights[:len(selected_indices)], dim=0)
        fused = sum(
            w[i] * hidden_states[idx] for i, idx in enumerate(selected_indices)
        )  # (B, seq_len, H)

        B = fused.shape[0]

        # --- Build wrist-only mask from image_grid_thw ---
        wrist_mask = None
        if image_grid_thw is not None and image_grid_thw.shape[0] > wrist_image_index:
            # Compute token counts per image
            tokens_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
            # For batched processing, image_grid_thw is stacked: B * num_images_per_sample rows
            num_images_per_sample = len(tokens_per_image) // B if B > 0 else 0

            if num_images_per_sample > wrist_image_index:
                wrist_mask = torch.zeros_like(image_token_mask)  # (B, seq_len)
                for b in range(B):
                    # Get image token positions for this sample
                    img_positions = image_token_mask[b].nonzero(as_tuple=True)[0]
                    if img_positions.numel() == 0:
                        continue
                    # Compute wrist token range within image tokens
                    img_idx_base = b * num_images_per_sample
                    offset = sum(int(tokens_per_image[img_idx_base + i]) for i in range(wrist_image_index))
                    wrist_len = int(tokens_per_image[img_idx_base + wrist_image_index])
                    end = min(offset + wrist_len, img_positions.numel())
                    if offset < end:
                        wrist_positions = img_positions[offset:end]
                        wrist_mask[b, wrist_positions] = True

        # --- Visibility cross-attention: wrist-only tokens ---
        vis_token_mask = wrist_mask if wrist_mask is not None else image_token_mask
        vis_kv, vis_attn_mask = self._extract_and_pad(fused, vis_token_mask, B)

        vis_queries = self.spatial_queries.unsqueeze(0).expand(B, -1, -1)
        vis_queries = self.norm_q(vis_queries)
        vis_out = self._cross_attend(vis_queries, vis_kv, vis_attn_mask, B)  # (B, 2, H)

        tip_vis = self.visibility_head(vis_out[:, 0])      # (B, 1): vis_logit
        trocar_vis = self.visibility_head(vis_out[:, 1])    # (B, 1): vis_logit

        # --- Global cross-attention: all image tokens ---
        all_kv, all_attn_mask = self._extract_and_pad(fused, image_token_mask, B)

        global_q = self.global_queries.unsqueeze(0).expand(B, -1, -1)
        global_q = self.norm_q(global_q)
        global_out = self._cross_attend(global_q, all_kv, all_attn_mask, B)  # (B, 2, H)

        global_feat = torch.cat([global_out[:, 0], global_out[:, 1]], dim=-1)  # (B, 2H)
        global_pred = self.global_head(global_feat)  # (B, 2): dist, phase_logit

        # Assemble: [tip_vis, trocar_vis, dist, phase]
        spatial_pred = torch.cat([
            tip_vis,              # tip visibility logit
            trocar_vis,           # trocar visibility logit
            global_pred,          # dist, phase logit
        ], dim=-1)  # (B, 4)

        return spatial_pred


class VLANeXt(nn.Module):
    def __init__(
        self, 
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7,
        num_actions=1,
        num_queries=16,
        num_history=0,
        loss_type="diffusion", # Options: "diffusion", "regression", "classification"
        future_image_loss_weight=0.0,
        num_train_timesteps=1000,
        num_inference_timesteps=10,
        scheduler_type="ddim", # Options: "ddim", "flow_match"
        condition_type="loose", # Options: "loose", "tight", "soft"
        policy_hidden_size=1024,
        policy_depth=24,
        policy_num_heads=16,
        policy_mlp_ratio=4.0,
        use_proprio_input_vlm=True,
        use_action_input_policy=False,
        use_transformer_proprio_projector=True,
        projector_depth=2,
        projector_num_heads=4,
        use_transformer_connector=True,
        connector_depth=2,
        connector_num_heads=4,
        backbone_mode="finetune", # Options: "frozen", "finetune", "lora", "last_n_unfreeze"
        lora_config=None,
        n_unfreeze_layers=4,  # used when backbone_mode == "last_n_unfreeze"
        gradient_checkpointing=True,
        num_bins=256,
        action_vqvae=None,
        generator_hidden_size=768,
        generator_depth=12,
        generator_num_heads=12,
        generator_mlp_ratio=4.0,
        attn_implementation="flash_attention_2",
        dct_loss_weight=0.1,
        dct_low_freq_weight=1.0,
        dct_high_freq_weight=3.0,
        dct_freq_split=0.5,
        dct_similarity_type="mse",  # Options: "mse", "mae", "cosine"
        spatial_loss_weight=0.0,
        proprio_dim=None,
        aux_distance_loss=None,
        direction_decoupled_loss=None,
        input_image_size=None,   # vision_only 경로에서 image_processor size 강제 (예: DINOv3 512).
    ):
        super().__init__()
        # Aux distance loss config (sim-only). Disabled if config missing/false.
        adl = aux_distance_loss or {}
        self.aux_dist_enabled = bool(adl.get("enabled", False))
        self.aux_dist_weight = float(adl.get("weight", 0.0))
        self.aux_dist_margin_mm = float(adl.get("margin_mm", 0.1))
        pos_max = adl.get("pos_denorm_mm", [0.30, 0.30, 0.25])
        self.register_buffer(
            "aux_pos_denorm_mm",
            torch.tensor(pos_max, dtype=torch.float32),
            persistent=False,
        )
        # Near-goal weighting: boost loss for samples where needle tip is close to trocar.
        # weight_per_sample = 1 + clamp(near_goal_scale_mm / (cur_dist_mm + 1), max=near_goal_max_boost)
        # Disabled when near_goal_scale_mm <= 0 (default).
        self.aux_dist_near_goal_scale_mm = float(adl.get("near_goal_scale_mm", 0.0))
        self.aux_dist_near_goal_max_boost = float(adl.get("near_goal_max_boost", 4.0))
        self.aux_dist_apply_to_main = bool(adl.get("apply_to_main_loss", True))
        # Hold-aware: when cur_dist <= hold_threshold_mm, drop margin to 0
        # (= "no further progress required when already inside target band").
        # Without this, aux_dist penalizes hold-state samples forever with a +margin floor.
        # 0 = disabled (legacy 동작).
        self.aux_dist_hold_threshold_mm = float(adl.get("hold_threshold_mm", 0.0))

        # Direction-decoupled action loss (replaces plain MSE on main loss).
        # Splits position-channel target into magnitude and unit-direction supervisions:
        #   loss = α * (|a_pred| - |a_gt|)² + β * (1 - cos(a_pred, a_gt))
        # Other channels (rotation, gripper) keep plain MSE.
        ddl = direction_decoupled_loss or {}
        self.ddl_enabled = bool(ddl.get("enabled", False))
        self.ddl_pos_mag_weight = float(ddl.get("pos_mag_weight", 1.0))
        self.ddl_pos_dir_weight = float(ddl.get("pos_dir_weight", 0.5))
        self.ddl_other_weight = float(ddl.get("other_weight", 1.0))
        self.ddl_min_mag = float(ddl.get("min_mag_threshold", 0.01))

        print(f"Initializing VLM {lmm_path} with attn_implementation: {attn_implementation}")
        if lmm_path == "vision_only":
            # Vision-only backbone — no LLM, no language conditioning.
            # Backbone-agnostic: SigLIP/SigLIP2 (NaFlex 또는 fixed-res), DINOv3/v2 모두 지원.
            self.model_family = "vision_only"
            self.lmm = None
            vision_attn = attn_implementation if attn_implementation != "flash_attention_2" else "sdpa"
            vlow = vision_encoder_path.lower()
            self.vision_needs_interp_pos = False  # DINOv3/v2 + native_res 외 사용 시 True
            is_conv_backbone = False  # 기본값. else 분기에서 ResNet/ConvNext 감지 시 True로
            if "naflex" in vlow:
                self.vision_encoder = Siglip2VisionModel.from_pretrained(
                    vision_encoder_path, dtype=torch.bfloat16, attn_implementation=vision_attn
                )
                image_processor = Siglip2ImageProcessor.from_pretrained(vision_encoder_path)
            elif "siglip" in vlow:
                self.vision_encoder = SiglipVisionModel.from_pretrained(
                    vision_encoder_path, dtype=torch.bfloat16, attn_implementation=vision_attn
                )
                image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
            else:
                # DINOv3 / DINOv2 / 그 외 backbone-agnostic 경로 (AutoModel + AutoImageProcessor)
                # ResNet 등 conv backbone은 sdpa/flash_attention 미지원 → eager 강제
                is_conv_backbone = ("resnet" in vlow or "convnext" in vlow)
                _vattn = "eager" if is_conv_backbone else vision_attn
                self.vision_encoder = AutoModel.from_pretrained(
                    vision_encoder_path, dtype=torch.bfloat16,
                    attn_implementation=_vattn, trust_remote_code=True,
                )
                # AutoModel이 dual-tower 형태로 로드되면 vision_model attribute 사용
                if hasattr(self.vision_encoder, "vision_model"):
                    self.vision_encoder = self.vision_encoder.vision_model
                image_processor = AutoImageProcessor.from_pretrained(
                    vision_encoder_path, trust_remote_code=True
                )
                # ViT 계열만 interpolate_pos_encoding 필요. conv 계열은 pos embed 없음.
                self.vision_needs_interp_pos = (not is_conv_backbone)
            # input_image_size config로 processor size override (DINOv3 224→512, SigLIP2 512→768 등).
            # native 해상도와 다르면 ViT pos embed interpolation 필요.
            if input_image_size and not is_conv_backbone:
                self.vision_needs_interp_pos = True
            if input_image_size:
                _sz = int(input_image_size)
                # ConvNext/ResNet-style processor는 shortest_edge 기반, ViT-style은 height/width.
                try:
                    cur_size = image_processor.size
                    if isinstance(cur_size, dict) and "shortest_edge" in cur_size:
                        image_processor.size = {"shortest_edge": _sz}
                    else:
                        image_processor.size = {"height": _sz, "width": _sz}
                    if hasattr(image_processor, "crop_size"):
                        try:
                            image_processor.crop_size = {"height": _sz, "width": _sz}
                        except Exception:
                            pass
                except Exception:
                    pass
            self.processor = LlamaProcessorWrapper(tokenizer=None, image_processor=image_processor)
            # Use policy_hidden_size as the common backbone width (no LLM to inherit from).
            self.hidden_size = policy_hidden_size
            _venc_cfg = self.vision_encoder.config
            _venc_out_dim = getattr(_venc_cfg, "hidden_size", None)
            if _venc_out_dim is None:
                # Conv backbones (ResNet 등): hidden_sizes는 stage별 list, 마지막 stage 차원 사용
                _venc_out_dim = _venc_cfg.hidden_sizes[-1]
            self.vision_projector = nn.Sequential(
                nn.Linear(_venc_out_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            # Conv backbone (ResNet/ConvNext)은 spatial token sequence에 positional 정보가 없음.
            # ACT/DETR 스타일: 2D sincos pos enc 추가 + 4-layer transformer encoder로
            # spatial reasoning을 부여 → policy cross-attn이 "어디 token이 어디 좌표인지" 알 수 있음.
            self.is_conv_backbone = bool(is_conv_backbone)
            if self.is_conv_backbone:
                _enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=8,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.conv_feature_encoder = nn.TransformerEncoder(_enc_layer, num_layers=4)
        elif "paligemma" in lmm_path.lower():
            self.model_family = "paligemma"
            self.lmm = PaliGemmaForConditionalGeneration.from_pretrained(
                lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        elif "llama" in lmm_path.lower():
            self.model_family = "llama"
            self.lmm = LlamaForCausalLM.from_pretrained(
                lmm_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                vision_encoder_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            tokenizer = AutoTokenizer.from_pretrained(lmm_path)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
            self.processor = LlamaProcessorWrapper(tokenizer, image_processor)
            self.hidden_size = self.lmm.config.hidden_size
            self.vision_projector = nn.Sequential(
                nn.Linear(self.vision_encoder.config.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        elif "qwen" in lmm_path.lower():
            self.model_family = "qwen"
            if "qwen3.5" in lmm_path.lower() or "qwen3_5" in lmm_path.lower():
                self.lmm = AutoModelForImageTextToText.from_pretrained(
                    lmm_path, dtype=torch.bfloat16, trust_remote_code=True
                )
            else:
                self.lmm = Qwen3VLForConditionalGeneration.from_pretrained(
                    lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
                )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        
        self.backbone_mode = backbone_mode
        if backbone_mode == "frozen":
            if self.lmm is not None:
                self.lmm.requires_grad_(False)
            if self.model_family in ("llama", "vision_only"):
                self.vision_encoder.requires_grad_(False)
        elif backbone_mode == "lora":
            lora_config = lora_config or {}
            if self.model_family == "vision_only":
                # Vision-only LoRA targets SigLIP attention projections (no LLM).
                peft_config = LoraConfig(
                    r=lora_config.get("r", 16),
                    lora_alpha=lora_config.get("lora_alpha", 32),
                    lora_dropout=lora_config.get("lora_dropout", 0.05),
                    target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"]),
                    bias="none",
                )
                self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
                if hasattr(self.vision_encoder, "print_trainable_parameters"):
                    self.vision_encoder.print_trainable_parameters()
            else:
                peft_config = LoraConfig(
                    r=lora_config.get("r", 16),
                    lora_alpha=lora_config.get("lora_alpha", 32),
                    lora_dropout=lora_config.get("lora_dropout", 0.05),
                    target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.lmm = get_peft_model(self.lmm, peft_config)
                if hasattr(self.lmm, "print_trainable_parameters"):
                    self.lmm.print_trainable_parameters()
                if self.model_family == "llama":
                    self.vision_encoder.requires_grad_(False)
        elif backbone_mode == "finetune":
            if self.lmm is not None:
                self.lmm.requires_grad_(True)
            if self.model_family in ("llama", "vision_only"):
                self.vision_encoder.requires_grad_(True)
        elif backbone_mode == "last_n_unfreeze":
            if self.model_family != "vision_only":
                raise ValueError("last_n_unfreeze currently only supported for vision_only")
            self.vision_encoder.requires_grad_(False)
            # SigLIP2/DINO ViT: layers at vision_encoder.encoder.layers[i] (after vision_model unwrap above)
            ve = self.vision_encoder
            layers = None
            for path in ("encoder.layers", "layers", "vision_model.encoder.layers"):
                obj = ve
                ok = True
                for part in path.split('.'):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        ok = False; break
                if ok:
                    layers = obj; break
            if layers is None:
                raise RuntimeError("Could not locate vision encoder layers for last_n_unfreeze")
            total = len(layers)
            n = max(1, min(n_unfreeze_layers, total))
            for li in range(total - n, total):
                for p in layers[li].parameters():
                    p.requires_grad_(True)
            print(f"[backbone] last_n_unfreeze: unfroze last {n}/{total} vision layers")
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        if gradient_checkpointing:
            if self.lmm is not None:
                model_to_configure = self.lmm
                if hasattr(model_to_configure, "gradient_checkpointing_enable"):
                    model_to_configure.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                if hasattr(self.lmm, "enable_input_require_grads"):
                    self.lmm.enable_input_require_grads()
                config = self.lmm.config
                if hasattr(config, "use_cache"):
                    config.use_cache = False
            if self.model_family in ("llama", "vision_only"):
                 if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                    try:
                        self.vision_encoder.gradient_checkpointing_enable()
                    except ValueError:
                        # ResNet 등 일부 conv backbone은 GC 미지원 — frozen이라 어차피 grad 필요없음
                        pass

        self.num_queries = num_queries
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.condition_type = condition_type
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.future_image_loss_weight = future_image_loss_weight
        self.enable_future_image_loss = (future_image_loss_weight > 0)
        self.dct_loss_weight = dct_loss_weight
        self.dct_low_freq_weight = dct_low_freq_weight
        self.dct_high_freq_weight = dct_high_freq_weight
        self.dct_freq_split = dct_freq_split
        self.dct_similarity_type = dct_similarity_type

        self.spatial_loss_weight = spatial_loss_weight
        if spatial_loss_weight > 0:
            self.spatial_head = SpatialCrossAttentionHead(
                hidden_size=self.hidden_size,
                num_layers_to_use=4,
                num_heads=4,
            )
        else:
            self.spatial_head = None

        self.action_vqvae_config = action_vqvae
        if self.action_vqvae_config.get('enabled', False):
            self.action_vqvae = ActionVQVAE(
                action_dim=action_dim,
                latent_codes_per_step=3, 
                codebook_size=self.action_vqvae_config.get('codebook_size', 1024),
                hidden_size=self.action_vqvae_config.get('hidden_size', 256),
                depth=self.action_vqvae_config.get('depth', 2),
                num_heads=self.action_vqvae_config.get('num_heads', 4)
            )
        else:
            self.action_vqvae = None

        if self.enable_future_image_loss:
            print("Initializing Future Image Generator Components...")
            self.vq_model = Emu3p5VisionVQModel.from_pretrained("BAAI/Emu3.5-VisionTokenizer", trust_remote_code=True)
            self.vq_model.requires_grad_(False)
            self.vq_codebook_size = self.vq_model.config.codebook_size
            
            self.generator = ImageGeneratorTransformer(
                vocab_size=self.vq_codebook_size,
                vlm_hidden_size=self.hidden_size,
                hidden_size=generator_hidden_size,
                depth=generator_depth,
                num_heads=generator_num_heads,
                mlp_ratio=generator_mlp_ratio
            )
        else:
            self.vq_model = None
            self.generator = None

        if self.use_proprio_input_vlm:
            projector_input_dim = proprio_dim if proprio_dim is not None else action_dim
            self.proprio_dim = projector_input_dim  # eval/collator에서 6/8 분기 판단용
            if use_transformer_proprio_projector:
                self.action_projector = ActionTransformerProjector(
                    action_dim=projector_input_dim,
                    hidden_size=self.hidden_size,
                    depth=projector_depth,
                    num_heads=projector_num_heads
                )
            else:
                self.action_projector = nn.Linear(projector_input_dim, self.hidden_size)
        else:
            self.action_projector = None
        
        self.meta_queries = nn.Parameter(
            torch.randn(num_queries, self.hidden_size) * 0.02
        )
        self.meta_queries_norm = nn.RMSNorm(self.hidden_size)
        if self.condition_type == "loose":
            if use_transformer_connector:
                self.connector = ConnectorTransformer(
                    input_dim=self.hidden_size,
                    output_dim=self.hidden_size,
                    depth=connector_depth,
                    num_heads=connector_num_heads
                )
            else:
                self.connector = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size) # Project to diffusion cond dim
                )
        else:
            self.connector = None

        gen_hidden_dim = generator_hidden_size if self.enable_future_image_loss else None
        if loss_type == "regression":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionRegressionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim
                )
            elif condition_type == "loose":
                self.action_head = ActionRegressionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for regression: {condition_type}")
            self.noise_scheduler = None
        elif loss_type == "classification":
            is_vqvae = (self.action_vqvae is not None)
            
            if condition_type == "loose":
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes
                    )
                else:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False
                    )
            elif condition_type in ["tight", "soft"]:
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes,
                        gen_hidden_size=gen_hidden_dim
                    )
                else:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False,
                        gen_hidden_size=gen_hidden_dim
                    )
            else:
                raise NotImplementedError(f"Classification policy does not support {condition_type}.")
            self.noise_scheduler = None
        elif loss_type == "diffusion":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionDiffusionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim
                )
            elif condition_type == "loose":
                self.action_head = ActionDiffusionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for diffusion: {condition_type}")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if loss_type == "diffusion":
            if scheduler_type == "ddim":
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_train_timesteps,
                    clip_sample=False,
                    prediction_type="epsilon"
                )
            elif scheduler_type == "flow_match": 
                self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def forward_action_vqvae_pretrain(self, actions):
        if self.action_vqvae is None:
            raise RuntimeError("Action VQ-VAE not initialized.")
            
        actions = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
        loss = self.action_vqvae(actions)
        return loss

    def get_vlm_condition(self, input_ids, attention_mask, proprioception=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None):
        # NOTE: Even with frozen VLM (requires_grad=False on lmm params),
        # we do NOT wrap in torch.no_grad() so that gradient can flow back
        # to trainable inputs: meta_queries and action_projector.
        # Gradient checkpointing on the VLM keeps VRAM manageable.
        if self.model_family == "paligemma":
            connector_out, hidden_states = self._get_vlm_condition_paligemma(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values)
        elif self.model_family == "llama":
            connector_out, hidden_states = self._get_vlm_condition_llama(input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask)
        elif self.model_family == "qwen":
            connector_out, hidden_states = self._get_vlm_condition_qwen(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)
        elif self.model_family == "vision_only":
            connector_out, hidden_states = self._get_vlm_condition_vision_only(pixel_values, proprioception, proprio_attention_mask)
        return connector_out, hidden_states

    @property
    def _backbone(self):
        """Return the underlying transformer backbone, unwrapping PeftModel if present.

        LoRA replaces Linear layers in-place, so forward through this backbone
        still goes through LoRA adapters.
        """
        from peft import PeftModel
        lmm = self.lmm
        if isinstance(lmm, PeftModel):
            lmm = lmm.base_model.model  # unwrap LoRA wrapper
        return lmm.model  # the actual transformer backbone (e.g. Qwen3_5Model)

    def _build_qwen_mm_token_type_ids(self, input_ids):
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.lmm.config, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1

        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is None:
            video_token_id = getattr(self.lmm.config, "video_token_id", None)
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2

        return mm_token_type_ids

    def _get_vlm_condition_qwen(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw):
        B = input_ids.shape[0]
        
        backbone = self._backbone
        lmm_config = self.lmm.config
        pad_token_id = getattr(lmm_config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        inputs_embeds = backbone.get_input_embeddings()(input_ids)
        
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            proprio_ids = torch.full((B, proprioception.shape[1]), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([proprio_ids, input_ids], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries_norm(self.meta_queries).unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            queries_ids = torch.full((B, self.num_queries), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            extended_input_ids = torch.cat([input_ids, queries_ids], dim=1)
        else:
            extended_input_ids = input_ids

        mm_token_type_ids = self._build_qwen_mm_token_type_ids(extended_input_ids)

        rope_kwargs = {
            "input_ids": extended_input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask
        }

        position_ids, _ = backbone.get_rope_index(**rope_kwargs)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"] or self.spatial_loss_weight > 0)
        forward_kwargs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "output_hidden_states": output_hidden_states_flag,
        }
        outputs = backbone(**forward_kwargs)
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        # Build image token mask for spatial head
        image_token_mask = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)  # image or video tokens
        # Verify length matches hidden_states; if not, fallback to all tokens
        if hidden_states is not None:
            hs_len = hidden_states[-1].shape[1]
            if image_token_mask.shape[1] != hs_len:
                image_token_mask = torch.ones(B, hs_len, dtype=torch.bool, device=image_token_mask.device)
        self._image_token_mask = image_token_mask  # cache for spatial head
        self._image_grid_thw = image_grid_thw  # cache for spatial head (wrist masking)

        return connector_out, hidden_states

    def _get_vlm_condition_vision_only(self, pixel_values, proprioception, proprio_attention_mask):
        """Vision-only conditioning: SigLIP2 patch tokens → action head, no LLM.

        Returns hidden_states as a tuple of per-layer states; each layer state has
        [projected_vision_tokens, proprio_tokens, meta_queries] concatenated. Vision
        part varies per layer (SigLIP layer outputs); proprio + queries are static
        across layers (they bypass SigLIP). Action head's cross-attention attends
        over the full sequence per block.
        """
        pixel_values = pixel_values.to(dtype=self.vision_encoder.dtype)
        output_hidden_states_flag = (
            self.condition_type in ["tight", "soft"] or self.spatial_loss_weight > 0
        )
        fwd_kwargs = {"output_hidden_states": output_hidden_states_flag}
        if getattr(self, "vision_needs_interp_pos", False):
            fwd_kwargs["interpolate_pos_encoding"] = True
        try:
            vision_outputs = self.vision_encoder(pixel_values, **fwd_kwargs)
        except TypeError:
            fwd_kwargs.pop("interpolate_pos_encoding", None)
            vision_outputs = self.vision_encoder(pixel_values, **fwd_kwargs)
        last_hidden = vision_outputs.last_hidden_state  # (B*views, N, D_v) ViT; (B*views, D, H, W) conv

        # Conv backbones (ResNet 등): last_hidden_state는 (B, C, H, W). Flatten spatial→token 차원으로.
        # 또한 conv는 per-stage hidden_states가 서로 다른 (C, H, W)라 layer-wise 공유 projector 불가 →
        # output_hidden_states를 사용하지 않고 last_hidden만 사용 (multi-scale 기능 손실 트레이드).
        is_conv = (last_hidden.dim() == 4)
        conv_HW = None  # (H, W) for conv path — pos enc 계산용
        if is_conv:
            B_v, C, H, W = last_hidden.shape
            last_hidden = last_hidden.permute(0, 2, 3, 1).reshape(B_v, H * W, C)
            conv_HW = (H, W)

        # Infer batch size from proprio (collator pads pixel_values to B*views).
        if proprioception is not None:
            B = proprioception.shape[0]
        else:
            B = last_hidden.shape[0]

        proj_dtype = next(self.vision_projector.parameters()).dtype
        def _proj_and_reshape(hs):
            x = self.vision_projector(hs.to(dtype=proj_dtype))
            if x.shape[0] != B:
                num_views = x.shape[0] // B
                x = x.view(B, num_views, -1, x.shape[-1]).flatten(1, 2)
            return x

        if output_hidden_states_flag and not is_conv:
            projected_layers = [_proj_and_reshape(hs) for hs in vision_outputs.hidden_states]
        else:
            projected_layers = [_proj_and_reshape(last_hidden)]

        # Conv backbone: ACT/DETR 스타일 spatial 보정.
        # (1) projected feature에 2D sincos positional encoding 합산
        # (2) 4-layer transformer encoder 통과 → spatial reasoning 부여
        # 이렇게 안 하면 token sequence가 위치 정보 없이 들어가서 policy가 mean action으로 수렴.
        if is_conv and conv_HW is not None and getattr(self, "is_conv_backbone", False):
            H, W = conv_HW
            buf_name = f"_conv_pos_{H}x{W}"
            if not hasattr(self, buf_name):
                pos = _build_2d_sincos_pos_embed(H, W, self.hidden_size)
                self.register_buffer(buf_name, pos, persistent=False)
            x = projected_layers[0]
            # multi-view면 (B, V*H*W, D) 형태로 들어옴 → V 번 타일링한 pos를 더해야 함
            T_v = (x.shape[1] // (H * W)) if (x.shape[1] % (H * W) == 0) else 1
            pos = getattr(self, buf_name).to(device=x.device, dtype=x.dtype)
            if T_v > 1:
                pos = pos.repeat(T_v, 1)  # (V*H*W, D)
            x = x + pos.unsqueeze(0)
            # encoder의 weight dtype에 맞춤 (autocast 안 쓰는 경로에서 dtype mismatch 방지)
            _enc_dtype = next(self.conv_feature_encoder.parameters()).dtype
            x = self.conv_feature_encoder(x.to(dtype=_enc_dtype)).to(dtype=projected_layers[0].dtype)
            projected_layers = [x]

        # Static extra tokens (proprio + meta queries) appended to every layer.
        extras = []
        ref_dtype = projected_layers[0].dtype
        ref_device = projected_layers[0].device
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(
                proprioception.to(device=ref_device, dtype=ref_dtype)
            )
            extras.append(proprio_embeds)
        if self.condition_type != "tight":
            queries_embeds = self.meta_queries_norm(self.meta_queries).unsqueeze(0).expand(B, -1, -1).to(ref_dtype)
            extras.append(queries_embeds)
        extras = torch.cat(extras, dim=1) if extras else None

        if extras is not None:
            hidden_states = tuple(torch.cat([h, extras], dim=1) for h in projected_layers)
        else:
            hidden_states = tuple(projected_layers)

        # Conv backbone: only 1 layer available, replicate so action_head 24 blocks all get a vision input
        # (policy의 vlm_hidden_states[-len(blocks):] slicing이 1-element list에서 1개만 반환 → 1 block만 동작 방지).
        if is_conv and len(hidden_states) == 1:
            hidden_states = hidden_states * 32

        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            # Pool the meta-query slice at the last layer.
            connector_out = self.connector(hidden_states[-1][:, -self.num_queries:, :])

        # Spatial head image-token mask: only the vision-patch portion.
        if self.spatial_loss_weight > 0:
            hs_len = hidden_states[-1].shape[1]
            image_token_count = projected_layers[0].shape[1]
            image_token_mask = torch.zeros(B, hs_len, dtype=torch.bool, device=ref_device)
            image_token_mask[:, :image_token_count] = True
            self._image_token_mask = image_token_mask
            self._image_grid_thw = None

        return connector_out, hidden_states

    def _get_vlm_condition_llama(self, input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask):
        B = input_ids.shape[0]
        pixel_values = pixel_values.to(dtype=self.vision_encoder.dtype)
        
        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_feats = vision_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_feats)

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)
        
        text_embeds = self._backbone.embed_tokens(input_ids)

        proprio_embeds = None
        if self.use_proprio_input_vlm and proprioception is not None:
             proprio_embeds = self.action_projector(proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype))

        embeds_list = [image_embeds]
        image_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list = [image_mask]

        if proprio_embeds is not None:
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                p_mask = torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                mask_list.append(p_mask)
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries_norm(self.meta_queries).unsqueeze(0).expand(B, -1, -1).to(text_embeds.dtype)
            embeds_list.append(queries_embeds)
            queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            mask_list.append(queries_mask)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        combined_attention_mask = torch.cat(mask_list, dim=1)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"] or self.spatial_loss_weight > 0)
        outputs = self._backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=output_hidden_states_flag
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _get_vlm_condition_paligemma(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values):
        B = input_ids.shape[0]
        
        backbone = self._backbone

        inputs_embeds = backbone.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = backbone.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = backbone.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries_norm(self.meta_queries).unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"] )
        outputs = backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states_flag,
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _compute_gen_loss_and_feats(self, future_images, vlm_hidden_states):
        with torch.no_grad():
            future_images = future_images.to(device=self.vq_model.device, dtype=self.vq_model.dtype)
            _, _, (_, _, token_ids) = self.vq_model.encode(future_images)
            B = future_images.shape[0]
            token_ids = token_ids.view(B, -1)
        
        sos_token = torch.zeros((B, 1), dtype=token_ids.dtype, device=token_ids.device)
        gen_input = torch.cat([sos_token, token_ids[:, :-1]], dim=1)
        
        gen_logits, gen_hidden_states = self.generator(gen_input, vlm_hidden_states)
        loss_img = F.cross_entropy(gen_logits.reshape(-1, self.vq_codebook_size), token_ids.reshape(-1))
        
        return loss_img, gen_hidden_states

    def _compute_spatial_loss(self, hidden_states, spatial_targets):
        """Compute auxiliary spatial loss from VLM hidden states.

        spatial_targets layout (8D from data, we use [4:8]):
            [4:6] visibility: needle_tip_visible, trocar_visible (0 or 1)
            [6]   dist: normalized 3D distance
            [7]   phase: 0=align, 1=insert

        spatial_pred layout (4D):
            [0:2] visibility logits
            [2]   dist
            [3]   phase logit
        """
        image_token_mask = getattr(self, '_image_token_mask', None)
        if image_token_mask is None:
            B, S, H = hidden_states[-1].shape
            image_token_mask = torch.ones(B, S, dtype=torch.bool, device=hidden_states[-1].device)
        image_grid_thw = getattr(self, '_image_grid_thw', None)
        spatial_pred = self.spatial_head(hidden_states, image_token_mask, image_grid_thw=image_grid_thw)  # (B, 4)
        spatial_targets = spatial_targets.to(spatial_pred.dtype)

        # --- Visibility loss (BCE) ---
        vis_pred = spatial_pred[:, 0:2]       # (B, 2): tip_vis, trocar_vis logits
        vis_target = spatial_targets[:, 4:6]  # (B, 2): 0 or 1
        loss_vis = F.binary_cross_entropy_with_logits(vis_pred, vis_target)

        # --- Distance loss (always valid, uses 3D coords) ---
        dist_pred = spatial_pred[:, 2]
        dist_target = spatial_targets[:, 6]
        loss_dist = F.mse_loss(dist_pred, dist_target)

        # --- Phase loss (BCE) ---
        phase_pred = spatial_pred[:, 3]
        phase_target = spatial_targets[:, 7]
        loss_phase = F.binary_cross_entropy_with_logits(phase_pred, phase_target)

        spatial_loss = loss_dist + 0.1 * loss_vis + 0.1 * loss_phase
        spatial_detail = {
            "spatial/distance": loss_dist.item(),
            "spatial/visibility": loss_vis.item(),
            "spatial/phase": loss_phase.item(),
            "spatial/total": spatial_loss.item(),
        }
        return spatial_loss, spatial_detail

    def _compute_dct_loss(self, pred, target):
        B, T, D = pred.shape

        if not hasattr(self, '_dct_matrix') or self._dct_matrix.shape[0] != T or self._dct_matrix.device != pred.device:
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            
            dct_m[0, :] *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            
            self._dct_matrix = dct_m

        split_idx = max(1, int(T * self.dct_freq_split))
        freq_weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        freq_weights[:split_idx] = self.dct_low_freq_weight
        freq_weights[split_idx:] = self.dct_high_freq_weight
        freq_weights = freq_weights.view(1, T, 1)

        pred_perm = pred.permute(0, 2, 1)
        pred_dct = torch.matmul(pred_perm, self._dct_matrix.t())
        pred_dct = pred_dct.permute(0, 2, 1)

        target_perm = target.permute(0, 2, 1)
        target_dct = torch.matmul(target_perm, self._dct_matrix.t())
        target_dct = target_dct.permute(0, 2, 1)

        sim_type = self.dct_similarity_type
        if sim_type == "mse":
            diff = (pred_dct - target_dct) ** 2
            return (diff * freq_weights).mean()
        elif sim_type == "mae":
            diff = (pred_dct - target_dct).abs()
            return (diff * freq_weights).mean()
        elif sim_type == "cosine":
            pred_norm = torch.nn.functional.normalize(pred_dct, dim=-1)
            target_norm = torch.nn.functional.normalize(target_dct, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1, keepdim=True)
            cos_dist = 1.0 - cos_sim
            return (cos_dist * freq_weights).mean()
        else:
            raise ValueError(f"Unknown dct_similarity_type: {sim_type!r}. "
                             f"Options are: 'mse', 'mae', 'cosine'.")

    def forward(self, input_ids=None, attention_mask=None, actions=None, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, future_images=None, spatial_targets=None, action_weights=None, needle_tip_pos=None, trocar_entry_pos=None, task=None):
        if task == "action_vqvae_pretrain":
            return self.forward_action_vqvae_pretrain(actions), {}

        if self.loss_type == "regression":
            return self._forward_regression(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, spatial_targets
            )
        elif self.loss_type == "classification":
            return self._forward_classification(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, spatial_targets
            )
        elif self.loss_type == "diffusion":
            return self._forward_diffusion(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, spatial_targets,
                action_weights=action_weights,
                needle_tip_pos=needle_tip_pos,
                trocar_entry_pos=trocar_entry_pos,
            )

    def _forward_classification(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, spatial_targets=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )
        
        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)

        policy_history = history_actions if self.use_action_input_policy else None
        
        if self.condition_type in ["tight", "soft"]:
            if self.enable_future_image_loss:
                pred_logits = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred_logits = self.action_head(hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred_logits = self.action_head(cond_input, history_actions=policy_history)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)
        
        pred_action_continuous = None
        loss = 0.0

        if self.action_vqvae is not None:
             with torch.no_grad():
                 self.action_vqvae.eval()
                 actions_input = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
                 _, indices, _ = self.action_vqvae.encode(actions_input)
             
             loss = F.cross_entropy(
                 pred_logits.reshape(-1, self.action_vqvae.codebook_size), 
                 indices.reshape(-1)
             )

             if self.dct_loss_weight > 0:
                 probs = F.softmax(pred_logits, dim=-1)
                 pred_action_continuous = self.action_vqvae.decode_probs(probs)
        else:
            logits = pred_logits
            pose_logits = logits[:, :, :self.action_dim - 1, :]
            gripper_logits = logits[:, :, -1:, :2]
            
            gt_pose = torch.clamp(actions[:, :, :6], -1, 1)
            gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()
            
            gt_gripper = torch.clamp(actions[:, :, 6:7], -1, 1)
            gt_gripper_idx = ((gt_gripper + 1) / 2).round().long() # 0 or 1

            loss_pose = F.cross_entropy(pose_logits.reshape(-1, self.num_bins), gt_pose_idx.reshape(-1))
            loss_gripper = F.cross_entropy(gripper_logits.reshape(-1, 2), gt_gripper_idx.reshape(-1))
            
            loss = (loss_pose + loss_gripper) / 2.0

            if self.dct_loss_weight > 0:
                pose_probs = F.softmax(pose_logits, dim=-1)
                bin_centers = torch.linspace(-1, 1, self.num_bins, device=actions.device, dtype=pose_probs.dtype)
                pred_pose = torch.sum(pose_probs * bin_centers, dim=-1)

                gripper_probs = F.softmax(gripper_logits, dim=-1)
                p1 = gripper_probs[..., 1]
                pred_gripper = -1.0 + 2.0 * p1
                
                pred_action_continuous = torch.cat([pred_pose, pred_gripper], dim=-1)

        loss_dict = {"loss/main": loss.item() if isinstance(loss, torch.Tensor) else loss}

        if self.dct_loss_weight > 0 and pred_action_continuous is not None:
             loss_dct = self._compute_dct_loss(pred_action_continuous.float(), actions.float())
             loss_dict["loss/dct"] = loss_dct.item()
             loss = loss + self.dct_loss_weight * loss_dct

        if self.future_image_loss_weight > 0:
            loss_dict["loss/future_image"] = loss_img.item() if isinstance(loss_img, torch.Tensor) else loss_img
            loss = loss + self.future_image_loss_weight * loss_img
        if self.spatial_loss_weight > 0 and self.spatial_head is not None and spatial_targets is not None and hidden_states is not None:
            spatial_loss, spatial_detail = self._compute_spatial_loss(hidden_states, spatial_targets)
            loss_dict.update(spatial_detail)
            loss = loss + self.spatial_loss_weight * spatial_loss
        return loss, loss_dict

    def _forward_regression(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, spatial_targets=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)
        
        policy_history = history_actions if self.use_action_input_policy else None
        
        if self.condition_type in ["tight", "soft"]:
             if self.enable_future_image_loss:
                 pred_actions = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
             else:
                 pred_actions = self.action_head(hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
             cond_input = connector_out.mean(dim=1)
             pred_actions = self.action_head(cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")

        if actions.ndim == 2: actions = actions.unsqueeze(1)
        loss = F.mse_loss(pred_actions, actions)
        loss_dict = {"loss/main": loss.item()}

        if self.dct_loss_weight > 0:
            loss_dct = self._compute_dct_loss(pred_actions.float(), actions.float())
            loss_dict["loss/dct"] = loss_dct.item()
            loss = loss + self.dct_loss_weight * loss_dct

        if self.future_image_loss_weight > 0:
            loss_dict["loss/future_image"] = loss_img.item() if isinstance(loss_img, torch.Tensor) else loss_img
            loss = loss + self.future_image_loss_weight * loss_img
        if self.spatial_loss_weight > 0 and self.spatial_head is not None and spatial_targets is not None and hidden_states is not None:
            spatial_loss, spatial_detail = self._compute_spatial_loss(hidden_states, spatial_targets)
            loss_dict.update(spatial_detail)
            loss = loss + self.spatial_loss_weight * spatial_loss
        return loss, loss_dict

    def _forward_diffusion(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, spatial_targets=None, action_weights=None, needle_tip_pos=None, trocar_entry_pos=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=proprioception, proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)
        noise = torch.randn_like(actions)
        B = actions.shape[0]
        
        if self.scheduler_type == "flow_match":
            sigmas = torch.rand((B,), device=actions.device)
            sigmas_expanded = sigmas.view(B, *([1] * (actions.ndim - 1)))
            noisy_actions = (1.0 - sigmas_expanded) * actions + sigmas_expanded * noise
            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            timesteps = sigmas * self.noise_scheduler.config.num_train_timesteps
            target = noise - actions
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            target = noise
            
        policy_history = history_actions if self.use_action_input_policy else None

        if self.condition_type in ["tight", "soft"]:
            if self.enable_future_image_loss:
                pred = self.action_head(noisy_actions, timesteps, hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred = self.action_head(noisy_actions, timesteps, hidden_states, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred = self.action_head(noisy_actions, timesteps, cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        # Near-goal sample weighting: boost loss when needle tip is close to trocar.
        # Reuses GT positions already passed for aux_dist_loss. (B,) per-sample weight.
        near_goal_w = None
        cur_dist_mm = None
        if (self.aux_dist_near_goal_scale_mm > 0
                and needle_tip_pos is not None and trocar_entry_pos is not None):
            tip_b = needle_tip_pos.float()      # (B, 3) in mm
            trocar_b = trocar_entry_pos.float() # (B, 3) in mm
            cur_dist_mm = torch.norm(tip_b - trocar_b, dim=-1)  # (B,) mm
            boost = self.aux_dist_near_goal_scale_mm / (cur_dist_mm + 1.0)
            near_goal_w = (1.0 + boost).clamp(max=1.0 + self.aux_dist_near_goal_max_boost)

        # Combine action_weights × near_goal_w for main MSE (only if apply_to_main_loss).
        effective_w = action_weights
        if near_goal_w is not None and self.aux_dist_apply_to_main:
            effective_w = near_goal_w if action_weights is None else (action_weights * near_goal_w)

        if self.ddl_enabled:
            # Direction-decoupled loss for position channels (first 3 dims).
            # Other channels (rotation + gripper) use plain squared error.
            pos_pred = pred[..., :3]
            pos_target = target[..., :3]
            mag_pred = torch.norm(pos_pred, dim=-1)                      # (B, T)
            mag_target = torch.norm(pos_target, dim=-1)                  # (B, T)
            mag_diff_sq = (mag_pred - mag_target) ** 2                   # (B, T)

            # F.cosine_similarity handles small-norm cases safely (NaN-free with eps).
            # Mask out frames where either pred or target magnitude is too small —
            # direction is undefined and gradient explodes through the division.
            cos_sim = F.cosine_similarity(pos_pred, pos_target, dim=-1, eps=1e-3)  # (B, T)
            dir_loss_raw = 1.0 - cos_sim                                 # (B, T) ∈ [0, 2]
            valid = (
                (mag_target > self.ddl_min_mag)
                & (mag_pred.detach() > self.ddl_min_mag)                 # detach: mask doesn't backprop
            ).float()
            dir_loss = dir_loss_raw * valid                              # (B, T)

            other_pred = pred[..., 3:]
            other_target = target[..., 3:]
            other_diff_sq = ((other_pred - other_target) ** 2).mean(dim=-1)  # (B, T)

            per_frame = (
                self.ddl_pos_mag_weight * mag_diff_sq
                + self.ddl_pos_dir_weight * dir_loss
                + self.ddl_other_weight * other_diff_sq
            )                                                              # (B, T)

            if effective_w is not None:
                w_b = effective_w.view(B, *([1] * (per_frame.ndim - 1)))
                loss = (w_b * per_frame).mean()
            else:
                loss = per_frame.mean()
            loss_dict = {
                "loss/main": loss.item(),
                "loss/main_pos_mag": mag_diff_sq.mean().item(),
                "loss/main_pos_dir": dir_loss.mean().item(),
                "loss/main_other": other_diff_sq.mean().item(),
            }
        else:
            if effective_w is not None:
                w = effective_w.view(B, *([1] * (pred.ndim - 1)))  # (B, 1, 1) or (B, 1)
                loss = (w * (pred - target) ** 2).mean()
            else:
                loss = F.mse_loss(pred, target)
            loss_dict = {"loss/main": loss.item()}

        # Reconstruct predicted x_start (clean action) — needed by DCT and aux distance losses.
        pred_x_start = None
        need_x_start = self.dct_loss_weight > 0 or (
            self.aux_dist_enabled and self.aux_dist_weight > 0
            and needle_tip_pos is not None and trocar_entry_pos is not None
        )
        if need_x_start:
            if self.scheduler_type == "flow_match":
                pred_x_start = noisy_actions - sigmas_expanded * pred
            elif self.scheduler_type == "ddim":
                def view_right(t):
                    while t.ndim < pred.ndim:
                        t = t.unsqueeze(-1)
                    return t
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=pred.device, dtype=pred.dtype)
                alpha_prod_t = alphas_cumprod[timesteps]
                pred_x_start = (noisy_actions - view_right((1 - alpha_prod_t).sqrt()) * pred) / view_right(alpha_prod_t.sqrt())

        if self.dct_loss_weight > 0 and pred_x_start is not None:
            loss_dct = self._compute_dct_loss(pred_x_start.float(), actions.float())
            loss_dict["loss/dct"] = loss_dct.item()
            loss = loss + self.dct_loss_weight * loss_dct

        # --- Auxiliary distance loss (sim-only) ---
        # Penalize predicted action chunks that increase needle-tip → trocar distance.
        # Uses small-angle approx: tip displacement ≈ ee_pos delta.
        if (self.aux_dist_enabled and self.aux_dist_weight > 0
                and needle_tip_pos is not None and trocar_entry_pos is not None
                and pred_x_start is not None):
            pred_pos_norm = pred_x_start[..., :3].float()                         # (B, T, 3) in [-1, 1]
            denorm = self.aux_pos_denorm_mm.to(pred_pos_norm.device).view(1, 1, 3)
            pred_pos_mm = pred_pos_norm * denorm                                  # (B, T, 3) mm
            tip = needle_tip_pos.float().unsqueeze(1)                             # (B, 1, 3)
            trocar = trocar_entry_pos.float().unsqueeze(1)                        # (B, 1, 3)
            cum_disp = torch.cumsum(pred_pos_mm, dim=1)                           # (B, T, 3)
            pred_tip_chunk = tip + cum_disp                                       # (B, T, 3)
            cur_dist = torch.norm(tip - trocar, dim=-1)                           # (B, 1)
            pred_dist = torch.norm(pred_tip_chunk - trocar, dim=-1)               # (B, T)
            # Hold-aware margin (soft ramp): linear interp from 0 (at cur_dist=0)
            # to margin_mm (at cur_dist >= hold_thr). Removes the discontinuity that
            # caused gnorm explosion in #27 hard-threshold variant.
            if self.aux_dist_hold_threshold_mm > 0:
                ratio = (cur_dist / self.aux_dist_hold_threshold_mm).clamp(0.0, 1.0)
                margin_eff = self.aux_dist_margin_mm * ratio                       # (B, 1)
            else:
                margin_eff = self.aux_dist_margin_mm                               # scalar (legacy)
            aux_per_sample = F.relu(pred_dist - cur_dist + margin_eff).mean(dim=-1)  # (B,)
            if near_goal_w is not None:
                loss_aux = (near_goal_w * aux_per_sample).mean()
            else:
                loss_aux = aux_per_sample.mean()
            loss_dict["loss/aux_dist"] = loss_aux.item()
            loss = loss + self.aux_dist_weight * loss_aux

        if self.future_image_loss_weight > 0:
            loss_dict["loss/future_image"] = loss_img.item() if isinstance(loss_img, torch.Tensor) else loss_img
            loss = loss + self.future_image_loss_weight * loss_img
        if self.spatial_loss_weight > 0 and self.spatial_head is not None and spatial_targets is not None and hidden_states is not None:
            spatial_loss, spatial_detail = self._compute_spatial_loss(hidden_states, spatial_targets)
            loss_dict.update(spatial_detail)
            loss = loss + self.spatial_loss_weight * spatial_loss
        return loss, loss_dict

    @torch.no_grad()
    def predict_spatial(self, hidden_states):
        """Run spatial head on VLM hidden states. Returns dict or None."""
        if self.spatial_head is None or hidden_states is None:
            return None
        image_token_mask = getattr(self, '_image_token_mask', None)
        if image_token_mask is None:
            B, S, H = hidden_states[-1].shape
            image_token_mask = torch.ones(B, S, dtype=torch.bool, device=hidden_states[-1].device)
        image_grid_thw = getattr(self, '_image_grid_thw', None)
        raw = self.spatial_head(hidden_states, image_token_mask, image_grid_thw=image_grid_thw)  # (B, 4)
        pred = raw[0].float().cpu().numpy()  # single sample
        tip_vis = torch.sigmoid(raw[0, 0]).item()
        trocar_vis = torch.sigmoid(raw[0, 1]).item()
        phase = torch.sigmoid(raw[0, 3]).item()
        return {
            "tip_visible": tip_vis,         # probability
            "trocar_visible": trocar_vis,   # probability
            "dist_norm": pred[2],           # normalized distance
            "phase": phase,                 # probability (0=align, 1=insert)
        }

    @torch.no_grad()
    def predict_action(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, return_spatial=False):
        B = input_ids.shape[0]

        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )
        
        policy_history = history_actions if self.use_action_input_policy else None
        gen_hidden_states = None
        if self.enable_future_image_loss and self.condition_type in ["tight", "soft"]:
             num_img_tokens = 256 
             curr_ids = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
             gen_context = hidden_states
             for _ in range(num_img_tokens):
                 logits, _ = self.generator(curr_ids, gen_context)
                 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                 curr_ids = torch.cat([curr_ids, next_token], dim=1)
             gen_input = curr_ids[:, :-1]
             _, gen_hidden_states = self.generator(gen_input, gen_context)

        if self.loss_type == "regression":
            if self.condition_type in ["tight", "soft"]:
                if self.enable_future_image_loss:
                    action = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                else:
                    action = self.action_head(hidden_states, history_actions=policy_history)
            elif self.condition_type == "loose":
                cond_input = connector_out.mean(dim=1)
                action = self.action_head(cond_input, history_actions=policy_history)
            if action.ndim == 2 and self.num_actions > 1:
                action = action.view(action.shape[0], self.num_actions, self.action_dim)
            action = action.to(dtype=self.lmm.dtype)

        elif self.loss_type == "classification":
            if self.action_vqvae is not None:
                if self.condition_type in ["tight", "soft"]:
                    if self.enable_future_image_loss:
                        logits = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                    else:
                        logits = self.action_head(hidden_states, history_actions=policy_history)
                else:
                    cond_input = connector_out.mean(dim=1)
                    logits = self.action_head(cond_input, history_actions=policy_history)

                indices = torch.argmax(logits, dim=-1) # (B, T, Latent_Codes)
                action = self.action_vqvae.decode_indices(indices)
                action = action.to(dtype=self.lmm.dtype)
            else:
                if self.condition_type in ["tight", "soft"]:
                     if self.enable_future_image_loss:
                         logits = self.action_head(hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                     else:
                         logits = self.action_head(hidden_states, history_actions=policy_history)
                else:
                     cond_input = connector_out.mean(dim=1)
                     logits = self.action_head(cond_input, history_actions=policy_history)
                pose_logits = logits[:, :, :self.action_dim - 1, :]
                gripper_logits = logits[:, :, -1:, :2]
                pose_idx = torch.argmax(pose_logits, dim=-1)
                gripper_idx = torch.argmax(gripper_logits, dim=-1)
                pose_pred = (pose_idx.float() / (self.num_bins - 1)) * 2 - 1
                gripper_pred = gripper_idx.float() * 2 - 1
                action = torch.cat([pose_pred, gripper_pred], dim=-1).to(dtype=self.lmm.dtype)

        elif self.loss_type == "diffusion":
            # vision_only: self.lmm is None — fall back to vision_encoder dtype.
            _ref_dtype = self.lmm.dtype if self.lmm is not None else self.vision_encoder.dtype
            action = torch.randn(B, self.num_actions, self.action_dim, device=input_ids.device).to(_ref_dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), t, device=input_ids.device)
                if self.scheduler_type != "flow_match": timesteps = timesteps.long()
                if self.condition_type in ["tight", "soft"]:
                    if self.enable_future_image_loss:
                        output = self.action_head(action, timesteps, hidden_states, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                    else:
                        output = self.action_head(action, timesteps, hidden_states, history_actions=policy_history)
                else:
                    cond_input = connector_out.mean(dim=1)
                    output = self.action_head(action, timesteps, cond_input, history_actions=policy_history)

                action = self.noise_scheduler.step(output, t, action).prev_sample
                action = action.to(dtype=_ref_dtype)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if return_spatial:
            spatial_pred = self.predict_spatial(hidden_states)
            return action, spatial_pred
        return action

    @torch.no_grad()
    def predict_image(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, max_new_tokens=1024):
        _, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, 
            proprioception=proprioception,
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, 
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )
        gen_vlm_ctx = hidden_states
        
        curr_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            logits, _ = self.generator(curr_ids, gen_vlm_ctx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        generated_tokens = curr_ids[:, 1:]
        H_latent = int(generated_tokens.shape[1]**0.5)
        decoded_images = self.vq_model.decode_code(generated_tokens, shape=(input_ids.shape[0], H_latent, H_latent))
        return decoded_images

if __name__ == "__main__":
    print("Testing VLANeXt Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Initialize Model (Minimal Config)
    model = VLANeXt(
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        action_dim=7, num_actions=4, num_history=2,
        backbone_mode="finetune", gradient_checkpointing=False
    ).to(device, dtype)
    processor = model.processor

    def run_test(modality="image"):
        print(f"\n=== Testing {modality.capitalize()} ===")
        B = 2
        # Dummy Data
        img = Image.new('RGB', (64, 64), color='red')
        media = [img] * B if modality == "image" else [[img]*8] * B
        content_key = "image" if modality == "image" else "video"
        
        # Process
        msgs = [[{"role": "user", "content": [{"type": content_key, content_key: m}, {"type": "text", "text": "Task."}]}] for m in media]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, **{f"{modality}s": media}, padding=True, return_tensors="pt")
        
        # Move to device & cast
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in ["pixel_values", "pixel_values_videos"]:
            if k in inputs: inputs[k] = inputs[k].to(dtype)
            
        # Filter valid args for forward
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        fwd_args = {k: v for k, v in inputs.items() if k in valid_keys}

        # Tensors
        act_gt = torch.randn(B, 4, 7, device=device, dtype=dtype)
        proprio = torch.randn(B, 2, 7, device=device, dtype=dtype)
        hist_act = torch.randn(B, 2, 7, device=device, dtype=dtype)

        # Tests
        print(f"Action Gen Loss: {model(actions=act_gt, proprioception=proprio, history_actions=hist_act, **fwd_args).item():.4f}")
        print(f"Action Pred Shape: {model.predict_action(proprioception=proprio, history_actions=hist_act, **fwd_args).shape}")

    run_test("image")
    run_test("video")
    print("\nTest Passed!")
