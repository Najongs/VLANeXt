"""
Diffusion Policy (Chi et al. 2023) baseline using a Conditional 1D U-Net for action chunk denoising.

Compatible with `scripts/train.py` (model_type: 'dp') and `scripts/sim_eval.py:predict_action`.
- Forward signature mirrors VLANeXt.
- Uses same dataset (sim_act_align.py) and same SigLIP2 image processor for inputs.
- Vision backbone: ResNet18 (single-view), global-avg-pooled.
- Conditioning: cat(vision_global, proprio_flat) → FiLM into 1D U-Net.
- Loss: epsilon prediction (DDPM).
- Inference: DDIM (configurable num_inference_timesteps).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from transformers import SiglipImageProcessor
from torchvision.models import resnet18

try:
    from .rt2_like_baseline import LlamaProcessorWrapper
except ImportError:
    class LlamaProcessorWrapper:
        def __init__(self, tokenizer, image_processor):
            self.tokenizer = tokenizer
            self.image_processor = image_processor


# ─── building blocks ─────────────────────────────────────────────────────────


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning (scale & shift from cond)."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel: int = 5, n_groups: int = 8):
        super().__init__()
        self.b1 = Conv1dBlock(in_ch, out_ch, kernel, n_groups)
        self.b2 = Conv1dBlock(out_ch, out_ch, kernel, n_groups)
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.b1(x)                                          # (B, out, T)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)    # each (B, out)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.b2(h)
        return h + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """Slim conditional 1D U-Net for action chunk denoising. Operates on (B, T, action_dim)."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        dsed = diffusion_step_embed_dim
        cond_total = dsed + cond_dim

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # down path
        self.downs = nn.ModuleList()
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        for ind, (in_ch, out_ch) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_ch, out_ch, cond_total, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_ch, out_ch, cond_total, kernel_size, n_groups),
                nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1) if not is_last else nn.Identity(),
            ]))

        mid_dim = down_dims[-1]
        self.mid_1 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_total, kernel_size, n_groups)
        self.mid_2 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_total, kernel_size, n_groups)

        # ups: one fewer level than downs (first down absorbs input projection).
        # All ups upsample by 2 — they invert the strided downs (down idx 0 .. len-2).
        # The last down (idx len-1) is Identity, has no matching up; final_conv brings
        # channels back to input_dim. Skip from that last down is consumed by first up.
        self.ups = nn.ModuleList()
        for ind, (in_ch, out_ch) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                ConditionalResidualBlock1D(out_ch * 2, in_ch, cond_total, kernel_size, n_groups),
                ConditionalResidualBlock1D(in_ch, in_ch, cond_total, kernel_size, n_groups),
                nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(all_dims[1], all_dims[1], kernel_size, n_groups),
            nn.Conv1d(all_dims[1], input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """sample (B, T, action_dim), timestep (B,), global_cond (B, cond_dim) -> (B, T, action_dim)"""
        x = sample.transpose(1, 2)  # (B, action_dim, T)
        t_emb = self.time_embed(timestep.to(global_cond.dtype))
        cond = torch.cat([t_emb, global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.downs:
            x = res1(x, cond); x = res2(x, cond)
            skips.append(x)
            x = down(x)

        x = self.mid_1(x, cond); x = self.mid_2(x, cond)

        for res1, res2, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond); x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)            # (B, action_dim, T)
        return x.transpose(1, 2)          # (B, T, action_dim)


# ─── Diffusion Policy ────────────────────────────────────────────────────────


class DiffusionPolicy(nn.Module):
    """Vision (ResNet18) + Proprio (history) → global cond → ConditionalUnet1D → action chunk (DDPM/DDIM)."""

    def __init__(
        self,
        action_dim: int = 6,
        num_actions: int = 8,           # chunk_size = future_len
        num_history: int = 8,
        vision_feat_dim: int = 512,      # ResNet18 layer4 out
        proprio_emb_dim: int = 64,
        diffusion_step_embed_dim: int = 128,
        unet_down_dims=(256, 512, 1024),
        unet_kernel_size: int = 5,
        n_groups: int = 8,
        num_train_timesteps: int = 100,
        num_inference_timesteps: int = 16,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        vision_pretrained: bool = True,
        vision_encoder_path: str = "google/siglip2-so400m-patch16-512",
        use_proprio_input_vlm: bool = True,
        **_unused,
    ):
        super().__init__()
        # === Vision backbone ===
        weights = "DEFAULT" if vision_pretrained else None
        net = resnet18(weights=weights)
        # use up to layer4 + adaptive avg pool to (1,1)
        self.vision_backbone = nn.Sequential(*list(net.children())[:-2])
        self.vision_pool = nn.AdaptiveAvgPool2d(1)

        # === Proprio embed ===
        self.proprio_embed = nn.Linear(action_dim, proprio_emb_dim)
        proprio_total = proprio_emb_dim * max(num_history, 1)

        # === Global cond dim ===
        cond_dim = vision_feat_dim + proprio_total

        # === U-Net ===
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=n_groups,
        )

        # === Schedulers (train vs inference) ===
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=True,
        )
        self.eval_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )

        # === Config attrs (train.py / sim_eval.py compatibility) ===
        self.action_dim = action_dim
        self.proprio_dim = action_dim  # eval bridge reads this; default 8 would add sensor → mismatch
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = False
        self.spatial_head = None
        self.loss_type = "dp_diffusion"
        image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
        self.processor = LlamaProcessorWrapper(tokenizer=None, image_processor=image_processor)

    # ---------------------------------------------------------------- helpers

    def _global_cond(self, pixel_values: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Vision (B*V, 3, H, W) + proprio (B, T_hist, action_dim) → (B, cond_dim)."""
        B = proprio.shape[0]
        feats = self.vision_backbone(pixel_values)               # (B*V, 512, H', W')
        feats = self.vision_pool(feats).flatten(1)               # (B*V, 512)
        V = feats.shape[0] // B
        if V > 1:
            feats = feats.view(B, V, -1).mean(dim=1)             # avg across views (single view = no-op)
        else:
            feats = feats.view(B, -1)
        proprio_emb = self.proprio_embed(proprio)                # (B, T_hist, proprio_emb_dim)
        proprio_flat = proprio_emb.flatten(1)                    # (B, T_hist * proprio_emb_dim)
        return torch.cat([feats, proprio_flat], dim=-1)

    # ---------------------------------------------------------------- train

    def forward(
        self,
        input_ids=None, attention_mask=None,
        actions=None, proprioception=None, history_actions=None,
        proprio_attention_mask=None,
        pixel_values=None, pixel_values_videos=None,
        image_grid_thw=None, video_grid_thw=None,
        **kwargs,
    ):
        assert pixel_values is not None and actions is not None
        B, T, A = actions.shape

        proprio = (proprioception if proprioception is not None
                   else torch.zeros(B, self.num_history, A, device=actions.device, dtype=actions.dtype))
        proprio = proprio.to(actions.dtype)

        global_cond = self._global_cond(pixel_values, proprio)   # (B, cond_dim)

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=actions.device).long()
        noisy_actions = self.train_scheduler.add_noise(actions, noise, timesteps)

        pred = self.unet(noisy_actions, timesteps, global_cond)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        loss = F.mse_loss(pred.float(), target.float())
        return loss, {"loss_total": loss.detach(), "loss_mse": loss.detach()}

    # ---------------------------------------------------------------- eval

    @torch.no_grad()
    def predict_action(
        self,
        input_ids=None, attention_mask=None,
        proprioception=None, history_actions=None,
        proprio_attention_mask=None,
        pixel_values=None, pixel_values_videos=None,
        image_grid_thw=None, video_grid_thw=None,
        return_spatial: bool = False,
    ):
        assert pixel_values is not None
        model_dtype = self.proprio_embed.weight.dtype
        if proprioception is not None:
            B = proprioception.shape[0]
            proprio = proprioception.to(model_dtype)
        else:
            B = pixel_values.shape[0]
            proprio = torch.zeros(B, self.num_history, self.action_dim,
                                  device=pixel_values.device, dtype=model_dtype)

        global_cond = self._global_cond(pixel_values.to(model_dtype), proprio)

        sample = torch.randn(B, self.num_actions, self.action_dim,
                             device=pixel_values.device, dtype=model_dtype)
        self.eval_scheduler.set_timesteps(self.num_inference_timesteps, device=pixel_values.device)
        for t in self.eval_scheduler.timesteps:
            t_batch = torch.full((B,), t.item(), device=pixel_values.device, dtype=torch.long)
            pred = self.unet(sample, t_batch, global_cond)
            sample = self.eval_scheduler.step(pred, t, sample).prev_sample

        if return_spatial:
            return sample, None
        return sample
