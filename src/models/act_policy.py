"""
ACT (Action Chunking Transformer) baseline for the same VLANeXt train/eval pipeline.

Compatible with `scripts/train.py` (model_type: 'act') and `scripts/sim_eval.py:predict_action`.
- Forward signature mirrors VLANeXt: (input_ids, attention_mask, actions, proprioception,
  history_actions, pixel_values, **kwargs) -> (loss, loss_dict)
- predict_action signature mirrors VLANeXt.predict_action.
- Uses same dataset (sim_act_align.py) and same SigLIP2 image processor for inputs.
  Vision backbone here is a ResNet18 (lightweight) — Aloha-style.

Reference: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao 2023).
This is a minimal port — single-view, no CVAE-style style token tricks beyond the standard latent z.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipImageProcessor
from torchvision.models import resnet18

try:
    from .rt2_like_baseline import LlamaProcessorWrapper
except ImportError:
    class LlamaProcessorWrapper:
        def __init__(self, tokenizer, image_processor):
            self.tokenizer = tokenizer
            self.image_processor = image_processor


def _sinusoidal_pos_embed(num_pos: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """1D sinusoidal positional embedding (num_pos, dim)."""
    pe = torch.zeros(num_pos, dim, device=device, dtype=dtype)
    position = torch.arange(0, num_pos, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ResNet18Backbone(nn.Module):
    """ResNet18 minus avgpool/fc. Outputs (B, C=512, H', W') spatial features."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = resnet18(weights=weights)
        self.net = nn.Sequential(*list(net.children())[:-2])
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, C, H/32, W/32)
        return self.net(x)


class ACTPolicy(nn.Module):
    """ACT — vision Transformer encoder + CVAE latent + Transformer decoder over action queries."""

    def __init__(
        self,
        action_dim: int = 6,
        num_actions: int = 8,           # action chunk size (= future_len)
        num_history: int = 8,           # proprio history len
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        vision_pretrained: bool = True,
        # processor (for eval-bridge image preprocessing compatibility)
        vision_encoder_path: str = "google/siglip2-so400m-patch16-512",
        # eval-bridge flags
        use_proprio_input_vlm: bool = True,
        **_unused,
    ):
        super().__init__()

        # === Vision backbone ===
        self.vision_backbone = ResNet18Backbone(pretrained=vision_pretrained)
        self.vision_proj = nn.Conv2d(self.vision_backbone.feature_dim, hidden_dim, kernel_size=1)

        # === Proprio projector ===
        self.proprio_proj = nn.Linear(action_dim, hidden_dim)

        # === Latent z (CVAE) ===
        self.latent_dim = latent_dim
        self.cvae_cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cvae_action_proj = nn.Linear(action_dim, hidden_dim)
        cvae_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, num_layers=num_encoder_layers)
        self.cvae_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.cvae_to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # === Main encoder (vision tokens + proprio token + z token) ===
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # === Decoder (action queries cross-attend to encoder output) ===
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.action_queries = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.02)

        self.action_head = nn.Linear(hidden_dim, action_dim)

        # === Cached / configured attributes (used by train.py + sim_eval.py) ===
        self.action_dim = action_dim
        self.proprio_dim = action_dim  # eval bridge reads this to size proprio (default 8 → would add sensor)
        self.num_actions = num_actions
        self.num_history = num_history
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = False
        self.spatial_head = None  # ACT does not predict spatial aux
        self.loss_type = "act"
        # processor (vision-only — no tokenizer)
        image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
        self.processor = LlamaProcessorWrapper(tokenizer=None, image_processor=image_processor)

    # ---------------------------------------------------------------- helpers

    def _encode_vision(self, pixel_values: torch.Tensor, B: int) -> torch.Tensor:
        """pixel_values (B*V, 3, H, W) -> vision tokens (B, V*Hp*Wp, hidden_dim)."""
        feats = self.vision_backbone(pixel_values)            # (B*V, 512, Hp, Wp)
        feats = self.vision_proj(feats)                       # (B*V, hidden_dim, Hp, Wp)
        BV, C, Hp, Wp = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)             # (B*V, Hp*Wp, hidden_dim)
        V = BV // B
        tokens = tokens.view(B, V * Hp * Wp, C)
        return tokens

    def _encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """(B, T_hist, action_dim) -> (B, T_hist, hidden_dim) with sinusoidal time pos."""
        x = self.proprio_proj(proprio)
        pe = _sinusoidal_pos_embed(x.shape[1], x.shape[-1], device=x.device, dtype=x.dtype)
        return x + pe.unsqueeze(0)

    def _cvae_encode(self, actions: torch.Tensor, proprio_tokens: torch.Tensor) -> tuple:
        """Encode (proprio, action_chunk) into latent z. Returns (z, mu, logvar)."""
        B = actions.shape[0]
        action_tokens = self.cvae_action_proj(actions)        # (B, T_act, hidden_dim)
        cls = self.cvae_cls.expand(B, -1, -1)                 # (B, 1, hidden_dim)
        seq = torch.cat([cls, proprio_tokens, action_tokens], dim=1)
        pe = _sinusoidal_pos_embed(seq.shape[1], seq.shape[-1], device=seq.device, dtype=seq.dtype)
        seq = seq + pe.unsqueeze(0)
        out = self.cvae_encoder(seq)
        cls_out = out[:, 0]                                   # (B, hidden_dim)
        mu = self.cvae_to_mu(cls_out)
        logvar = self.cvae_to_logvar(cls_out)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def _decode(self, vision_tokens, proprio_tokens, z) -> torch.Tensor:
        B = vision_tokens.shape[0]
        z_token = self.z_proj(z).unsqueeze(1)                 # (B, 1, hidden_dim)
        enc_in = torch.cat([z_token, proprio_tokens, vision_tokens], dim=1)
        pe = _sinusoidal_pos_embed(enc_in.shape[1], enc_in.shape[-1], device=enc_in.device, dtype=enc_in.dtype)
        memory = self.encoder(enc_in + pe.unsqueeze(0))       # (B, S, hidden_dim)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder(queries, memory)               # (B, num_actions, hidden_dim)
        return self.action_head(dec_out)                      # (B, num_actions, action_dim)

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
        assert pixel_values is not None and actions is not None, "actions+pixel_values required for training"
        B = actions.shape[0]
        # Vision
        vision_tokens = self._encode_vision(pixel_values, B)
        # Proprio (fallback to zeros if dataset didn't supply, to keep shapes uniform)
        if proprioception is None:
            proprio = torch.zeros(B, self.num_history, self.action_dim,
                                  device=actions.device, dtype=actions.dtype)
        else:
            proprio = proprioception.to(actions.dtype)
        proprio_tokens = self._encode_proprio(proprio)
        # CVAE (only at train time — GT actions encoded)
        z, mu, logvar = self._cvae_encode(actions.to(vision_tokens.dtype), proprio_tokens)
        # Decode
        action_pred = self._decode(vision_tokens, proprio_tokens, z)
        # Losses
        rec_loss = F.l1_loss(action_pred.float(), actions.float())
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        total = rec_loss + self.kl_weight * kl_loss
        return total, {"loss_total": total.detach(), "loss_l1": rec_loss.detach(), "loss_kl": kl_loss.detach()}

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
        model_dtype = self.proprio_proj.weight.dtype
        B = pixel_values.shape[0]
        if proprioception is None:
            proprio = torch.zeros(B, self.num_history, self.action_dim,
                                  device=pixel_values.device, dtype=model_dtype)
        else:
            proprio = proprioception.to(model_dtype)
            B = proprio.shape[0]
        vision_tokens = self._encode_vision(pixel_values.to(model_dtype), B)
        proprio_tokens = self._encode_proprio(proprio)
        z = torch.zeros(B, self.latent_dim, device=vision_tokens.device, dtype=model_dtype)
        action_pred = self._decode(vision_tokens, proprio_tokens, z)
        if return_spatial:
            return action_pred, None
        return action_pred
