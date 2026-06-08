#!/usr/bin/env python3
"""Visualize DCT of GT vs Pred action chunks.

For a trained VLANeXt checkpoint, samples training-data batches, runs predict_action
to get predicted action chunks, computes DCT-II (orthonormal) on both GT and Pred
along the time axis (T=8), then plots:
  - Top row : time-domain (GT vs Pred) per action dim
  - Bottom row: |DCT coef| per frequency bin per action dim
  - Heatmap summary : batch-mean |DCT| of GT, Pred, and their diff

Usage:
  python scripts/visualize_dct.py \
      --config config/sim_train_align_qwen_withReal_v11_submm_tight_config.yaml \
      --checkpoint /data/public/NAS/VLANeXt/checkpoints/VLANeXt_Qwen35_withReal/reach_recover_v11_submm_tight/checkpoint_1500.pt \
      --n-samples 4
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, '/data/public/NAS/VLANeXt')
from src.models.VLANeXt import VLANeXt
from src.datasets.sim_act_align import SimActAlign
from scripts.train import DataCollatorForVLANeXt


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def dct_matrix(T, device):
    """DCT-II orthonormal matrix. Same formula as VLANeXt._compute_dct_loss."""
    n = torch.arange(T, device=device).float()
    k = torch.arange(T, device=device).float()
    M = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
    M[0, :] *= 1.0 / np.sqrt(T)
    M[1:, :] *= np.sqrt(2.0 / T)
    return M  # [T, T]


def dct_transform(x, M):
    """x: [B, T, D] → DCT over time per channel → [B, T, D]."""
    perm = x.permute(0, 2, 1)        # [B, D, T]
    out = torch.matmul(perm, M.t())   # [B, D, T]
    return out.permute(0, 2, 1)       # [B, T, D]


def build_model(cfg, device):
    m = cfg['model']
    return VLANeXt(
        lmm_path=m['lmm_path'],
        vision_encoder_path=m.get('vision_encoder_path', 'google/siglip2-base-patch16-256'),
        action_dim=m['action_dim'],
        num_actions=cfg['data']['future_len'],
        num_queries=m['num_queries'],
        num_history=cfg['data']['history_len'],
        loss_type=m.get('loss_type', 'diffusion'),
        future_image_loss_weight=float(m.get('future_image_loss_weight', 0.0)),
        num_train_timesteps=m.get('num_train_timesteps', 1000),
        num_inference_timesteps=m.get('num_inference_timesteps', 10),
        scheduler_type=m['scheduler_type'],
        condition_type=m.get('condition_type', 'soft'),
        policy_hidden_size=m['policy_hidden_size'],
        policy_depth=m['policy_depth'],
        policy_num_heads=m['policy_num_heads'],
        policy_mlp_ratio=m['policy_mlp_ratio'],
        use_proprio_input_vlm=m.get('use_proprio_input_vlm', True),
        use_action_input_policy=m.get('use_action_input_policy', False),
        use_transformer_proprio_projector=m.get('use_transformer_proprio_projector', False),
        projector_depth=m['projector_depth'],
        projector_num_heads=m['projector_num_heads'],
        use_transformer_connector=m['use_transformer_connector'],
        connector_depth=m['connector_depth'],
        connector_num_heads=m['connector_num_heads'],
        backbone_mode=m.get('backbone_mode', 'frozen'),
        n_unfreeze_layers=m.get('n_unfreeze_layers', 4),
        gradient_checkpointing=False,
        num_bins=m.get('num_bins', 256),
        generator_hidden_size=m.get('generator_hidden_size', 768),
        generator_depth=m.get('generator_depth', 12),
        generator_num_heads=m.get('generator_num_heads', 12),
        generator_mlp_ratio=m.get('generator_mlp_ratio', 4.0),
        action_vqvae=m.get('action_vqvae', None),
        dct_loss_weight=m.get('dct_loss_weight', 0.1),
        dct_low_freq_weight=m.get('dct_low_freq_weight', 1.0),
        dct_high_freq_weight=m.get('dct_high_freq_weight', 1.0),
        dct_freq_split=m.get('dct_freq_split', 0.125),
        dct_similarity_type=m.get('dct_similarity_type', 'mae'),
        aux_distance_loss=m.get('aux_distance_loss', None),
        aux_lateral_loss=m.get('aux_lateral_loss', None),
        aux_hold_loss=m.get('aux_hold_loss', None),
        direction_decoupled_loss=m.get('direction_decoupled_loss', None),
        proprio_dim=m.get('proprio_dim', None),
        input_image_size=m.get('input_image_size', None),
        attn_implementation=m.get('attn_implementation', 'flash_attention_2'),
    ).to(device, dtype=torch.bfloat16)


def build_dataset(cfg):
    d = cfg['data']
    return SimActAlign(
        data_dir=d['data_root'],
        dataset_name='sim_align',
        history_len=d['history_len'],
        future_len=d['future_len'],
        full_sequence=bool(d.get('full_sequence', True)),
        input_modality=d['input_modality'],
        view_mode=d['view_mode'],
        load_future_image=False,
        future_image_mode='horizon',
        buffer_size=100,  # small — just want a few samples
        cam_exterior=d.get('cam_exterior', 'tool_camera'),
        cam_wrist=d.get('cam_wrist', ''),
        cam_top=d.get('cam_top', ''),
        skip_history_padding=bool(d.get('skip_history_padding', True)),
        use_sensor=cfg['model'].get('use_sensor', False),
        sensor_encoding=cfg['model'].get('sensor_encoding', 'binary'),
        sensor_clip_mm=cfg['model'].get('sensor_clip_mm', 30.0),
        near_goal_oversample=d.get('near_goal_oversample', 1.0),
        near_goal_threshold_mm=d.get('near_goal_threshold_mm', 15.0),
        local_crop_enabled=d.get('local_crop_enabled', False),
        local_crop_size=d.get('local_crop_size', 320),
        use_keypoint_proprio=cfg['model'].get('use_keypoint_proprio', False),
        crop_around_trocar=d.get('crop_around_trocar', False),
        crop_window_px=d.get('crop_window_px', 256),
        crop_target_size_px=d.get('crop_target_size_px', 512),
        crop_mode=d.get('crop_mode', 'center'),
    )


ACTION_LABELS = ['ΔX (mm)', 'ΔY (mm)', 'ΔZ (mm)', 'Δrx (deg)', 'Δry (deg)', 'Δrz (deg)']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='train config yaml (defines model + data)')
    ap.add_argument('--checkpoint', required=True, help='checkpoint .pt path')
    ap.add_argument('--out-dir', default='figures/dct_examples')
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--motion-std-threshold', type=float, default=0.0,
                    help='Min std(GT pos chunk over T) — keep samples with motion > this')
    ap.add_argument('--max-candidates', type=int, default=64,
                    help='Max candidates to scan when filtering by motion')
    ap.add_argument('--motion-metric', choices=['std', 'accel'], default='std',
                    help='std: motion magnitude (filters out static). '
                         'accel: |1st diff| std — filters for transition phases '
                         '(motion onset / convergence). Mid-episode steady velocity → low accel std.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = load_config(args.config)

    print(f'Building model from config: {args.config}')
    model = build_model(cfg, device)

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt)
    if list(sd.keys())[0].startswith('module.'):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    miss, unx = model.load_state_dict(sd, strict=False)
    print(f'  Missing: {len(miss)}, Unexpected: {len(unx)}')
    model.eval()
    del ckpt, sd

    ds = build_dataset(cfg)
    collator = DataCollatorForVLANeXt(
        processor=model.processor,
        use_proprio_input_vlm=cfg['model'].get('use_proprio_input_vlm', True),
        use_action_input_policy=cfg['model'].get('use_action_input_policy', False),
        input_modality=cfg['data']['input_modality'],
        view_mode=cfg['data']['view_mode'],
        fps=15.0,
        augmentation={'enabled': False},
        load_future_image=False,
    )
    # Single bigger batch (Qwen vision uses flat pixel_values so can't splice across batches).
    # Sample probe_batch at once, run inference on the full batch, then select top-N by motion.
    probe_batch = min(max(args.n_samples * 4, args.n_samples), args.max_candidates)
    loader = DataLoader(ds, batch_size=probe_batch, num_workers=0, collate_fn=collator)

    T = cfg['data']['future_len']
    M = dct_matrix(T, device)
    split = max(1, int(T * cfg['model'].get('dct_freq_split', 0.125)))

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Sampling 1 probe batch={probe_batch}, motion_threshold={args.motion_std_threshold}')
    batch = next(iter(loader))
    (inputs, gt_actions, proprio, hist_actions, _fut,
     _sp, _aw, _nt, _te, _td, _src) = batch

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    for k in ('pixel_values', 'pixel_values_videos'):
        if k in inputs:
            inputs[k] = inputs[k].to(torch.bfloat16)
    gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
    if proprio is not None:
        proprio = proprio.to(device, dtype=torch.bfloat16)
    if hist_actions is not None:
        hist_actions = hist_actions.to(device, dtype=torch.bfloat16)

    print('Running predict_action on full probe batch ...')
    with torch.no_grad():
        pred = model.predict_action(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            proprioception=proprio,
            history_actions=hist_actions,
            pixel_values=inputs.get('pixel_values'),
            pixel_values_videos=inputs.get('pixel_values_videos'),
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'),
        )

    gt_full = gt_actions.float()
    pr_full = pred.float()

    # Select top-N by GT translational motion
    pos = gt_full[..., :3]
    if args.motion_metric == 'std':
        # Total motion magnitude — picks active samples but middle steady-velocity also passes
        motions = pos.std(dim=1).max(dim=-1).values  # [B_probe]
    else:
        # Acceleration (1st diff) std — picks TRANSITION samples (onset / convergence).
        # Steady velocity → diff const → diff.std ≈ 0. Onset/convergence → diff changes → high.
        accel = pos[:, 1:] - pos[:, :-1]            # [B, T-1, 3]
        motions = accel.std(dim=1).max(dim=-1).values

    if args.motion_std_threshold > 0:
        passing = (motions > args.motion_std_threshold).nonzero(as_tuple=True)[0]
        if len(passing) == 0:
            raise RuntimeError(
                f'No samples in probe batch passed motion threshold {args.motion_std_threshold}. '
                f'Max motion in batch was {motions.max().item():.3f}. '
                f'Try lowering --motion-std-threshold or increasing --max-candidates.')
        sub = passing[torch.argsort(motions[passing], descending=True)]
    else:
        sub = torch.argsort(motions, descending=True)

    sel = sub[:args.n_samples].cpu()
    print(f'Selected {len(sel)}/{args.n_samples} samples (from probe of {probe_batch})')
    print(f'Motion magnitudes (selected): {motions[sel.to(device)].cpu().numpy()}')

    # Slice GT/pred by sample index — these ARE per-sample tensors (unlike pixel_values)
    gt_f = gt_full[sel.to(device)]
    pr_f = pr_full[sel.to(device)]
    gt_dct = dct_transform(gt_f, M)
    pr_dct = dct_transform(pr_f, M)

    selected_motions = motions[sel.to(device)]

    B, _, D = gt_f.shape
    print(f'Action chunk B={B} T={T} D={D}')
    print(f'GT range: [{gt_f.min():.3f}, {gt_f.max():.3f}]')
    print(f'Pred range: [{pr_f.min():.3f}, {pr_f.max():.3f}]')

    # ----- per-sample figure -----
    for i in range(B):
        fig, axes = plt.subplots(2, D, figsize=(3.6 * D, 7.2))
        for d in range(D):
            label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f'dim {d}'

            ax_t = axes[0, d]
            x = np.arange(T)
            ax_t.plot(x, gt_f[i, :, d].cpu().numpy(), '-o', color='#0066cc', label='GT',
                      linewidth=2, markersize=6)
            ax_t.plot(x, pr_f[i, :, d].cpu().numpy(), '-s', color='#cc3300', label='Pred',
                      linewidth=2, markersize=6, alpha=0.85)
            ax_t.set_title(label, fontsize=11, fontweight='bold')
            ax_t.set_xlabel('t (chunk step)')
            ax_t.grid(alpha=0.3, linestyle=':')
            if d == 0:
                ax_t.set_ylabel('action (normalized [-1,1])')
                ax_t.legend(fontsize=9, loc='best')

            ax_f = axes[1, d]
            gt_mag = gt_dct[i, :, d].abs().cpu().numpy()
            pr_mag = pr_dct[i, :, d].abs().cpu().numpy()
            xb = np.arange(T)
            w = 0.4
            ax_f.bar(xb - w / 2, gt_mag, w, color='#0066cc', label='GT', alpha=0.85)
            ax_f.bar(xb + w / 2, pr_mag, w, color='#cc3300', label='Pred', alpha=0.85)
            ax_f.set_xlabel('freq bin k')
            ax_f.grid(alpha=0.3, linestyle=':', axis='y')
            ax_f.set_xticks(xb)
            ax_f.axvline(split - 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax_f.text(split - 0.5, ax_f.get_ylim()[1] * 0.92, ' high freq →',
                      fontsize=8, color='gray', ha='left')
            if d == 0:
                ax_f.set_ylabel('|DCT coef|')

            mae = float(np.mean(np.abs(gt_mag - pr_mag)))
            ax_f.text(0.97, 0.95, f'|Δ|={mae:.3f}', transform=ax_f.transAxes,
                      ha='right', va='top', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='gray', alpha=0.9))

        m_i = selected_motions[i].item() if selected_motions is not None else 0.0
        m_label = 'GT pos-std' if args.motion_metric == 'std' else 'GT accel-std (transition)'
        fig.suptitle(
            f'GT vs Pred action chunk — sample {i + 1}/{B}  '
            f'({m_label}={m_i:.3f}, T={T}, freq_split={cfg["model"].get("dct_freq_split", 0.125)})',
            fontsize=12, fontweight='bold', y=1.00,
        )
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'dct_sample_{i + 1}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {out_path}')

    # ----- heatmap summary across batch -----
    gt_avg = gt_dct.abs().mean(dim=0).cpu().numpy()  # [T, D]
    pr_avg = pr_dct.abs().mean(dim=0).cpu().numpy()
    diff = (gt_dct - pr_dct).abs().mean(dim=0).cpu().numpy()

    vmax = max(gt_avg.max(), pr_avg.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['GT |DCT| (batch mean)', 'Pred |DCT| (batch mean)', '|GT − Pred|']
    for ax, mat, title, vm in zip(axes, [gt_avg, pr_avg, diff], titles,
                                  [vmax, vmax, diff.max()]):
        im = ax.imshow(mat, aspect='auto', cmap='viridis',
                       vmin=0, vmax=vm, interpolation='nearest')
        ax.set_xlabel('action dim')
        ax.set_ylabel('freq bin k (0=DC, high index = high freq)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(D))
        ax.set_xticklabels(
            [ACTION_LABELS[i].split()[0] if i < len(ACTION_LABELS) else f'd{i}'
             for i in range(D)], rotation=30, ha='right')
        ax.set_yticks(range(T))
        ax.axhline(split - 0.5, color='white', linestyle='--', alpha=0.8, linewidth=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'DCT magnitude heatmap (batch mean of {B} samples)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, 'dct_heatmap_summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
