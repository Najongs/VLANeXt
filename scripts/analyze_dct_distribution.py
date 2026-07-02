#!/usr/bin/env python3
"""DCT k-bin distribution analysis on a single training-data episode.

For a given HDF5 episode file, builds all sliding-window chunks of length T,
normalizes actions to [-1,1] (matching SimActAlign), computes DCT-II per chunk,
and plots:
  (a) Box plot of |DCT[k]| per (k, action_dim) — bin-energy distribution
  (b) Time series of |DCT[k]| per chunk-start step — *when* each freq bin is excited
  (c) Heatmap of mean |DCT[k]| per (k, dim) — overall energy concentration

Usage:
  python scripts/analyze_dct_distribution.py \
      --h5 /home/najo/NAS/VLANeXt/dataset/approach/approach_00/collected_data_merged/w0_episode_20260507_174548.h5 \
      --T 8 --out-dir figures/dct_distribution
"""
import argparse
import os
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Same constants as src/datasets/sim_act_align.py
ACTION_MIN = np.array([-0.37, -0.37, -0.37, -0.0025, -0.0007, -0.007], dtype=np.float32)
ACTION_MAX = np.array([0.37, 0.37, 0.37, 0.0025, 0.0007, 0.007], dtype=np.float32)
ACTION_LABELS = ['ΔX (mm)', 'ΔY (mm)', 'ΔZ (mm)', 'Δrx (deg)', 'Δry (deg)', 'Δrz (deg)']


def dct_matrix(T):
    n = np.arange(T).astype(np.float32)
    k = np.arange(T).astype(np.float32)
    M = np.cos((np.pi / T) * (n + 0.5)[None, :] * k[:, None])
    M[0, :] *= 1.0 / np.sqrt(T)
    M[1:, :] *= np.sqrt(2.0 / T)
    return M  # [T, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True, help='single .h5 episode file')
    ap.add_argument('--T', type=int, default=8, help='chunk length (DCT size)')
    ap.add_argument('--out-dir', default='figures/dct_distribution')
    ap.add_argument('--freq-split', type=float, default=0.125,
                    help='Low/high freq boundary (matches train config)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.h5}')
    with h5py.File(args.h5, 'r') as f:
        action_raw = f['action'][:]            # (N, 7) — last dim usually gripper
        try:
            phase = f['phase'][:]              # (N,)
        except Exception:
            phase = np.zeros(action_raw.shape[0], dtype=np.int32)

    N_total = action_raw.shape[0]
    D = ACTION_MIN.shape[0]                    # 6 (xyz + rot)
    actions = action_raw[:, :D]                # drop gripper

    # Normalize to [-1, 1] same as SimActAlign
    norm = 2 * (actions - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1
    norm = np.clip(norm, -1, 1)
    print(f'Episode steps: {N_total}, action_dim: {D}, clipping: '
          f'{(np.abs(norm) >= 1.0).any(axis=1).sum()} steps saturate')

    T = args.T
    N_chunks = N_total - T + 1
    if N_chunks <= 0:
        raise RuntimeError(f'Episode too short ({N_total}) for chunk length T={T}')

    chunks = np.stack([norm[i:i + T] for i in range(N_chunks)])   # [N_chunks, T, D]
    print(f'Number of sliding chunks (T={T}): {N_chunks}')

    # DCT per chunk: dct[k, d] = sum_t M[k, t] x[t, d]
    M = dct_matrix(T)
    chunks_perm = chunks.transpose(0, 2, 1)                       # [N_chunks, D, T]
    dct_perm = np.einsum('kt,bdt->bdk', M, chunks_perm)           # [N_chunks, D, T]
    dct = dct_perm.transpose(0, 2, 1)                             # [N_chunks, T, D]
    dct_abs = np.abs(dct)

    split = max(1, int(T * args.freq_split))

    # ====================================================================
    # (a) Box plot of |DCT[k]| per (k, action_dim)
    # ====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for d in range(D):
        ax = axes[d // 3, d % 3]
        # data: list of arrays, one per k
        data_per_k = [dct_abs[:, k, d] for k in range(T)]
        bp = ax.boxplot(data_per_k, positions=range(T), widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, k in zip(bp['boxes'], range(T)):
            patch.set_facecolor('#94c4f5' if k < split else '#f5a994')
            patch.set_alpha(0.85)
        # Overlay mean as red dots
        means = [arr.mean() for arr in data_per_k]
        ax.plot(range(T), means, 'r-o', markersize=5, linewidth=1.5, label='mean')

        ax.axvline(split - 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('freq bin k (0=DC)')
        ax.set_ylabel('|DCT coef|')
        ax.set_title(ACTION_LABELS[d], fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':', axis='y')
        ax.set_xticks(range(T))
        if d == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'|DCT[k]| distribution per (k, action dim) — {Path(args.h5).name}\n'
                 f'{N_chunks} sliding chunks, T={T}, freq_split={args.freq_split} '
                 f'(blue=low, red=high)',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_a = out_dir / 'dist_box_per_kdim.png'
    fig.savefig(out_a, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_a}')

    # ====================================================================
    # (b) Time series of |DCT[k]| per chunk-start step (which frame excites freq)
    # ====================================================================
    fig, axes = plt.subplots(D, 1, figsize=(13, 2.0 * D), sharex=True)
    if D == 1:
        axes = [axes]
    chunk_starts = np.arange(N_chunks)
    cmap_low = plt.cm.Blues(np.linspace(0.4, 0.9, split))
    cmap_high = plt.cm.Reds(np.linspace(0.4, 0.9, T - split))
    palette = list(cmap_low) + list(cmap_high)

    for d in range(D):
        ax = axes[d]
        for k in range(T):
            ax.plot(chunk_starts, dct_abs[:, k, d], color=palette[k], linewidth=1.2,
                    alpha=0.85 if k < split else 0.7,
                    label=f'k={k} {"(DC)" if k==0 else "(low)" if k<split else "(high)"}'
                          if d == 0 else None)
        ax.set_ylabel(f'{ACTION_LABELS[d]}\n|DCT[k]|')
        ax.grid(alpha=0.3, linestyle=':')
        # Phase shading
        if (phase != phase[0]).any():
            phase_starts = [0]
            for t in range(1, N_total):
                if phase[t] != phase[t - 1]:
                    phase_starts.append(t)
            phase_starts.append(N_total)
            colors = ['#e0f2fe', '#fef3c7', '#fee2e2']
            for i, (s, e) in enumerate(zip(phase_starts[:-1], phase_starts[1:])):
                ax.axvspan(s, min(e, N_chunks), color=colors[i % len(colors)],
                           alpha=0.18, zorder=0)
        if d == D - 1:
            ax.set_xlabel('chunk-start step (within episode)')
    if D >= 1:
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    fig.suptitle(f'|DCT[k]| over time — when each freq bin gets excited\n'
                 f'{Path(args.h5).name} (phase shading)', fontsize=12, fontweight='bold', y=1.005)
    fig.tight_layout()
    out_b = out_dir / 'dist_timeseries.png'
    fig.savefig(out_b, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_b}')

    # ====================================================================
    # (c) Heatmap of mean and median |DCT[k]| per (k, dim)
    # ====================================================================
    mean_mag = dct_abs.mean(axis=0)            # [T, D]
    median_mag = np.median(dct_abs, axis=0)
    p95_mag = np.percentile(dct_abs, 95, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = mean_mag.max()
    titles = [
        'mean |DCT[k]| per (k, dim)',
        'median |DCT[k]| per (k, dim)',
        '95th percentile |DCT[k]| per (k, dim)',
    ]
    for ax, mat, title in zip(axes, [mean_mag, median_mag, p95_mag], titles):
        im = ax.imshow(mat, aspect='auto', cmap='viridis',
                       vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xlabel('action dim')
        ax.set_ylabel('freq bin k (0=DC, high index = high freq)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(D))
        ax.set_xticklabels([lbl.split()[0] for lbl in ACTION_LABELS], rotation=30, ha='right')
        ax.set_yticks(range(T))
        ax.axhline(split - 0.5, color='white', linestyle='--', alpha=0.85, linewidth=1.2)
        # annotate values
        for ki in range(T):
            for di in range(D):
                v = mat[ki, di]
                ax.text(di, ki, f'{v:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if v < vmax * 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'DCT energy summary — {Path(args.h5).name} ({N_chunks} chunks)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    out_c = out_dir / 'dist_heatmap_summary.png'
    fig.savefig(out_c, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_c}')

    # ====================================================================
    # Print summary stats
    # ====================================================================
    print('\n=== DCT bin energy summary ===')
    print(f'{"k":>3} {"region":>5}  ' + '  '.join(f'{lbl.split()[0]:>8}' for lbl in ACTION_LABELS))
    for k in range(T):
        region = 'low ' if k < split else 'high'
        vals = '  '.join(f'{mean_mag[k, d]:>8.4f}' for d in range(D))
        print(f'{k:>3} {region:>5}  {vals}')

    # Low vs high energy ratio per dim
    print('\n=== low-vs-high energy ratio per dim ===')
    low_energy = mean_mag[:split].sum(axis=0)
    high_energy = mean_mag[split:].sum(axis=0)
    total = mean_mag.sum(axis=0)
    print(f'{"dim":>6}  {"low%":>6}  {"high%":>6}  {"total":>8}')
    for d in range(D):
        low_pct = 100 * low_energy[d] / max(total[d], 1e-9)
        high_pct = 100 * high_energy[d] / max(total[d], 1e-9)
        print(f'{ACTION_LABELS[d].split()[0]:>6}  {low_pct:>6.1f}  {high_pct:>6.1f}  {total[d]:>8.4f}')

    print('Done.')


if __name__ == '__main__':
    main()
