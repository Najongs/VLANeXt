# Champion Model — Performance Summary (2026-05-19)

**ckpt**: `checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_aux_strong/checkpoint_10000.pt` 🏆
**Train config**: `config/sim_train_align_siglip2_b24_ft10mm_aux_strong_config.yaml`
**Pipeline**: vision-only VLA (no LLM, no KP servo, no sensor handoff). Single tool_camera, 1D sensor for safety brake only.

## Architecture (short)

- SigLIP2-so400m-patch16-512 (frozen, bf16) → 1024 tokens × 1152 dim
- 3-layer MLP projector (LLaVA-style, Linear-LN-SiLU ×3)
- Proprio Linear(6→1152) ×8 history + 32 learnable meta_queries
- Action diffusion head (10-step flow-match, action chunk 8)
- Loss = main_flow_match + 0.1·DCT + 0.5·aux_distance

## Sim SR (paper main grid, 27 cells, eval@512)

Grid: x∈{−10, 0, +10}mm, y∈{−25, 0, +25}mm, z=0, phantom angle ∈ {…}.
Success: dist ≤ 5mm AND angle ≤ 10° AND sensor > 20mm AND hold 20 steps.

| angle range | n | SR | mean min_dist | true align err (\|min_dist−10\|) | mean \|final_angle\| |
|---|---:|---:|---:|---:|---:|
| **±5°** | 27 | **85.19%** (23/27) | 9.35mm | 1.94mm | 5.36° |
| ±10° | 27 | 77.78% (21/27) | — | — | — |

**Per-cell weakness (ang±5)**: 4 failures concentrated at y=−25 (cells (0,−25)×3 + (+10,−25,−5°)). Other 8 (x,y) cells: 100%.

Heatmap: `vqa_samples/eval_noff_pooled_heatmap.png` (3-panel: SR / mean min_dist / |min_dist−10mm|).

## Comparison (27-cell, ang±10°)

| Rank | ckpt | SR |
|---|---|---:|
| 🥇 | aux_strong/10000 | **77.78%** |
| 2 | HARD_cotrain_lr_low/3000 | 70.37% |
| 3 | HOLD_focus_v2/3000 | 62.96% |

## Inference

- Pure inference: 226.6 ms/step (RTX 3090, bf16, 1043M params, 4.4 Hz)
- Mesa render overhead: ~85 ms/step on top
- Action chunk 8 × execute 1 (theoretical ~35 Hz with full chunk exec; not used)

## Real robot

⚠️ Pending. Sim-to-real dry-run after paper main number frozen.

See `EXPERIMENTS_fine_align.md` for full experiment log + dead-end list.
