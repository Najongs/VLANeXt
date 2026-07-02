# VLANeXt SigLIP2 NEARGOAL Eval Results - 2026-06-10

## Setup

- Eval mode: `align`
- GPUs: `0,1,2,3,4` with 5 shards
- Episodes per model: 27
- Execution: `--max-steps 250 --num-steps-execute 2 --num-inference-timesteps 10 --no-early-term`
- Grid: `x=[-10, 0, 10]mm`, `y=[-25, 0, 25]mm`, `z=[0]mm`, `angle=[-5, 0, 5]deg`
- Output prefix: `align_eval_stepflat_exec2_diff10`

The first attempted run used the wrong default grid and was moved aside with `wrong_grid` suffixes. The table below uses only the corrected grid outputs.

## Summary

| Model | n | Success | Reach@5 | Reach@2 | Reach@1 | MinLat med | MinLat mean | Hold 2.5mm/20 | SettledLat med | FinalLat mean | FinalAngle mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `siglip_sim` | 27 | 0.0% | 11.1% | 11.1% | 3.7% | 18.71mm | 16.30mm | 11.1% | 18.89mm | 16.60mm | 65.17deg |
| `siglip_withReal` | 27 | 0.0% | 11.1% | 7.4% | 7.4% | 15.74mm | 13.20mm | 7.4% | 15.80mm | 13.75mm | 65.63deg |
| `qwen_sim` | 27 | 0.0% | 11.1% | 0.0% | 0.0% | 24.43mm | 21.77mm | 0.0% | 24.93mm | 22.01mm | 67.97deg |
| `qwen_withReal` | 27 | 0.0% | 11.1% | 3.7% | 3.7% | 25.73mm | 22.44mm | 7.4% | 25.92mm | 22.44mm | 67.81deg |
| `gemma_withReal` | 27 | 0.0% | 3.7% | 0.0% | 0.0% | 17.92mm | 16.00mm | 0.0% | 18.62mm | 16.60mm | 65.75deg |

`Reach@K`, `MinLat`, `Hold`, and `SettledLat` are computed from `traj_ep*.npz` lateral trajectories. `Success`, `FinalLat`, and `FinalAngle` are from merged `metrics_summary.csv`.

## Interpretation

- None of the five checkpoints passed the task-level success criterion on this 27-cell corrected grid.
- `siglip_withReal` is the best by median/mean lateral closeness and final lateral error.
- `siglip_sim` has the best Reach@2 and Hold 2.5mm/20-step rate, but its median/final lateral is worse than `siglip_withReal`.
- The Qwen variants are clearly worse on this eval than the SigLIP variants by lateral metrics.
- `gemma_withReal` is close to `siglip_sim` on final lateral, but reaches tight thresholds less often.

## Merged Outputs

- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/v2_dual_finetune/align_eval_stepflat_exec2_diff10/`
- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/v2_dual_finetune_withReal/align_eval_stepflat_exec2_diff10/`
- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/v2_dual_finetune_qwen/align_eval_stepflat_exec2_diff10/`
- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/v2_dual_finetune_qwen_withReal/align_eval_stepflat_exec2_diff10/`
- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/v2_dual_finetune_gemma_withReal/align_eval_stepflat_exec2_diff10/`

## Script Fix

`Run_Eval_Parallel.sh` now forwards `--num-steps-execute` and `--num-inference-timesteps` into `scripts/merge_eval_shards.py` as `--exec-steps` and `--diff-steps`. This fixes the previous `exec1` lookup when eval outputs were actually created under `exec2`.
