# Project Status — 2026-06-08

## Current conclusion

The final paper/SOTA fine-alignment model is:

- Model: Qwen3.5-2B-VL + ConnectorTransformer + DiT diffusion action head
- Config: `config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml`
- Expected checkpoint: `checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_flat_1500.pt`
- Honest eval protocol: `--no-early-term`, 27-cell grid, full 250 steps, `--num-steps-execute 2`, diffusion steps 10
- Reported result: R5 100, R2 96.3, R1 55.6, HoldSR 100, Settled 1.82 +/- 0.07 mm

## Local server state

This workspace has been normalized to:

`/home/najo/NAS/VLANeXt`

The previous server path `/data/public/NAS/VLANeXt` was replaced across configs/scripts.

The local `checkpoints/` directory currently does not contain the final Qwen v11 checkpoint tree above. It mainly contains earlier DeepSpeed-style checkpoints:

- `checkpoints/VLANeXt_Qwen3.5_fine/...`
- `checkpoints/VLANeXt_SigLIP2_NEARGOAL/...`
- `checkpoints/VLANeXt_droid.pt`

The local `dataset/` directory also lacks the NEARGOAL datasets needed to reproduce v11 training:

- `NEARGOAL_eval_match_v2`
- `NEARGOAL_angle_only_v2`
- `NEARGOAL_yneg_hold_v1`
- `NEARGOAL_perfect_strict_v1`
- `NEARGOAL_perfect_hold_v1`
- `NEARGOAL_yneg_v1`
- `NEARGOAL_ypos_v1`
- `NEARGOAL_submm_hold_v1`

Those checkpoint/data artifacts need to be synced from the machine where the final model was found before rerunning v11 eval/training locally.

## Entry points

Training:

```bash
GPUS=0 bash Run_Train.sh
```

Default config is now `config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml`.

Evaluation:

```bash
GPUS=0,1 bash Run_Eval_Parallel.sh align \
  checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_flat_1500.pt \
  --train-config config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml \
  --no-early-term --max-steps 250 --eval-seed 2026 \
  --perturb-mode grid --xy-steps 3 --z-steps 2 --angle-steps 3 --repeats 1
```

Real replay collection no longer uploads artifacts by default. To sync after collection, explicitly set:

```bash
SYNC_REAL_ARTIFACTS=1 REMOTE_DATA_TARGET=user@host:/path/to/dataset bash Run_Collect_Real_Align_Replay.sh ...
```

## Local 50k configs

Two local 50k baseline configs are present but not yet tracked:

- `config/output_dir_b100_baseline_model_50000step_qwen.yaml`
- `config/output_dir_b100_baseline_model_50000step_gemma.yaml`

They were renamed internally to reflect their actual backbone and 50k schedule.
