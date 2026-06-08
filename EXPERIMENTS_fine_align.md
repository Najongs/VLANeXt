# Fine Alignment Experiments — Master Cheatsheet

**Last revised**: 2026-05-25 (architecture clarification + ablation reframing)
**Companion**: `ablation.md` (axis-by-axis ablation analysis), `attic/` (deprecated logs)

---

## 0. TL;DR

> Surgical needle-trocar **sub-mm sustained alignment** is gated by **architecture**, not by data or loss engineering. Vision-only encoders (any family) hit a precision-stability trade-off; **VLM (Qwen3.5-2B-VL) + DiT diffusion head** breaks the trade-off and achieves R2 96.3%, HoldSR 100%, Settled 1.82 ± 0.07 mm — **3.3× sustained-alignment improvement** over the best vision-only baseline.

### 🏆 Paper SOTA (one row)
```
ckpt:   checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_flat_1500.pt
config: config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml
eval:   align_eval_step1500_exec2_diff10_noET (--no-early-term, 27-cell, full 250-step)
honest: R5 100  R2 96.3  R1 55.6  mLat 0.79  HoldSR 100  Settled 1.82±0.07  Sf_p99 3.72  max 3.74
```

---

## 1. Model architecture (corrected 2026-05-25)

### Ours: VLANeXt
```
Image (256×256)
    ↓
Qwen3.5-2B-VL (HuggingFace AutoModelForImageTextToText)
    ├─ internal vision tower (Qwen's own — NOT external SigLIP2)
    ├─ vision projector
    └─ 24-layer language decoder (linear+full attn hybrid, +LoRA r=16 on q/k/v/o)
    ↓
ConnectorTransformer (32 learnable queries, depth=2, heads=4)
    ↓
Action Head — DiT diffusion policy
    (depth=24, hidden=1152, heads=16, flow-match, 10 inference steps)
    ↓
Action chunk (future_len=8, action_dim=6)  →  exec=2 at deployment
```

**Trainable**: Qwen LoRA + connector + action head (~100M of 2.85B total).

### Baseline comparison
| Method | Architecture | Notes |
|---|---|---|
| ACT | ResNet18 + CVAE + Transformer | scratch (62M) |
| Diffusion Policy | ResNet18 + Conditional U-Net 1D | scratch (89M) |
| SigLIP2 + DiT | SigLIP2-so400m vision-only + DiT | LM 없음, `lmm_path: vision_only` |
| 🏆 **Ours** | **Qwen3.5-2B-VL + DiT** | VL backbone (internal vision tower), `lmm_path: Qwen/Qwen3.5-2B` |

→ **vision-only baseline은 SigLIP2를 명시적으로 load** (vision_only path, no LM)
→ **Ours는 Qwen3.5-2B-VL 단독** (AutoModelForImageTextToText, Qwen's own vision tower)
→ **별도 SigLIP2를 Qwen에 붙인 게 아님** — Qwen3.5-2B-VL 자체가 multimodal

---

## 2. Evaluation Protocol (★ honest)

### Critical fix (2026-05-24)
**모든 fine-align eval에 `--no-early-term` flag 필수.** [[feedback_no_early_term_mandatory]]
- 기존 sim_eval은 `check_success()` (3D dist<5mm + 20 hold) 만족 시 episode break
- 평균 ~120/250 step에서 끊김 → 모델별 다른 시점 측정 = artifact
- `_noET` suffix = honest eval, 그 외는 신뢰 X

### 27-cell evaluation grid
- XY ±10 mm × Y ±25 mm × angle ±5° (retreat=2)
- 250-step trajectory (full)

### Metric suite (4 axis)
| Axis | Metric | Definition | Unit |
|---|---|---|---|
| **Reach** | R5 / R2 / R1 | episode min lat < 5/2/1 mm | % |
| **Precision** | min_lat | episode-wise min lateral | mm |
| **Hold** | HoldSR | 20-step contig lat<5mm | % |
| | Max30<2.5 | last-30 max < 2.5mm | % |
| **Stability** | Settled ± std | last-30 mean ± std | mm |
| **Safety** | Sf_p95 / p99 / max | settled distribution percentiles | mm |

★ 의료적 직관: **Sf_max + violation count (settled>5mm 개수)** 가 가장 명확. n=27이라 p99 ≈ max.

---

## 3. Main Comparison Table (paper Table 1)

★ 각 baseline은 best ckpt 선정 (각 stage-by-stage 최적 선택, cherry-pick 아님)
★ 모든 수치 honest eval, 250-step trajectory, exec 컬럼에 명시

| Method | Backbone | Params | Eval ckpt / exec | R5 | R2 | R1 | min_lat | HoldSR | Settled | Sf_p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT | ResNet18 + CVAE | 62M | ck5000 / e1 | 100 | 59.3 | 22.2 | 1.44 | 70.4 | 2.91 ± 0.30 | 7.06 | 13.43 |
| Diffusion Policy | ResNet18 + CondU1D | 89M | ck15000 / e1 | 100 | 48.1 | 18.5 | 2.29 | 48.1 | 3.63 ± 0.40 | 6.23 | 4.69 |
| SigLIP2 + DiT (vision-only) | SigLIP2-so400m | 1.4B | v5_combo ck2000 / e4 | 100 | 88.9 | 59.3 | 0.87 | 88.9 | 5.95 ± 0.28 ⚠ | 11.77 | 12.19 |
| 🏆 **Ours (VLANeXt)** | **Qwen3.5-2B-VL** | 2.85B | v11 ck1500 / e2 | **100** | **96.3** | 55.6 | **0.79** | **100** | **1.82 ± 0.07** | **3.72** | **3.74** |

### Key takeaways
- Ours wins **all axes except R1** (SigLIP2 +3.7 pp peak instantaneous). 단 R1은 "한 번 닿음"이고, Settled 1.82 vs 5.95 = **sustained alignment 3.3× 개선** 이 진짜 차별성.
- **DP는 좁아 보이는 Sf_p99 6.23이지만 mode collapse + uniform failure** — reach 48%로 trocar 못 닿음. "안정한 실패" trap.
- **Ours는 single training, single ckpt, single exec=2** — deployment complexity 최소.

---

## 4. Champion Recipe (v11 SOTA, end-to-end)

### Stage 1 — Qwen base 20k (fresh)
```yaml
lmm_path: "Qwen/Qwen3.5-2B"   # AutoModelForImageTextToText (VL)
backbone_mode: "frozen"        # LoRA r=16 on q/k/v/o
max_steps: 20000
learning_rate: 1.0e-5
data: approach + 10mm_align + NEARGOAL wide  # ~13K episodes
```
→ ckpt: `VLANeXt_Qwen35_NEARGOAL/(stage1)/checkpoint_20000.pt`

### Stage 2 — reach_recover v2 (cascade 1)
```yaml
pretrained_checkpoint: stage1_ck20000
+ NEARGOAL_yneg_v1 + NEARGOAL_ypos_v1   # y-balance 추가
max_steps: 1500
learning_rate: 1.0e-6
aux_distance: weight 0.5, near_goal_boost 10x at 2mm
aux_lateral:  weight 0.5, near_goal_boost 10x at 1mm
aux_hold:     pos 0.3 / rot 0.5, threshold_mm 2.5, soft_scale 1.0
```
→ ckpt: `reach_recover_v2_aggressive/checkpoint_flat_1500.pt`

### Stage 3 — submm_tight v11 (cascade 2, ★ paper SOTA)
```yaml
pretrained_checkpoint: v2_ck1500
+ NEARGOAL_submm_hold_v1 (1800ep, 1.5mm perturb + 250-step hold) × 2 oversample
max_steps: 1500
learning_rate: 3.0e-7
aux_hold: threshold_mm 1.5 ★, soft_scale 0.7   # tightening = 핵심
```
→ ckpt: `reach_recover_v11_submm_tight/checkpoint_flat_1500.pt`

### Deployment
```
--no-early-term  --num-steps-execute 2  --num-inference-timesteps 10
```

---

## 5. What Worked / What Didn't (ablation 종합 — REVISED 2026-05-25 honest)

### ✅ What worked (drivers, by honest controlled delta)
| Tier | Driver | Honest Δ | 비고 |
|---|---|---|---|
| ★★★★★ | **Qwen VL backbone** (vs SigLIP2 vision-only) | Settled 5.95 → 1.82 mm (3.3×), Sf_p99 11.77 → 3.72 (3.2×), y=-25 1/9 → 6/9 | main contribution |
| ★★★★ | **Hold-rich data 2× oversample** (perfect_strict + perfect_hold) | v2→v5: Settled −0.35 mm, R2 +7.4 pp, R1 +18.6 pp | win-win, no trade |
| ★★★★ | **Chain matching** (base + cascade) | enables architecture comparison | methodology |
| ★★ | **Y-balance data** (vision-only context) | y=-25 1/9 → 2/9 (+1 only). Real fix is architecture (Qwen 1/9 → 6/9) | architecture-bottlenecked |
| ★★ | **exec=2 default** (Qwen universal) | all-axis top: R2/HoldSR/Settled simultaneously | free deployment dial |
| ★★ | **--no-early-term eval** ★ | 4 mm Settled artifact 제거 | methodology essential |
| ★ | **aux_hold threshold 1.5 + submm_hold_v1** (v5→v11) | Settled −0.03 (noise), R1 −7.4 pp ⚠ | trade-off; v5 = R1 champion |

### 🚨 Honest negative (paper-defensible)
| Try | Honest Finding |
|---|---|
| **Loss component (dist / +lat / +hold / +full) combination** | controlled rerun: R2/R1/HoldSR identical (81.5/51.9/85.2), Settled within std band → **statistically indistinguishable** |

→ **Loss component choice는 not a critical lever**. flow matching이 sufficient as primary objective.

### ❌ What didn't (negative results — paper appendix)
| Try | Result | Diagnosis |
|---|---|---|
| Encoder unfreeze (SigLIP2 last4) | mean SR 34% ± σ 23pp (4 seeds) | seed lottery, < frozen 48% |
| DCT loss (w=0.1) | paired-diff ±1 episode | noise-level |
| UV-based crop | catastrophic fail (R2=0) | distribution shock |
| Center crop + 2× zoom | R2 ↓ 33pp / Settled ↓ 3mm | Pareto trade-off only |
| KP proprio (uv+dist) | no holdSR lift | role → safety brake only |
| Sensor proprio (use_sensor=true) | lr 발산 / no gain | scale + lr instability |
| yneg25 tight band data | null effect | architecture-bottlenecked |
| direction_decoupled_loss | gnorm 폭주 | harmful |
| 3D / depth (DA3D, 3D-DA) | not pursued | policy decision |
| Multi-view (wrist + tool) | not pursued | single-view tool_camera only |

→ **결론**: 진짜 lever는 **architecture (VLM)** + **data (hold-rich oversample)**. Loss + inference는 supporting; flow matching이 main BC objective로 충분.

---

## 6. Reproducibility checklist

### Environment
- `EGL_PLATFORM=device` (no Mesa) — [[feedback_mujoco_eval_hang]]
- GPU 0 dead → CUDA_VISIBLE_DEVICES offset (CV=0 → physical GPU 1) [[feedback_gpu_index_mapping]]
- 4+ eval 동시 EGL init 금지 → 30s stagger [[feedback_nvkms_lock_race]]

### Training
- Default lr **1e-6** for finetune, **3e-7** for cascade2 (v11) [[feedback_learning_rate_ceiling]]
- `train.py` deepspeed/non-deepspeed branches 둘 다 인자 동시 추가 [[feedback_train_branch_parity]]
- `PYTHONPATH=.` 필수

### Eval
- `--no-early-term` 필수 ★
- `--num-steps-execute 2` (Qwen) [[feedback_inference_axis_exec2]]
- Multi-metric rank (`scripts/rank_models.py`) — SR 단일 신뢰 금지 [[feedback_model_ranking_composite]]

### Data
- approach_00 cap 5000 (y-bias fix)
- y-balance via yneg_v1 + ypos_v1
- submm_hold_v1 (1.5mm perturb + 250 hold, 1800ep, 2× oversample) ★

---

## 7. Tooling

| Script | Purpose |
|---|---|
| `scripts/train.py` | VLANeXt training (deepspeed/single) |
| `scripts/sim_eval_align_only.py` | 27-cell eval (`--no-early-term` line 1485-1487) |
| `scripts/honest_metrics.py` | 8-metric from traj_ep*.npz |
| `scripts/compare_honest_vs_origterm.py` | Δ honest vs early-term |
| `scripts/rank_models.py` | 9-metric composite rank-sum |
| `Sim/Save_dataset_align_NEARGOAL.py` | dataset gen (TPIK 3-priority hierarchy) |
| `Sim/11_submm_hold.sh` | submm_hold_v1 dataset gen |
| `figures/make_main_comparison.py` | paper figure generation |
| `figures/make_failure_topology.py` | failure-mode scatter |
| `figures/make_encoder_ablation.py` | encoder ablation figures |

---

## 8. File map

| | Path |
|---|---|
| Paper SOTA config | `config/sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml` |
| Paper SOTA ckpt | `checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_flat_1500.pt` |
| Paper SOTA eval | `…/v11_submm_tight/align_eval_step1500_exec2_diff10_noET/` |
| Figures (PPT-ready) | `figures/fig{1..6}_*.png` + `.pdf` |
| Ablation analysis | `ablation.md` |

---

## 9. Open work / future axes

1. **Encoder ablation chain matching** — DINOv3 + ConvNeXt chain50k training in progress (2026-05-25). Update §B once eval done.
2. **Multi-seed evaluation** of v11 / champions — currently single seed 2026.
3. **ACT/DP with lateral_mm trajectory logging** for full Settled comparison.
4. **Input resolution upgrade** (256 → 384/512) — needs sim HDF5 regen [[project_input_resolution_ceiling]].
5. **Real robot transfer** (in progress, user side) [[project_real_align_deploy_start]].

---

## 10. Archive

| | Path |
|---|---|
| Pre-2026-05-23 daily logs | `attic/EXPERIMENTS_fine_align_history.md` (2286 lines, deprecated) |
| Pre-honest era backup | `attic/EXPERIMENTS_fine_align.md.bak_pre_honest_20260524` |
| Pre-architecture-correction backup | `attic/EXPERIMENTS_fine_align.md.bak_pre_qwen_correct_20260525` |
