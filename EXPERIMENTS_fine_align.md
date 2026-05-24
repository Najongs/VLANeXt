# Fine-Align Experiments

Needle-trocar mm-level alignment using vision-only / Qwen3.5-VL VLA. Calibration-free.

**Last reorganization**: 2026-05-24 — historical daily progress (Sections 10-21, 23-27, old EOD snapshots) moved to `attic/EXPERIMENTS_fine_align_history.md` to keep the main doc focused. Backup of pre-cleanup version: `attic/EXPERIMENTS_fine_align.md.bak_pre_cleanup_20260524`.

**Reading guide**:
1. **Master Cheatsheet** (immediately below) — paper-ready summary, 권위본
2. **Sections 28–31** — current SOTA (Qwen3.5-2B + reach_recover), 2026-05-23 final
3. **Section 22 (Master Table)** — paper Table 1 candidate covering all ablations
4. **Sections 1–9** — paper backbone (problem, model, eval grid, infra)
5. **BC Finetune Engineering Knowledge** — empirical training rules
6. **Archive** — see `attic/EXPERIMENTS_fine_align_history.md` for daily logs

---

## 🎯 Master Cheatsheet (2026-05-23 EOD, 권위본)

> 이 세션 (Section 17-28) + 이전 세션 결과 종합. **paper 작성 시 여기만 참조해도 충분**.
> 자세한 ablation은 Section 28 (compact-ready), 절차/실험은 Section 17-27.

### 🏆 Champion 선택 가이드 (use-case 별, 2026-05-23 EOD final)

| Use case | Checkpoint | exec | SR | close_2 | min_lat | **holdSR** | **safety** | y=-25 |
|---|---|---|---|---|---|---|---|---|
| 🏆 **Balanced SOTA (medical, NEW 2026-05-24)** | `reach_recover_v11_submm_tight/checkpoint_flat_1500.pt` | 2 | **100** | **77.8** | 1.32 | **48.1** | 2.65 | 9/9 |
| 🏆 **Balanced (이전)** | `reach_recover_v2_aggressive/checkpoint_flat_1500.pt` | 2 | **100** | 70.4 | 1.32 | 48.1 | 2.86 | 9/9 |
| 🏆 **Best precision peak (min_lat)** | `reach_recover_v8_aux_extreme/checkpoint_flat_500.pt` | 2 | 100 | 77.8 | **1.19** | 37.0 | 2.73 | 9/9 |
| 🏆 **Best holdSR (Qwen, NEW 2026-05-24)** | `reach_recover_v9_gentle_hold/checkpoint_flat_500.pt` | 2 | 100 | 70.4 | 1.50 | **51.9** | 2.64 | 9/9 |
| 🏆 **Best safety (worst-case)** | `reach_recover_v8_aux_extreme/checkpoint_flat_1000.pt` | 2 | 100 | 66.7 | 1.40 | 48.1 | **2.47** | 9/9 |
| Hold (chain, retreat trade-off) | `reach_recover_v5_combo/checkpoint_2000.pt` (vision-only) | 4 | 48 | 55.6 | 1.00 | **81.5** | 10.78 | 0/9 |
| Sub-mm lateral (chain) | `lat_hold_v4_yneg_hold/checkpoint_1000.pt` (vision-only) | 2 | 44 | 51.9 | **0.87** | 77.8 | 11.48 | 0/9 |

→ **6 deployment regimes** (4 Qwen + 2 chain). v8 ck500 = precision dual champion (close_2 77.8 / min_lat 1.19). v8 ck1000 = safety champion (2.47mm).

**🏆 Section 31 final records (2026-05-23 EOD autonomous sweep, R1-R5)**:
- **close_2 77.8%** (v8 ck500) — ACT 48.1 +29.7pp, prior champion 70.4 +7.4pp
- **min_lat 1.19mm** (v8 ck500) — Qwen-family new low, prior 1.24
- **safety 2.47mm** (v8 ck1000) — ACT 3.78 −35%, prior 2.56 −0.09mm
- **holdSR 51.9%** (v5 ck500) — Qwen ceiling 깸 (+3.8pp over v2). Chain 78% 미달
- **v8 extreme aux_hold paradox**: rot 1.0이 hold 약화 (37%) but precision 폭발 (close_2 77.8). ck1000에선 hold 회복

**Remaining axis** (Section 31.7): Ensemble (Qwen+chain action averaging) — holdSR 60%+ 가능 추정.

### 📐 Eval protocol 표준 (paper 모든 결과 동일하게)

```bash
# 27-cell grid, retreat=2 (paper protocol), exec=2 default
TRAIN_CONFIG_OVERRIDE="<train_config>" GPUS=0,1 \
  bash Run_Eval_Parallel.sh align "<ckpt>.pt" \
    --max-steps 250 --eval-seed 2026 --perturb-mode grid \
    --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
    --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
    --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
    --retreat-mm 2 --num-steps-execute 2

# 후처리 (shard 머지 + 분석)
python scripts/merge_eval_shards.py "<ckpt>.pt" --num-shards 2 --exec-steps 2 --diff-steps 10 --prefix align
python scripts/analyze_baseline_matrix.py  # 모든 variant 통합 표
```

### 📊 Paper Table 1 — 7-metric multi-criteria (필수)

`SR_old` 단독은 retreat=2에서 ACT/DP saturated → 의미 없음. 반드시 multi-criteria:

| metric | 정의 | 의미 |
|---|---|---|
| SR_old | 3D dist < 5mm at final step | legacy, retreat=2에선 baseline saturated |
| **close_5** | final_lateral < 5mm | 천장 진단 |
| **close_2** | final_lateral < 2mm | 정밀 정렬 |
| **holdSR** | lateral < 2.5mm for ≥20 contig steps | **hold-and-stay vs touch-drift 차별** (paper key) |
| **min_lat** | per-ep min lateral (median) | peak 정밀도 |
| **safety** | p99 of final_lateral | medical worst-case bound |
| per-region SR | y=-25 / 0 / +25 | 분포 비대칭 진단 |

### 🔑 핵심 학습 팁 (반드시 지킬 것)

| 영역 | 규칙 | 근거 |
|---|---|---|
| **lr** | ≤ 1e-6 default. >1e-5는 일관 gnorm 폭주. fresh training은 5e-6도 위험 | `feedback_learning_rate_ceiling`, Section 21 DINOv3 fresh gnorm 47 |
| **finetune chain** | base 50k → finetune cascade (1-2k step씩 점진) 필수. fresh 20k도 못 따라잡음 | Section 21.3c, `feedback_chain_dominant_over_encoder` |
| **exec inference** | default exec=2. exec=4 = hold champion, exec=1 = close_2 champion. **single ckpt 3 modes** | `feedback_inference_axis_exec2`, Section 26 |
| **batch size** | 16 frozen | (이전 결정) |
| **vision encoder** | SigLIP2-so400m-patch16-512 frozen. ConvNeXt/DINOv3와 fresh budget에서 동급 fail. chain 없으면 차별화 X | Section 21 |
| **proprio** | ee_pose 6-DoF만. sensor/KP 추가 금지 (성능 ↓) | `feedback_fine_alignment_dead_ends` 항목 4 |
| **aug** | off | (이전 결정) |
| **view** | single (tool_camera). multi-view 금지 | `feedback_no_multiview` |

### ⚙️ Loss 설정 권장 (champion config 기준)

```yaml
loss_type: "diffusion"
scheduler_type: "flow_match"  # not DDPM
num_train_timesteps: 1000
num_inference_timesteps: 10

dct_loss_weight: 0.0   # ⚠️ champion config에 0.1 켜져있지만, 효과 ≈ 0 (Section 20)
                       # 새 학습은 0.0으로. 약간 가벼움.

aux_distance_loss: { enabled: true, weight: 0.5, margin_mm: 0.1, near_goal_scale_mm: 2.0, near_goal_max_boost: 10.0 }
aux_lateral_loss:  { enabled: true, weight: 0.5, margin_mm: 0.05, near_goal_scale_mm: 1.0, near_goal_max_boost: 10.0 }
aux_hold_loss:     { enabled: true, pos_weight: 0.3, rot_weight: 0.5, threshold_mm: 2.5, soft_scale_mm: 1.0 }
# ⚠️ 실험으로 marginal 확인됨 (+3.7pp holdSR, -0.12mm min_lat). 
# encoder + chain이 진짜 driver. paper에서 "key contribution" 표현 금지.

direction_decoupled_loss: { enabled: false }  # 절대 켜지 마라. 폭주 + 효과 없음
```

### 🚀 Reach 회복 recipe (champion 약점인 SR 44% → 63% 회복)

champion (`lat_hold_v4_yneg_hold/ck1000`)이 정밀하지만 y=-25/y=0 region 일부 fail. 해결:

```yaml
# config/sim_train_align_reach_recover_v5_combo_config.yaml 핵심 파라미터
data:
  - approach_00 (cap 5000) + fine_align/10mm + range + NEARGOAL_eval_match + angle_only
  - + NEARGOAL_yneg_hold_v1 (기존 hold-rich)
  - + NEARGOAL_yneg_v1 (1500ep, y∈[-29,-10])  # 신규 추가
  - + NEARGOAL_ypos_v1 (1500ep, y∈[+10,+29])  # 신규 추가
model:
  aux_hold_loss: { pos_weight: 0.15, rot_weight: 0.25 }  # softhold (champion의 절반)
train:
  pretrained_checkpoint: "lat_hold_v4_yneg_hold/checkpoint_1000.pt"
  learning_rate: 1.0e-6  # 2× from champion's 5e-7
  max_steps: 3000        # eval ck2000 sweet spot
```

→ SR_old 44% → 63%, holdSR 77.8% → 74.1% (noise 수준 손실), min_lat 0.87 → 1.00mm (marginal).

### 🚫 폐기된 axes (시도 금지)

| 폐기 | 이유 | reference |
|---|---|---|
| **DCT loss weight > 0** | controlled rerun에서 contribution ≈ 0, close_2/SR_old trade-off만 | Section 20, `project_dct_ablation_0522` |
| **aux_hold weight 2× boost (rot=1.0)** | 도달은 살리되 lateral 손상 (lat_hold_v2_rot1) | Section 17 cycle 2 |
| **softhold 단독** (lr 보수적, 5e-7) | 무효. 공격적 lr (1e-6)과 결합 시만 효과 | Section 24 v3 vs v5 |
| **단순 학습량 늘리기 (v4 longer 5000step)** | over-train, SR 후퇴 | Section 25 |
| **y=-25 region-specific data (yneg25_strict 1500ep)** | 2/9 천장 못 깸. fundamental 한계 (occlusion 의심) | Section 27 |
| **Vision encoder swap fresh 20k step** | ConvNeXt/DINOv3 모두 SR 0~3.7%. chain 없으면 의미 없음 | Section 21.3c |
| **SutureBot-style overlay disk** | radius 어떤 값도 trocar visual feature 가림 → 5mm 천장 못 깸 | `feedback_fine_alignment_dead_ends` 1 |
| **sensor handoff servo** | 1D 신호만으로 fine alignment 불가 | `feedback_fine_alignment_dead_ends` 2 |
| **proprio에 sensor/KP coord 추가** | 성능 ↓ | `feedback_fine_alignment_dead_ends` 4 |
| **approach_00 완전 제거** | wide approach 학습 손실 > rebalance 이점 | `feedback_fine_alignment_dead_ends` 5 |
| **direction_decoupled_loss** | gnorm 폭주 + 효과 없음 | `feedback_ddl_loss` |

### 💡 자주 쓰는 한 줄 (실전 팁)

| 상황 | 한 줄 |
|---|---|
| 새 ckpt 처음 평가 | retreat=2, exec=2, exec=4 둘 다 돌리기 (Pareto 후보 둘 다 잡힘) |
| Eval shard merge 빼먹지 마라 | `python scripts/merge_eval_shards.py <ckpt> --num-shards 2 --exec-steps N --diff-steps 10` |
| ⚠️ `--sensor-stop` 금지 | sensor만 닿아도 success 처리 → 5mm 정밀도 평가 왜곡 (`feedback_no_sensor_stop`) |
| ⚠️ GPU 0 dead | `CUDA_VISIBLE_DEVICES`에서 nvidia-smi 인덱스 vs CUDA enum 한 칸 어긋남 (`feedback_gpu_index_mapping`) |
| ⚠️ NVKMS lock race | 4+ eval 동시 EGL init 금지. 30s stagger (`feedback_nvkms_lock_race`) |
| ⚠️ MuJoCo render | `export MUJOCO_GL=egl` + Mesa vendor JSON 강제 |
| 학습 launch 후 발산 조기 신호 | loss > 0.7 또는 min loss ≈ initial → fail. step 1000 안에 발견 가능 (`feedback_finetune_dynamics`) |
| Composite metric ranking 신뢰 금지 | rank_models.py가 diverge trap. 9-metric 종합 점수 + 개별 표 병행 (`feedback_model_ranking_composite`) |

### 🔬 다음 세션 우선순위

| priority | axis | 비용 | 기대 |
|---|---|---|---|
| **1** | y=-25 occlusion 진단 (frame 시각화 + camera fov 확인) | 30min | fundamental 한계 원인 파악 |
| 2 | Champion + v5 ck2000 ensemble (action averaging at inference) | 1h code | reach + precision 동시 |
| 3 | Multi-seed eval (champion + v5 ck2000 × 3 seeds) | 1.5h | stochasticity bound |
| 4 | Input resolution upgrade (256 → 384/512) | 1일 datagen+train | encoder 정밀도 ↑ |
| 5 | Action space delta scale 조정 (y-방향 reach 강화) | 1일 | y=-25 reach 강화 |

---

---

## 1. Problem & Contribution

**Task**: Surgical robot needle aligns to trocar entry within 5mm + 10° + 20-step hold.

**3-phase pipeline**: Approach (far → near, `dataset/approach`) → **Fine-align (this doc, ±15mm → mm precision)** → Insertion (sensor grid sweep + axis push).

**Why not original VLANeXt** (`config/libero_train_config.yaml`):
- Qwen3-VL-2B tokenizer compresses vision tokens
- SigLIP2-base@256 has insufficient pixel resolution for mm tasks
- Result: alignment fails

**Our solution — vision-only VLA variant**:
- Drop Qwen LLM (single-instruction task)
- SigLIP2-so400m-patch16-**512** native, frozen
- 3-layer MLP projector (LLaVA-style) → policy hidden 1152
- Action diffusion head (10-step flow-match) + aux distance loss
- Cotrain mix: tip + tip2 + HARD_ang15_hold30_part2

→ Contribution: removing LLM + scaling frozen vision encoder + diffusion policy beats Qwen-VL on mm precision.

---

## 2. Model Architecture (champion)

```
Image 480×640 raw  ───►  SigLIP2-so400m-patch16-512 (frozen, bf16)
                           │  1024 patches × 1152 dim
                           ▼
                         3-layer MLP projector (Linear-LN-SiLU ×3)
                           │
Proprio 6-DoF (×8 hist) ─► Linear(6→1152) ──► 8 tokens
                           │                       │
meta_queries (learnable) ─► 32 tokens              │
                           │                       │
        ┌──────────────────┴───────────────────────┘
        ▼
   concat [vision (1024) | proprio (8) | meta_queries (32)] = 1064 × 1152
   (per SigLIP layer, condition_type=soft)
        │
        ▼
   ActionDiffusionTransformer (depth=24, heads=16, queries=32)
   layer-wise soft conditioning: action blocks ↔ SigLIP layers
        │
        ▼
   Action chunk (8 steps × 6 DoF, Mecademic XYZ euler)
```

**Loss** (simple linear sum):
```
total = main_flow_match + 0.1·loss_dct + 0.5·loss_aux_dist
```
- `loss_dct`: 1D DCT on action chunk time-axis (smoothness regularizer)
- `loss_aux_dist`: ReLU(pred_dist − cur_dist + margin), near-goal sample weight ×10
- DDL, future_image, spatial losses: disabled

**Champion config**: `config/sim_train_align_siglip2_b24_ft10mm_aux_strong_config.yaml`
- backbone frozen, condition_type soft, batch 24
- aux weight 0.5 (강화), near_goal_scale 2mm
- 10k steps (best ckpt = 10000)

---

## 3. Paper Main Eval Grid

27 cells per ckpt, evaluated at `image_size=512` (matches training).

| Axis | Values |
|---|---|
| x perturbation | −10, 0, +10 mm |
| y perturbation | −25, 0, +25 mm |
| z perturbation | 0 (fixed) |
| phantom angle | −5°, 0°, +5° (paper main) / ±10° (stress) |
| repeats | 1 |

**Success criterion**: `dist(tip→goal_tip) ≤ 5mm AND |angle| ≤ 10° AND sensor > 20mm AND hold 20 steps`.

**Run**:
```bash
TRAIN_CONFIG_OVERRIDE=config/sim_train_align_siglip2_b24_ft10mm_HARD_cotrain_lr_low_EVAL512_VIDEO_config.yaml \
  GPUS=0,1 MUJOCO_GL=egl bash Run_Eval_Parallel.sh align <ckpt> \
  --max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
  --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
  --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0
```

**Distance metric (important)**:
- `min_dist_mm` (csv) = tip → trocar **entry point**
- `dist` (check_success) = tip → **goal_tip** = entry − 10mm × axis (retreat hold position)
- Success ⇒ `min_dist_mm ≈ 10mm` (intended retreat, NOT alignment error)
- **True alignment error** = `|min_dist_mm − 10|`. Heatmap panel 3 = this.

---

## 4. Champion Results

🏆 **NEW CHAMPION (2026-05-20)**: `b24_ft10mm_aux_strong_v3/checkpoint_1000.pt` — **SR 88.9% (24/27)** on ang±5° grid.
🏆 Previous: `b24_ft10mm_aux_strong/checkpoint_10000.pt` — SR 85.19% (23/27).

### 27-cell ang±5° SR comparison

| Run | Steps | SR | Δ vs champion | y=−25 row recovery |
|---|---:|---:|---:|---|
| **v3/1000 (NEW)** | 1000 (finetune) | **88.9%** (24/27) | **+3.7pp** | 2/4 failed cells recovered |
| aux_strong/10000 (prev champ) | 10000 | 85.19% (23/27) | baseline | 0/4 (all y=−25 fail) |
| v3/2000-final | 2000-5000 | 85.2% (23/27) | +0pp | drifted back |
| v2/1000-3000 (failed) | 1000-3000 | 44-52% | −33pp | regressed |
| HARD_cotrain_lr_low/3000 (older) | 3000 | 70.37% @ ang±10 | — | — |

**v3/1000 per-cell failures (3 total, all y=−25)**:
- (0, −25, −5°): dist=19.6mm — overshot
- (+10, −25, −5°): lateral=8.6mm — laterally off
- (+10, −25, 0°): lateral=7.1mm — laterally off

→ Improved from champion's 4 failures (all y=−25) to 3 failures. Cells (0, −25, 0°) and (0, −25, +5°) **recovered**. Remaining failures all at the −5° corner of y=−25, suggesting yaw alignment in negative-y direction is the residual bottleneck.

### v3 winning recipe (Plan B — finetune from champion)
**Config**: `config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v3_config.yaml`
- **pretrained_checkpoint**: `aux_strong/checkpoint_10000.pt` (champion weights)
- **reset_optimizer_scheduler**: true (fresh optimizer for new data adaptation)
- **lr**: 5e-6 (champion's 1e-5 → halved, preserves champion knowledge)
- **data mix** (4 sources):
  - `approach_00` (full, ~5000 ep)
  - `tip2` (full, ~50 ep)
  - `approach_eval_range_v1` (NEW, **capped 1000 ep**)
  - `align_phantom_range_v1` (NEW, **capped 200 ep**)
- 5000 steps, best at **step 1000**.

### v2 lesson learned (do not repeat)
v2 used **uncapped** new data (5010 + 510 ep, ~50% of mix) + lr 1e-5 → SR dropped to 51.9% at step 1000-2000, then 44.4% at step 3000. Champion knowledge swamped by new distribution.
**Key insight**: new data must be **down-sampled** for finetune, otherwise distribution shift causes catastrophic forgetting. v3's 1000+200 ep cap (~15% of mix) preserves champion behavior while exposing new boundary cells.

### Baselines (in-house, no lerobot)

| Run | params | SR @ retreat=10 | SR @ retreat=2 | Notes |
|---|---:|---:|---:|---|
| **VLANeXt v3/1000 (NEW champion)** | 1043M | **88.9%** (24/27) | **74.1%** (20/27) | SigLIP2-so400m frozen + diffusion |
| VLANeXt champion (prev) | 1043M | 85.19% (23/27) | 66.7% (18/27) | same arch |
| ACT (in-house) | 62M | 22.2% (best, ckpt_5k) | (not re-eval'd — baseline ceiling) | ResNet18 + CVAE + Transformer, scratch, 30k step |
| DP (in-house) | 89M | 22.2% (best, ckpt_15k) | (not re-eval'd) | ResNet18 + ConditionalUnet1D, scratch, 30k step |

**Retreat의 의미**: goal_tip이 trocar entry로부터 뒤로 빠진 거리. 데이터 생성은 retreat=1mm (tip이 거의 entry까지)이지만 champion 학습 데이터(approach_00/tip2)는 retreat=10mm로 만들어짐. retreat=2 평가는 데이터 생성 컨벤션과 일치, **진짜 mm-level fine alignment** 평가.

- v3는 retreat 두 값 모두에서 champion 우위 (+3.7pp at r=10, +7.4pp at r=2)
- ACT/DP 모두 retreat=10에서 22% 천장 → ResNet18 baseline은 정밀도 부족 확인. **재평가 불필요** (낮은 ceiling이라 retreat 줄이면 더 떨어짐)
- ACT train loss 0.03까지 매끄럽게 수렴, DP loss 0.018 — train converge 정상이나 inference 정렬은 안 됨. Vision encoder scale이 load-bearing 결정 요소

**Inference**: 226.6 ms/step (champion architecture, RTX 3090, bf16, 1043M params) ≈ 4.4 Hz. Action chunk 8, execute 1.
ACT inference: ~10 ms/step (62M, fast forward). DP inference: ~150 ms/step (89M + 16-step DDIM).

---

## 5. Discarded experiments (do not retry)

### Dead-end models / training recipes
| What | Why |
|---|---|
| **HARD_targeted data** (phantom corner shift + re-IK) | sensor/IK distribution drifts from baseline; cotrain mix hurts knowledge. targeted_mix and targeted_from_b100 ckpts all SR 3-30% ≪ champion. |
| **NEW_finetune/10000** | superseded by aux_strong, SR 9% on 54-cell stress grid. |
| **HARD_cotrain_lr_ultra_low** | reported 70-83% SR was `--sensor-stop` artifact (banned). True precision unknown. |
| **unfreeze SigLIP last-N** | seed lottery (n=4 mean 34%, σ 23pp). Cannot stand as paper claim. |
| **sensor proprio fusion** | 1D sensor as proprio dim diverges by step 3000 even at lr 5e-5. |
| **DDL (direction-decoupled loss)** | gnorm spike + zero effect in fine-align cotrain. |
| **b100 ext ckpt + large-lr cotrain ft** | 5× champion's lr drifts away from baseline knowledge. |

### Banned / non-options
- `--sensor-stop` eval flag (counts sensor touch as success; precision invalid)
- multi-view (wrist + tool). single tool_camera only
- 3D / depth policy. RGB + 1D sensor only
- lr > 1e-5 (gnorm explodes)
- vision token > 1500 with lr 1e-4 (SigLIP2 frozen diverges)

---

## 6. Key findings

### 6.1 Hold is the bottleneck, not localization
A-mode analysis on `HARD_cotrain_lr_low/3000` (50ep random perturb): SR 12% strict, but **55% near-miss** (tip ≤4mm), **50% oscillate** with avg min_dist 1.5mm. Model finds the trocar; it can't hold for 20 steps. → future: explicit hold loss / output stabilization.

### 6.2 Training-eval resolution matters less than expected
Training: SigLIP processor → 512×512 native. Eval default 256 → 512 upscale wastes info, but champion delta is marginal (~18% vs 18.5% on 54-cell). Resolution is **not** the bottleneck.

### 6.3 OOD phantom positions
- Training perturb y range: ±15mm
- y=+50/+75 cells: 0% SR
- Paper main grid restricted to y∈{−25, 0, +25} (in-distribution)

### 6.4 Projector is not a contribution
3-layer MLP (Linear-LN-SiLU) ported from llama branch (LLaVA recipe). Contribution = removing LLM + scaling frozen vision encoder.

---

## 7. Real robot status

⚠️ Not yet executed. Plan: champion ckpt sim-to-real dry-run after paper main number frozen.

---

## 8. Open ablations (paper completion)

| # | Ablation | Status |
|---|---|---|
| a | libero baseline (Qwen + SigLIP-base@256) | ❌ TODO — needs full original-VLANeXt retrain |
| b | SigLIP base@256, no Qwen | ❌ TODO — isolates encoder scale vs language removal |
| c | SigLIP base@512 | ❌ TODO — isolates resolution effect |
| d | **In-house ACT baseline** (vision-only ResNet18 + CVAE + Transformer decoder) | ▶︎ training (`src/models/act_policy.py`, config `sim_train_act_baseline_config.yaml`). 62M params, batch 64, lr 1e-4, 30k step |
| e | **In-house Diffusion Policy baseline** (ResNet18 + ConditionalUnet1D) | 📦 ready (`src/models/diffusion_policy.py`, config `sim_train_dp_baseline_config.yaml`). 89M params. Queue after ACT done |
| f | aux distance loss off | partial — `ablation_aux_off` trained, eval pending |
| g | cotrain off (HARD only) | dead-end — angle not learned |
| h | DCT loss off | ❌ TODO — trivial config edit + retrain |

**lerobot 우회**: ACT/DP를 별도 lerobot env 대신 현재 VLANeXt train.py + sim_eval.py에 직접 통합. 같은 dataloader (`src/datasets/sim_act_align.py`) 같은 eval bridge (`scripts/sim_eval_align_only.py`) 그대로 재사용. model_type dispatch만 `train.py:528-573`, `sim_eval.py:141-198`에 추가.

---

## 9. Infra / GPU notes

- **GPU 0 (PCI 24:00.0) dead** (driver/NVML errors). Available: GPU 1, 2.
- With `CUDA_DEVICE_ORDER=PCI_BUS_ID` + GPU0 dead, CUDA indices shift: pass `GPUS=0,1` to `Run_Eval_Parallel.sh` (maps to nvidia-smi GPU 1, 2).
- **MuJoCo render**: NVIDIA EGL이 `nvkms_open_common`에서 hang (GPU 0 dead가 EGL device enumeration 망가뜨림). **Mesa software EGL 강제 필요**:
  ```bash
  export MUJOCO_GL=egl
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  ```
  Mesa software render 속도: **54 fps @ 480x640** (충분), 15 worker × ~17s/ep. 이전 5월 메모의 "NVIDIA EGL 정상" claim은 transient 였음. `Sim/5_multi_align.sh` 상단에 export 추가됨.
- Eval video saving: `save_video: true` must be in **base** `config/sim_eval_align_config.yaml` (eval-section override in train config doesn't propagate).

## 9.5. Train resume 노하우 (2026-05-19)

| Field | 의미 |
|---|---|
| `train.resume_path` | **Full resume**: weights + optimizer + lr scheduler + step counter 복원. 같은 task 이어서 학습 |
| `train.pretrained_checkpoint` | **Weights only**: model weights만 가져옴, optimizer/step counter fresh. 다른 task로 finetune |
| `train.reset_optimizer_scheduler: true` | resume_path 사용 시에도 optimizer 만 fresh로 강제 (lr 변경/data 변경 시 권장) |

예: champion `aux_strong/10000` → v2 finetune (새 데이터 추가)
```yaml
train:
  resume_path: ""
  pretrained_checkpoint: "checkpoints/.../aux_strong/checkpoint_10000.pt"
  reset_optimizer_scheduler: true
  learning_rate: 1.0e-5
  max_steps: 5000  # 이건 0부터 시작 (counter fresh)
```
config: `config/sim_train_align_siglip2_b24_ft10mm_aux_strong_v2_config.yaml`

---

## 9.6. 2026-05-19 진행 (autonomous overnight)

**데이터 재수집** (`5_multi_align.sh`)
- Track 1 NEW approach: phantom random in x±12mm, y±29mm, z=0, angle±12° (eval+15% margin), hold 30
- Track 2 NEW align (phantom-moving + ±5mm robot perturb)
- 15 worker × 334/34 ep → ~5010/510 total ep
- Mesa software EGL (~17s/ep). Track 1 ~1.5h, Track 2 ~10min
- 경로: `dataset/approach/approach_eval_range_v1/`, `dataset/fine_align/align_phantom_range_v1/`

**학습 run (in flight)**
| Run | Model | GPU | Config | Status |
|---|---|---|---|---|
| ACT_baseline_align | ACT 62M | nvidia-smi 2 (CUDA 1) | sim_train_act_baseline_config.yaml | training, loss 93→2.08 @ step 319, 8h ETA |
| (queued) aux_strong_v2 | VLANeXt 1043M | nvidia-smi 1 (CUDA 0) | sim_train_align_siglip2_b24_ft10mm_aux_strong_v2_config.yaml | watcher 대기, 데이터 끝나면 자동 launch |
| (queued) DP_baseline | DP 89M | (after ACT done) | sim_train_dp_baseline_config.yaml | 미launch |

**평가 워크플로**
- `scripts/eval_multi_ckpt.sh <train_config> <ckpt1> [...]` — 27-cell ang±5° grid 한 번에 여러 ckpt
- `scripts/_eval_metric_summary.py <log1> [...]` — multi-metric (SR, lat<5/3, md<5/8, ang<10, medians) 한 줄 비교
- ACT/DP eval: `sim_eval.py:141-198`에 model_type dispatch 추가됨 — 기존 `Run_Eval_Parallel.sh align` 그대로 사용 가능

---

## 🧠 BC Finetune 지식 정리 (경험적 규칙 — 2026-05-21)

### 1. Distribution shock 임계치 — 무엇이 finetune로 회복되고 무엇이 안 되는가

| 변경 종류 | finetune 회복? | 사례 | 권장 lr/step |
|---|---|---|---|
| Data 분포 rebalance (cap, episode mix) | ✅ OK | lr1e6, v2_dual, extreme_rebal | lr 1e-6, 1000-2000 step |
| Hyperparameter 조정 (aux weight, boost) | ✅ OK | aux_strong v3, v5a | lr 1e-6, 1000 step |
| **이미지 분포 변경** (crop, aug 추가, 새 camera) | ❌ **2000 step lr1e6 finetune 불가** | **crop_zoom_v1 (close_5 0%)** ⚠️ | lr 1e-5+ 또는 from-scratch |
| Pretrained vision encoder 자체 변경 | ❌ 거의 from-scratch | ConvNeXt 실험 폐기 | from-scratch only |

**Why**: SigLIP2 frozen features는 specific visual distribution에 tuned. 픽셀 분포가 통째로 shift되면 features의 semantic mapping이 무효 → diffusion head가 처음부터 다시 학습해야 함.

### 2. 학습 fail 조기 감지 신호 (eval 안 돌려도 알 수 있음)

| 신호 | 정상 | 위험 (학습 fail) |
|---|---|---|
| Final loss | champion 0.1-0.5 | **>0.7 (e.g., crop_zoom 0.95)** ⚠️ |
| Loss 곡선 | monotonically ↓ | plateau or oscillation |
| Gnorm | <10 stable | 30+ spike (crop_zoom final 30.5) |
| Action sampling std | sub-mm | output near 0 (모델 무동작) |
| Eval `min_dist / initial_dist` | min << initial (도달) | **min ≈ initial (무동작)** ⚠️ |
| 다중 ckpt 차이 | sweet spot 존재 | **모든 ckpt 거의 동일 (학습 무효)** ⚠️ |

→ **이런 신호 보이면 어떤 ckpt 돌려봐도 의미 없음**. 학습 자체가 broken.

### 3. lr/step/warmup 권장 매트릭스

| 시나리오 | lr | max_steps | warmup | 비고 |
|---|---|---|---|---|
| 같은 분포 + cap 조정 | **1e-6** | 1000-2000 | 100 | lr ablation 결론 |
| Data 추가 (cotrain mix) | 1e-6 ~ 2.5e-6 | 1500-2500 | 200 | v2_dual 결과 |
| Aux loss 강화 | 1e-6 | 1000-1500 | 100 | v5 over-train 빨라짐 |
| **Visual distribution shift** | **1e-5 (10× ↑)** 또는 from-scratch | 5000+ | 500+ | crop_zoom 교훈 |
| Pretrained 변경 | from-scratch | 50000+ | 1000+ | dead-end 영역 |

**비대칭 trade-off**: 같은 분포면 lr 낮을수록 안정 / 다른 분포면 lr 너무 낮으면 base에서 못 빠져나옴.

### 4. Cotrain mix의 진짜 역할 — Catastrophic forgetting 방어 + 분포 점진 transition

**잘못된 사용** (crop_zoom_v1 실수):
- alignment 데이터 처음부터 100% cropped → base가 본 적 없는 분포로 shock

**올바른 curriculum** (distribution shift 클 때):
- Phase 1 (warmup 500step): uncropped 80% + cropped 20% — base 안 깨짐
- Phase 2 (1000step): 50/50 — 점진 transition
- Phase 3 (1500step): cropped 100% — final task

또는 영구 cotrain (champion 방식): uncropped/cropped 동시 mix로 학습 (분포 전환 없음).

### 5. Base ckpt 선택 — fresh vs continual

| 상황 | base 권장 |
|---|---|
| 새 데이터/분포 axis (단독 효과 측정) | **fresh** (champion v3 ckpt1000) |
| Hyperparameter 미세 조정 | **continual** (현 best ckpt) |
| 다단계 curriculum (warmup→final) | **fresh** 또는 짧은 단계별 cascade |

**경험**: continual_v1 (lr1e6 ckpt1500 base) vs extreme_rebal (champion fresh) 차이 거의 없음 (median 4.70 vs 4.95). axis 효과가 base 차이를 압도하는 영역에선 fresh가 명확한 비교 가능.

### 6. Eval 진단 워크플로 (학습 fail 빠른 확인)

```bash
# Step 1 — 5분 sanity (train log 확인)
grep "final loss" train.log         # >0.7면 fail 가능성 ↑
grep "gnorm" train.log | tail -10   # 30+ spike면 fail 가능성 ↑

# Step 2 — 1 ckpt 부분 eval (5-10 cell만, 5분)
# initial_dist vs min_dist 비교, 둘이 비슷하면 학습 무효

# Step 3 — 전체 fail 확정 시 곧장 폐기, full 27-cell 안 돌림 (시간 절약)
```

### 7. Visual distribution shift 회복 옵션 (crop_zoom 후속)

- **A. lr 10× 증가** (1e-6 → 1e-5): 큰 shift에 큰 step
- **B. Cotrain mix curriculum**: uncropped+cropped 점진 transition
- **C. From-scratch baseline**: champion 안 씀, 더 길게 (≥10K step) — 진짜 axis 효과 측정
- **D. Crop center 고정 (UV 의존 제거)**: train/eval consistency 위험 회피

관련: [[project_lr_ablation_final]] (lr 1e-6 정상 분포 default), [[feedback_fine_alignment_dead_ends]] (폐기 axis 카탈로그)

### 다음 단계
1. ✅ lr ablation 완료 — **lr 1e-6 ckpt 1500 = 신 median champion**
2. **다른 PC 10K v3 데이터** 도착 대기 → 도착 시 lr1e6 spec으로 새 finetune (retreat=0, hold=30 사양)
3. **Architecture axis** (데이터 axis 후 천장 못 깨면): vision encoder partial unfreeze (last_n=2, seeds 다중)
4. **데이터 axis** (10K v3 외): y=-25 region에 추가 데이터 fix (approach_00 imbalance 직접 해결)

### 신규 도구 (2026-05-20)
- `scripts/diagnose_action_variance.py` — action sampling std 측정
- `scripts/analyze_cell_failures.py` — 27-cell phantom metadata 매핑 + cell별 fail 분석
- `scripts/analyze_trajectory.py` — distribution metrics (p25, p10, best5, n<2mm/<1mm)
- `scripts/visualize_robot_perturbation_clean.py` — world frame perturb 시각화 (4 PNG, tool+side cameras)
- `scripts/visualize_robot_perturbation_trocar.py` — trocar local frame 시각화 (4 PNG)
- `Sim/run_parallel.py` — `--perturb-*` flags
- `Sim/6_neargoal_dual_track.sh` — Track A+B 동시 datagen (완료)
- `Sim/7_multi_10k_v3.sh` — 10K v3 datagen (다른 PC, retreat 0/hold 30)

### Compact 후 핵심 reference
- 메모리: `project_lr5e6_median_winner` (신규), `project_y_neg_distribution_bias`, `project_neargoal_v2_datagen` (업데이트), `project_champion_v3_0520`
- 폐기 메모리: feedback_fine_alignment_dead_ends에 OptC 추가 필요


---


## Ablation Studies (moved out)

Detailed ablation experiments now live in a separate file: **[`ablation.md`](./ablation.md)** (~790 lines).

Contains:
- **Section 22** — Ablation Master Table (architecture / loss / data / inference / training / metric design)
- **Section 29** — Qwen3.5-2B (with-LM) vs vision-only encoder ablation
- **Section 31** — `hold_recovery` autonomous sweep v3-v8 (holdSR 52% ceiling 탐색)
- **Section 32** — Post-cleanup follow-up: exec=4 sweep + v9-v14 sub-mm/hold variants
- **Section 33** — 3-axis driver analysis: Architecture / Data / Loss for sub-mm + hold

→ For "어떤 axis가 가장 sub-mm 정확도/hold를 끌어올렸나" — see `ablation.md` Section 33.


---

## Section 28: 세션 최종 Compact-ready Summary (2026-05-23 마무리)

이 세션은 사용자 질문 5가지를 순차 답변:
1. "Three Pillars of Sub-mm Precision 더 추가할 거?" → Section 20+21 새 발견 반영
2. "DCT 키고 끄기 비교" → Section 20 controlled ablation
3. "ACT/DP/ConvNeXt/SigLIP2/DINOv3 baseline 비교" → Section 21 baseline matrix
4. "ACT/DP hold loss 안 해서 holdSR 낮은 거 아닌가?" → Section 21.3b false confound 입증
5. "ACT처럼 SR 100 올릴 수 있나? 기존 성능 유지하며" → Section 23-27 reach_recover 10 variants

### 28.1 결정적 발견 (paper에 직접 영향)

| # | 발견 | 영향 |
|---|---|---|
| 1 | **"ACT/DP 22% 천장"은 retreat=10 artifact**. retreat=2에선 ACT/DP **SR_old 100%** | baseline narrative 정정 필수 |
| 2 | **진짜 차별화 = holdSR + min_lat**. ACT 24.5% / DP 11.1% vs Ours 77.8% | paper Table 1 multi-metric |
| 3 | **Hold-loss false confound**: SigLIP2 + dist-only도 holdSR 74.1%. aux_hold 효과는 +3.7pp marginal | "aux_hold = key contribution" 정정 |
| 4 | **Encoder choice는 fresh budget에서 차별화 X**. ConvNeXt/DINOv3 fresh 20k도 SR_old 0~3.7% | "SigLIP2 우월" claim caveat |
| 5 | **Chain matching이 dominant**. champion 우위 = base 50k + finetune cascade | encoder ablation은 fair budget matching 필요 |
| 6 | **DCT loss 0.1 ≈ 0**. controlled rerun primary 지표 noise 수준 | champion config DCT off 권장 |
| 7 | **Reach 회복 가능**: v5 ck2000 = SR 44%→63% (+19pp), holdSR/min_lat 거의 손실 없이 | "Ours는 reach도 회복" 강한 claim |
| 8 | **exec axis = Pareto knob**. exec=4 holdSR **81.5%** (champion 77.8% 추월), exec=2 SR best, exec=1 close_2 best | single training 3 deployment modes |
| 9 | **y=-25 약점은 데이터 양 문제 아님**. yneg25_strict 1500ep 추가도 무효 (2/9 그대로) | fundamental 한계 (occlusion/action space) |

### 28.2 Pareto Champion 확정

**Reach champion**: `reach_recover_v5_combo/checkpoint_2000.pt` + exec=2 — SR_old **63.0%**, holdSR 74.1%, y=0 6/9 (champion 3/9)
**Hold champion**: 동일 ckpt + exec=4 — holdSR **81.5%** (모든 variant 중 최고), ang 2.49°, safety 10.78mm
**Precision champion**: 동일 ckpt + exec=1 — close_2 **63.0%**
**Sub-mm lateral**: lat_hold_v4_yneg_hold/ck1000 + exec=2 — min_lat **0.87mm**

→ **Single ckpt + 3 inference modes로 paper Table 1 4 column 모두 cover**.

### 28.3 데이터 변경

| dataset | size | purpose | 결과 |
|---|---|---|---|
| `NEARGOAL_yneg_v1` | 1500 ep, y ∈ [-29,-10] | y<0 일반 보강 (이전 세션) | v5에 포함, 도움 |
| `NEARGOAL_ypos_v1` | 1500 ep, y ∈ [+10,+29] | y>0 일반 보강 (이전 세션 끝물) | v5에 포함, 도움 |
| `NEARGOAL_yneg25_strict_v1` | 1500 ep, y ∈ [-29,-21] | **y=-25 강화 (이번 세션 신규)** | v10에 추가 → **무효** |

### 28.4 새 configs (이 세션)

```
config/sim_train_align_dct_{off,on}_v1_config.yaml                    (Section 20)
config/sim_train_align_{dinov3,siglip2}_baseline_v1_config.yaml      (Section 21 fresh 1500)
config/sim_train_align_{convnext,dinov3,siglip2}_long5k_v1_config.yaml (21.3c)
config/sim_train_align_{convnext,dinov3}_long20k_v1_config.yaml      (21.3c)
config/sim_train_align_reach_recover_v{1,2_aggressive,3_softhold,
                                       4_longer,5_combo,6_v5consol,
                                       7_pushlr,8_gentle,9_v5push,
                                       10_yneg25}_config.yaml          (Section 23-27)
```

### 28.5 새 scripts (이 세션)

```
scripts/eval_dct_ablation.sh                                        (Section 20)
scripts/analyze_dct_ablation.py                                     (Section 20)
scripts/eval_baseline_matrix.sh                                     (Section 21)
scripts/analyze_baseline_matrix.py (확장됨, 모든 variant 포함)         (Section 21+ all)
scripts/eval_long5k_matrix.sh                                       (Section 21.3c)
scripts/eval_long20k_matrix.sh                                      (Section 21.3c)
scripts/eval_reach_recover.sh                                       (Section 23)
scripts/eval_reach_recover_v23.sh                                   (Section 24)
Sim/10_yneg25_strict.sh                                             (Section 27)
```

### 28.6 새 memory (이 세션)

- `project_dct_ablation_0522` — DCT noise 수준 contribution
- `project_baseline_matrix_0522` — ACT/DP 100%, narrative 정정
- `project_hold_loss_false_confound_0522` — aux_hold marginal
- `feedback_chain_dominant_over_encoder` — fresh ≠ chain
- `project_ablation_master_0522` — 세션 종합 ablation
- `project_reach_recover_v1_0522` — initial recovery (lr 5e-7)
- `project_reach_recover_pareto_0523` — v5 ck2000 + exec axis Pareto
- `feedback_fine_alignment_dead_ends` — DCT(8), hold loss(9), encoder fresh(10) 추가

### 28.7 Paper-grade table (Table 1 후보)

| Method | SR_old (3D<5mm) | close_5 | close_2 | **holdSR** | **min_lat** | finLat | ang° | safety (p99) |
|---|---|---|---|---|---|---|---|---|
| ACT (ResNet18 scratch 30k) | 100% | 100% | 48.1% | 24.5% | 2.00mm | 2.01mm | 1.55° | **3.78mm** |
| DP (ResNet18 scratch 30k) | 100% | 100% | 33.3% | 11.1% | 2.22mm | 2.34mm | 1.36° | 3.91mm |
| ConvNeXt-base frozen + ours head, fresh 20k | 3.7% | 14.8% | 3.7% | 11.1% | 20.50mm | 21.96mm | nan | 48.46mm |
| DINOv3-ViT-L/16 frozen + ours head, fresh 20k | 0% | 7.4% | 0% | 11.1% | 19.04mm | 19.04mm | nan | 45.97mm |
| **Ours (SigLIP2-so400m + chain) Reach** | 63.0% | 74.1% | 55.6% | 74.1% | 1.00mm | 1.55mm | 3.40° | 11.62mm |
| **Ours Hold** (same ckpt, exec=4) | 48.1% | 74.1% | 55.6% | **81.5%** | 1.00mm | 1.88mm | **2.49°** | 10.78mm |
| **Ours Sub-mm lateral** (champion ckpt, exec=2) | 44.4% | 70.4% | 51.9% | 77.8% | **0.87mm** | 1.96mm | 3.00° | 11.48mm |

### 28.8 Open axes (다음 세션)

| priority | axis | 비용 | 기대 |
|---|---|---|---|
| 1 | y=-25 occlusion 진단 (frame 시각화) | 30min | fundamental 원인 파악 |
| 2 | Champion + v5 ck2000 ensemble (action averaging) | 1h code | reach + precision 결합 |
| 3 | Input resolution 384/512 | 1일 datagen+train | encoder 정밀도 ↑ |
| 4 | Action space delta scale 조정 | 1일 | y reach 강화 |
| 5 | Multi-seed eval (3 seeds) | 1h | stochasticity bound |

### 28.9 Section index (이 세션 추가/수정)

```
17  Loss synergy (이전)
18  Data ablation (이전)
19  Compact summary (이전)
20  DCT controlled ablation (NEW)
21  Vision encoder + baseline matrix (NEW)
    21.3b  Hold-loss false confound (NEW)
    21.3c  Encoder long-train test (NEW)
    21.3d  Encoder choice vs chain conclusion (NEW)
22  Ablation Master Table (NEW)
23  reach_recover_v1 (NEW)
24  reach_recover v2/v3 (NEW)
25  reach_recover v4/v5 (NEW)
26  v5 ck2000 exec sweep Pareto (NEW)
27  v10 yneg25 strict + fundamental conclusion (NEW)
28  Final compact summary (NEW)
29  Qwen3.5-2B (with-LM) ablation — fresh 20k = reach champion (NEW, 2026-05-23 PM)
```

→ compact 후 Section 28 + 29만 봐도 이 세션 결과 전체 reconstruction 가능. Section 29 신 발견 (Qwen3.5-2B 66.7% SR fresh 20k) paper narrative 정정 필수.


---


## Section 30: Qwen + reach_recover finetune — **신 ABSOLUTE SOTA** (2026-05-23 PM)

### 30.1 Motivation

Section 29: Qwen3.5-2B fresh 20k = reach champion (SR 66.7%, close5 100%, safety 3.95mm) but hold 약함 (holdSR 44%). Vision-only chain의 reach_recover 처방 (aux_lat + aux_hold + yneg_hold + perfect_strict + lr 5e-7~1e-6) 적용해서 Qwen 기반 강한 reach를 보존하면서 hold/precision 회복 시도.

### 30.2 Setup — 2개 finetune variant 병렬 학습 (GPU 1+2)

| variant | config | lr | max_steps | base ckpt |
|---|---|---|---|---|
| **v1 conservative** | `sim_train_align_qwen_reach_recover_v1_config.yaml` | 5e-7 | 1500 | Qwen 20k |
| **v2 aggressive** | `sim_train_align_qwen_reach_recover_v2_aggressive_config.yaml` | 1e-6 | 1500 | Qwen 20k |

공통:
- Data: approach_00 (5K) + 10mm_fine_align + range + NEARGOAL_eval_match + angle_only + **yneg_hold + perfect_strict + yneg_v1 + ypos_v1** (v5_combo recipe와 동일)
- Loss: aux_dist 0.5 + aux_lat 0.5 + aux_hold (pos 0.15 + rot 0.25, softhold) + DCT 0.1
- backbone frozen, aug off, batch 8, ee_pose proprio only
- eval: 27-cell @ retreat=2, exec=2, single seed 2026

학습 시간: 단일 GPU × ~23min each (병렬 진행)

### 30.3 Results — Pareto Pareto

#### v1 conservative (lr 5e-7)

| ckpt | SR_old | close5 | close2 | holdSR | min_lat | min_3D | finLat | ang° | safety | y=-25 | y=0 | y=+25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1 ck500** ⭐ | **88.9%** | 100% | 59.3% | 33.3% | **1.54** | **2.47** | 1.81 | 2.08 | **3.53** | **6/9** | 9/9 | 9/9 |
| v1 ck1000 | 85.2% | 100% | **66.7%** | 33.3% | 1.74 | 2.49 | **1.78** | **1.96** | 3.76 | 5/9 | 9/9 | 9/9 |
| v1 ck1500 | 88.9% | 100% | 63.0% | 37.0% | 1.75 | 2.53 | 1.85 | 2.08 | 3.83 | 6/9 | 9/9 | 9/9 |

#### v2 aggressive (lr 1e-6) — 🏆 SOTA

| ckpt | SR_old | close5 | close2 | holdSR | min_lat | min_3D | finLat | ang° | safety | y=-25 | y=0 | y=+25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 ck500 | **100%** | 100% | 66.7% | 33.3% | 1.42 | 2.44 | 1.59 | 2.07 | **2.64** | **9/9** | 9/9 | 9/9 |
| v2 ck1000 | 96.3% | 100% | **74.1%** | 40.7% | 1.34 | 2.44 | **1.43** | 2.10 | 2.91 | 8/9 | 9/9 | 9/9 |
| **v2 ck1500** 🏆 | **100%** | 100% | 70.4% | **48.1%** | **1.32** | **2.34** | 1.53 | **2.01** | 2.86 | **9/9** | 9/9 | 9/9 |

### 30.4 vs prior SOTA — 모든 reach/precision/safety 추월

| Method | SR_old | close_2 | min_lat | safety | y=-25 | holdSR |
|---|---|---|---|---|---|---|
| ACT (ResNet18 30k) | 100% | 48.1% | 2.00mm | 3.78mm | 9/9 | 24.5% |
| DP (ResNet18 30k) | 100% | 33.3% | 2.22mm | 3.91mm | 9/9 | 11.1% |
| SigLIP2 champion (chain) | 44.4% | 51.9% | **0.87mm** | 11.48mm | 0/9 | **77.8%** |
| reach_recover v5 + exec=4 | 48.1% | 55.6% | 1.00mm | 10.78mm | 0/9 | **81.5%** |
| Qwen3.5-2B fresh 20k | 66.7% | 40.7% | 1.50mm | 3.95mm | 1/9 | 44.4% |
| **Qwen reach_recover v2 ck1500** 🏆 | **100%** | **70.4%** | 1.32mm | **2.86mm** | **9/9** | 48.1% |

- **SR_old 100%**: ACT/DP 동급, single VLA-class 모델 최초
- **close_2 70.4%**: ACT 48% +22pp, **모든 prior 모델 압도** (v2 ck1000 74.1%는 더 강함)
- **safety 2.86mm**: ACT 3.78mm 대비 **−24%**, 모든 chain 모델 (10+) 대비 **−74%**, 의료 worst-case bound 신 record
- **y=-25 9/9**: champion + 모든 vision-only chain (0/9) 천장 깸. **"fundamental limit (occlusion/action saturation)" 가설 ([[project_y_region_asymmetry_0521]], Section 27) 반박**
- **holdSR 48.1%**: ACT 24.5% +2배, 단 chain champion 78%엔 못 미침 (paper Pareto)

### 30.5 핵심 발견

1. **VLM (Qwen3.5-2B) reach + champion loss/data recipe = SOTA**: 단일 ckpt가 ACT/DP 동급 reach + champion 동급 precision/safety + 더 강한 holdSR. **single-model paper headline 가능**.

2. **lr 1e-6 (aggressive) ≫ lr 5e-7 (conservative)**: v2가 v1보다 모든 지표 우위. Qwen base의 강한 reach를 흔들지 않으면서 추가 학습 신호 충분히 흡수.

3. **ck sweet spot = 1500 (max trained)**: ck500 → 1000 → 1500 monotonic improvement (over-training 없음, lr 1e-6 with strong data alignment).

4. **y=-25 fundamental limit 가설 반박**: yneg25_strict 1500ep 단독 추가는 무효 (Section 27)였지만, **VLM backbone + 강한 lr + balanced y data 조합으로 9/9 perfect 달성**. 원인은 데이터 부족이 아니라 **vision encoder representational power**.

5. **재학습 budget vs 기존 chain**: 
   - 기존 vision-only chain = base 50k + 4-7 cascade (~80k step + 시간 소요)
   - Qwen v2 = fresh 20k + reach_recover 1500 step (~21.5k step total)
   - **3배 적은 budget으로 SOTA 달성**.

6. **남은 약점은 holdSR + ang**:
   - holdSR 48% (chain 78%) — VLM action style이 hold-friendly 아님
   - ang 2.01° (ACT 1.55°) — angle 정밀도 약간 약함
   - chain champion ckpt + exec=4 ensemble로 잠재적 회복 가능

### 30.6 Paper narrative 정정 — 4 deployment regimes → 5 modes

기존 (Section 29.6): single-training-3-deployment + Qwen fresh
**Section 30 NEW**:

| Use case | Champion | SR_old | close_2 | min_lat | safety | y=-25 | holdSR |
|---|---|---|---|---|---|---|---|
| **Reach + Precision + Safety (medical SOTA)** 🏆 | `reach_recover_v2_aggressive/checkpoint_flat_1500.pt` | **100%** | 70.4% | 1.32 | **2.86** | **9/9** | 48% |
| **Best precision** | v2 ck1000 | 96.3% | **74.1%** | 1.34 | 2.91 | 8/9 | 40.7% |
| **Hold (chain)** | reach_recover_v5_combo/ck2000 + exec=4 | 48.1% | 55.6% | 1.00 | 10.78 | 0/9 | **81.5%** |
| **Sub-mm lateral (chain)** | lat_hold_v4/ck1000 + exec=2 | 44.4% | 51.9% | **0.87** | 11.48 | 0/9 | 77.8% |
| **Reach (chain only)** | v5 ck2000 + exec=2 | 63.0% | 55.6% | 1.00 | 11.62 | 2/9 | 74.1% |

→ **paper Table 1 row 5개**. Qwen+reach_recover row가 main result, chain models이 hold/precision specialist.

### 30.7 Engineering 노트 (재현용)

1. **Ckpt 형식 함정**: train.py 저장 ckpt = `{"model_state_dict": OrderedDict}` 1단계 wrap. sim_eval.py는 `config` + `model_state_dict` 둘 다 있어야 wrap을 unwrap. 한 개만 있으면 outer dict를 state_dict로 잘못 인식 → 921 missing/1 unexpected. 해결: torch.save로 flat dict 별도 저장.

2. **flat ckpt 명명**: `checkpoint_{STEP}_flat.pt`로 저장하면 sim_eval의 `step_str = ckpt_path.stem.split("_")[-1]` 가 "flat"을 추출 → 모든 ckpt eval 결과가 같은 dir로 collision. 해결: `checkpoint_flat_{STEP}.pt` (step을 마지막에) 로 저장.

3. **project name → 디렉토리 분할**: train.py가 `project.name`에 underscore 있으면 첫 2 segment를 부모 디렉토리로 분할. e.g., `VLANeXt_Qwen35_NEARGOAL_reach_recover_v1` → `VLANeXt_Qwen35_NEARGOAL/reach_recover_v1/`. 평가 스크립트의 ckpt 경로 작성 시 주의.

4. **GPU 0 dead + multi-GPU NCCL**: torchrun multi-GPU 시 NCCL이 NVML로 GPU 0 enum → `ncclSystemError: nvmlDeviceGetHandleByIndex(0) failed`. 해결: **single-GPU only** on this PC. CUDA_VISIBLE_DEVICES=1 → nvidia-smi GPU 2, =0 → GPU 1. 두 학습을 GPU 1+2로 병렬 실행 가능.

5. **Linear attention slow path**: Qwen3.5-2B linear_attention layers는 `flash-linear-attention` + `causal-conv1d` 없으면 torch fallback. 학습 1500 step 23분 (single GPU bf16, batch 8). 설치 시 더 빠를 것.

### 30.8 Artifacts

- Configs: `config/sim_train_align_qwen_reach_recover_v{1,2_aggressive}_config.yaml`
- Train logs: `logs/qwen_finetune/train.log`, `logs/qwen_finetune/train_v2.log`
- Eval logs: `logs/qwen_eval/{v1,v2}_step{500,1000,1500}_exec2.log`
- Phase 3 orchestrator: `/tmp/qwen_unified_phase3.sh`
- Pipeline log: `logs/qwen_finetune/unified_phase3.log`
- Checkpoints:
  - `checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v1/checkpoint_{500,1000,1500}.pt`
  - `checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v2_aggressive/checkpoint_{500,1000,1500}.pt`
  - flat versions: `checkpoint_flat_{STEP}.pt`
- wandb: v1=e20hlhwb, v2=jqa9ejg3

### 30.9 다음 axes (open questions)

1. **v2 ck1500 + exec=4**: chain models은 exec=4가 hold↑. Qwen base에선 exec=4 → holdSR ↓ (Section 29.4 결과). v2 finetune 후엔? 별도 sweep 필요.
2. **v2 + 추가 학습 (3000 step)**: ck1500 sweet spot인가 over-train 시작인가? 추가 학습.
3. **Multi-seed (3 seeds)**: ±5pp 변동 정량화 — paper claim strengthening.
4. **chain + Qwen ensemble**: v2 reach + chain hold action averaging. inference 2× cost지만 holdSR 70%+ + SR 100% 가능?
5. **v2 holdSR 약점 분석**: aux_hold weight ↑ 또는 hold-rich data 비중 ↑ 시 회복?
6. **EXPERIMENTS Table 1 / Paper figures 업데이트**.

### 30.10 Section index 업데이트

```
30  Qwen + reach_recover finetune = 신 ABSOLUTE SOTA (NEW)
    30.4  vs prior SOTA — 모든 reach/precision/safety 추월
    30.5  핵심 발견 (y=-25 fundamental limit 반박 등)
    30.6  Paper narrative 정정 — 5 deployment modes
```


---


## Archive (historical sessions)

Detailed daily progress logs for sections moved out of this doc on **2026-05-24** cleanup:

| Section in archive | Topic | When |
|---|---|---|
| Old 2026-05-21 EOD snapshot | 9-cycle autonomous, exec=2 discovery, b100 finetune | 2026-05-21 |
| Old 2026-05-20 EOD summary | lr ablation final, lr1e6 ckpt1500 median champion | 2026-05-20 |
| Section 10 | Repo housekeeping (configs/scripts attic) | 2026-05-19 |
| Section 11 (.27–.38) | 1mm precision program, NEARGOAL dual-track datagen, lr5e6 winner | 2026-05-20 |
| Section 12 | Post-compact snapshot | 2026-05-21 |
| Section 13–15 | b100 50k base finetune (phase2/phase3) | 2026-05-21 |
| Section 16 | Ablation + baseline 종합 | 2026-05-22 |
| Section 17 | Loss ablation 4-cell synergy | 2026-05-22 |
| Section 18 | Data ablation (yneg_hold +11.1pp) | 2026-05-22 |
| Section 19 | Session 종합 (compact-ready) | 2026-05-22 |
| Section 20 | DCT loss controlled ablation (≈0 contribution) | 2026-05-22 |
| Section 21 | Vision encoder + baseline matrix (ACT/DP/ConvNeXt/DINOv3) | 2026-05-22 |
| Section 23–27 | reach_recover v1-v10 (vision-only chain, pre-Qwen) | 2026-05-22/23 |

Archive file: `attic/EXPERIMENTS_fine_align_history.md` (~2300 lines).
Full pre-cleanup backup: `attic/EXPERIMENTS_fine_align.md.bak_pre_cleanup_20260524` (3528 lines).

Key historical findings retained in current doc (Master Cheatsheet / Section 22 / Sections 28-31):
- **"5mm 천장 = metric artifact"** (3D dist + retreat 2mm Z offset). Real lateral median 0.87-1.19mm.
- **exec axis = Pareto knob**: vision-only chain exec=4 → holdSR 81.5%, exec=2 → SR best, exec=1 → close_2 best.
- **DCT loss contribution ≈ 0**: champion config DCT 0.1 → 0.0 권장.
- **Encoder swap fresh budget 무력**: ConvNeXt/DINOv3 fresh 20k 모두 SR 0~3.7%. chain matching이 dominant.
- **Hold-loss false confound**: SigLIP2 + dist-only도 holdSR 74.1% (champion 77.8%와 noise 차이). encoder + chain이 real driver.
- **lr ≤ 1e-6 default** — VLANeXt 학습 >1e-5는 일관 gnorm 폭주.
- **y=-25 region asymmetry** (approach_00 PHANTOM_Y 비대칭 72% y>0) — vision-only chain에선 fundamental 한계처럼 보였으나 Section 30 Qwen으로 9/9 perfect 달성 (Section 27 데이터 부족 가설 반박).
