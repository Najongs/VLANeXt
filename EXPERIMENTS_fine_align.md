# Fine-Align Experiments

Needle-trocar mm-level alignment using vision-only VLA. Calibration-free.
Last reorg: 2026-05-19. Older logs in `attic/EXPERIMENTS_fine_align.md.bak_*`.

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

## 10. Repo housekeeping (2026-05-19)

- 70 → 18 active configs. Dead-ends → `config/archive/`.
- 35 → 18 active scripts. KP (demoted per `project_kp_role_brake`), libero benches, sensor_handoff → `scripts/attic/`.
- EXPERIMENTS backups → `attic/`.
- Background orphans (until-loop polling abandoned HARD_targeted datagen) killed.
- Untouched: `checkpoints/`, `dataset/`, `aTrained_model/`, `lerobot/`, `Sim/`, `logs/`, `outputs/`, `wandb/`.

---

## 11. 2026-05-20 1mm-Precision Program (Stage 1 + Stage 2)

### Motivation
Champion v3 caps at 5.46mm mean min_dist (retreat=2 SR 74.1%). User asked: 어떻게 mm-precision까지 갈 수 있을까?

Researched 5 surgical/precision VLA papers:
- **SutureBot** (NeurIPS 2025, arXiv:2510.20965) — goal-pixel overlay → ACT 3.2→1.3mm, π0 3.9→**1.0mm**. Only paper with mm-scale gain on architecturally similar VLA stack. **→ adopt**
- **DSP** (ICLR 2025) — noise self-filter, easy. Defer (real-data integration round)
- **DP4AuSu** (2025 Wiley) — DTW LWR demo preprocess, unverified 1mm claim, no code. Skip
- **SutureAgent** (2026 arXiv) — predicts pixels, not actions. Wrong layer
- **Dreamer v3 microrobot** (Nature MI 2025) — different physics, RL infra rewrite. Skip

### Precision diagnosis (3 bottlenecks)
1. **Primary**: inference `num_inference_timesteps=10` (`VLANeXt.py:1623`) — 8-step action chunk × 10 denoising = sub-mm refinement 불가
2. **Secondary**: 학습엔 GT trocar_entry_pos 입력, **inference엔 명시적 goal signal 없음** — wrist 카메라만으로 trocar 위치 추론 (정확히 SutureBot이 해결하는 문제)
3. **Tertiary**: 256×256 + patch16 = 패치당 ~2-3mm. Stage 1/2 후 별도 axis

### Stage 1 — Inference sweep (in flight, ETA ~1.5h)
- `scripts/sweep_diff_exec.sh` — diff_steps × exec_steps grid on v3/1000 @ retreat=2
- 8 cells: (diff, exec) ∈ {(10,1), (25,1), (50,1), (100,1), (25,2), (50,2), (25,4), (50,4)}
- Added `--num-inference-timesteps`, `--num-steps-execute` CLI to `sim_eval_align_only.py`
- Added `close_once_1mm_pct`, `close_once_2mm_pct`, `time_near_1mm`, `time_near_2mm`, `p50/p90 dist` to `analyze_trajectory.py`
- Results: `/tmp/sweep_diff_exec_results.md`

### Stage 2 — SutureBot goal-overlay (pipeline ready, waits for Stage 1)
**Key discovery during impl**: HDF5 already has GT UV in `observations/keypoints_wrist[:, 2:4]` (normalized [0,1]) + `keypoints_visibility[:, 1]`. **No projection code needed** — Save_dataset_*.py:project_to_2d already ran offline during collection.

**Files**:
- `src/utils/overlay_utils.py` — `draw_overlay()` + `apply_overlay_batch()` (cv2.circle, supports dropout/jitter)
- `src/datasets/sim_act_align.py` — overlay_enabled/color/radius/dropout/jitter params, applied BEFORE local crop/resize
- `scripts/train.py` — dataset wiring for overlay options
- `scripts/sim_eval_align_only.py` — `--overlay-source {gt,predicted,off}` CLI + apply on `frames["tool_camera"]` before preprocess
- `config/sim_train_align_siglip2_overlay_v1_config.yaml` — finetune from v3/1000, lr 5e-6, max_steps 3000, save 500, overlay red dot radius 3 + dropout 0.1

**Sanity (read-only)**:
- Color collision: 0 pure red px in all sampled tool_camera frames (any of red/blue/green safe)
- Dataloader smoke: 39-frame ep, mean 29.6 red px/frame (radius 3 disk area ≈28), 35/39 frames have overlay (10% dropout exact)
- Visual: red dot lands on trocar entry hole — see `/tmp/overlay_preview_zoom.png`

**Eval cells (after train)**:
1. v3/1000 baseline (no overlay) — reference
2. overlay_v1 + GT UV (oracle ceiling)
3. overlay_v1 + predicted UV (실전 시나리오, kp head 4.4px err)
4. overlay_v1 + no overlay (ablation: dependence on overlay)

**Targets**:
- Cell 3 (predicted): mean min_dist ≤ 3mm, SR(close_once_2mm) ≥ 60% (~3-4x precision gain expected per SutureBot)
- Cell 2 (GT oracle): mean min_dist ≤ 1.5mm → tertiary 천장 (vision resolution) 진짜 다음 axis 확정

### Autonomous orchestration
- Watcher 1 (`/tmp/launch_stage2_after_sweep.sh`): sweep PID 1090065 wait → overlay_v1 smoke train (200 steps) on GPU 1
- Watcher 2 (`/tmp/launch_overlay_full_then_eval.sh`): smoke log success → full 3000-step train → 4-cell eval (`scripts/eval_overlay_4cell.sh`)

### 11.1 Run-time bug fixes (2026-05-20 08:00-08:25)

1. **UnboundLocalError**: `from src.utils.overlay_utils import draw_overlay` inside conditional block shadowed module-level `draw_overlay` (sim_eval's replay overlay). → renamed to `_draw_goal_overlay` alias.
2. **CLI override silent fail**: `if "key" not in DictConfig` raises TypeError → removed guard since cfg.model/eval always exist.
3. **Pipe truncation**: sweep v1 used `2>&1 | tail -20` losing real errors and making `set -e` blind. v2/v3 save full per-cell logs to `/tmp/sweep_v3_logs/`.
4. **Stale npz contamination**: v1 cell 1 showed SR 3% because prior session's eval dirs (retreat=10) mixed with new (retreat=2). v3 sweep `rm -rf` each target dir before run.
5. **ckpt path mismatch**: project name `VLANeXt_SigLIP2_overlay_v1` → train saved to `VLANeXt_SigLIP2_overlay/v1/`. Watcher had wrong path; bash variable cached too early to fix in-flight.
6. **KP head ckpt naming**: actual is `head_best.pt` not `best.pt`. Smart eval fixed.

### 11.2 Smart eval design (per user 2026-05-20 ranking guidance)

User feedback: "SR지표가 비정확할 수 있으니 목표 지점에 정확하게 도달하는 다른 지표들이 많았는데 그걸 기준으로 종합적으로 판단해서 모델 좋은걸 찾아줘"

- **`scripts/rank_models.py`**: 9-metric rank-sum (close_once_2mm, close_once_1mm, time_near_2mm, handoff_ok, min_dist_mean, p90_dist, lateral_when_near, angle_when_near, retreat). Lower Σrank = better.
- **`scripts/eval_overlay_smart.sh`**: 2-stage
  - Stage A: sparse ckpt sweep (1000/2000/3000) with `--overlay-source predicted` (realistic) + v3 baseline reference
  - Stage B: ablation on winner with `gt` + `off` (oracle ceiling + dependency check)
- Output: `/tmp/overlay_smart_stageA.md`, `/tmp/overlay_smart_final.md`

### 11.3 Status snapshot (08:25)

- Sweep v3 cell 1 in progress (Episode 9/27, 66.7% SR ← real baseline, not v1's bogus 3%)
- Smart eval Stage A cell 1 (overlay_v1 step 1000 + predicted UV) started
- GPU 1 + 2 fully utilized

### 11.4 ⚠️ v1 overlay 진단: radius=3 px가 invisible (1.8% of SigLIP patch token)

Smart eval Stage A 중간 결과 (n=27 each):

| label | n | 2mm% | 1mm% | t≤2mm | handoff | mean_min | p90 | lat<5 | ang<5 | retreat | Σrank |
|-------|---|------|------|-------|---------|----------|-----|-------|-------|---------|-------|
| v3_baseline (sweep v3 cell 1) | 17 | 5.9 | 0.0 | 0.00 | 0.0 | 4.96 | 30.74 | 1.47 | 3.26 | 0.00 | 16 |
| ovlPred_step1000 | 27 | 7.4 | 0.0 | 0.00 | 0.0 | 5.29 | 30.02 | 1.49 | 3.98 | 0.00 | 18 |
| ovlPred_step2000 | 27 | 0.0 | 0.0 | 0.00 | 0.0 | 5.34 | 29.96 | 1.51 | 3.68 | 0.00 | 20 |

⚠️ overlay 모델이 v3 baseline 대비 모든 정밀도 지표에서 동일/나쁨.

**근본 원인 (math)**:
- v1 config: `radius_px: 3` @ 640×480 raw render
- → resize 256×256 = 1.2px disk (= 2.4px diameter)
- → SigLIP2 **patch16 (16×16=256 px²)** 1.8%만 차지 → patch token avg color에 0.018%만 기여 → **invisible**

**Stage 1 sweep v3 결과** (denoising count 비교, 진행 중 cell 1-2):

| diff | exec | n | SR5mm | mean_min |
|------|------|---|-------|----------|
| 10 | 1 | 17 | 58.8 | 4.96 |
| 25 | 1 | 27 | 48.1 | 5.65 |

→ 더 많은 denoising step이 도움 안 됨. Diagnosis #1 (denoising quantization) **부정**. Real bottleneck은 #2 (no goal signal) confirmed. Sweep v3 killed (cells 3-8 skipped).

### 11.5 v2 overlay 재학습 (radius=20, in-flight 09:34)

- `config/sim_train_align_siglip2_overlay_v2_config.yaml`: `radius_px: 20` (40px diameter @ 640×480 → 16px @ 256×256 = 1 full SigLIP patch token, **visible**)
- 나머지 v1과 동일 (lr 5e-6, max_steps 3000, dropout 0.1)
- GPU 2에서 학습, watcher가 자동 eval 발사
- v1 smart eval GPU 1에서 계속 (Stage B GT/off ablation으로 v1 진단 종결)

### 11.6 v2 (radius=20) + v3 (radius=30) 동시 진행 (10:35)

v1 Stage B 결과로 v2 단독으론 부족할 가능성 높아, radius ablation 위해 v3 (radius=30) 추가:

- `config/sim_train_align_siglip2_overlay_v3_config.yaml`: `radius_px: 30` (60px diameter @ 640×480 → 24px @ 256×256 = 1.5 SigLIP patch token)
- Watcher chain: v1 smart eval 종료 → v3 train on GPU 1 → v3 smart eval on GPU 1 (GPU 2는 v2 eval 진행중)

**v2 step1000 + predicted 중간 결과 (n=20/27)**:
- SR(close_5mm) = 40%, SR(close_2mm) = 0%, mean_min = 6.45mm
- v3 baseline (66.7% at ep18)보다 약간 낮으나 진행 중

**비교 결과 (10:34)**:
- v1 step3000 + off (overlay 없이): SR(close_5) 66.7% @ ep18 ≈ v3 baseline
- v1 step3000 + predicted: similar
- v1 step3000 + GT: similar
- → v1 radius=3 모델은 overlay 입력 완전히 무시 (radius bug 확정)

**예상 결과 (v2/v3 학습이 의미있다면)**:
- v2 (radius=20)부터 overlay 사용 학습 시작 가능
- v3 (radius=30)에서 명확한 SR2 향상 보여야 함
- 두 모델 모두 v3 baseline에 못 미치면 → overlay 접근 자체 폐기, 다른 방향 (e.g., proprio goal, hierarchical) 모색

### 11.7 v1 Stage B 최종 결론 (10:42)

```
| label                    | n  | 2mm% | mean_min | Σrank |
|--------------------------|----|------|----------|-------|
| v3_baseline              | 27 | 3.7  | 5.04     | 16 🏆 |
| ovlPred_step3000(gt)     | 27 | 3.7  | 5.42     | 19    |
| ovlPred_step3000(off)    | 27 | 0.0  | 5.45     | 20    |
```

- **GT oracle UV도 v3 baseline에 패배** → v1 모델은 overlay signal을 완전히 무시
- off (no overlay) ≈ GT ≈ predicted → 모델이 overlay 픽셀을 noise로 취급
- **radius=3 invisible 가설 (1.8% patch token area) 확정**

### 11.8 v2 (radius=20) 중간 결과 (11:05)

```
| label                  | n  | 2mm% | mean_min | Σrank |
|------------------------|----|------|----------|-------|
| v3_baseline            | 27 | 3.7  | 5.04     | 16 🏆 |
| v2_step1000_predicted  | 27 | 3.7  | 5.42     | 18    |
| v1_step3000_off        | 27 | 0.0  | 5.45     | 21    |
```

- v2 step1000은 v3 baseline 거의 동등 (mean_min 5.42 vs 5.04)
- step2000 partial (n=24): mean_min=6.15mm — 오히려 악화 추세
- step3000 + GT/off 필요

### 11.9 가설 재평가

v2 (radius=20)이 v1 (radius=3)와 유사하면 → "overlay 자체가 성능 boost 어려움":
1. **v3 baseline이 이미 vision으로 trocar 잘 localize 함** — 추가 goal signal 정보 가치 낮음
2. **Action precision bottleneck** (diffusion noise floor ~5mm) — vision improvement만으로 못 뚫음
3. **SutureBot 결과 vs ours**: 그쪽은 π0 3.9→1.0mm. 우리는 5.0mm fixed.
   - 가능한 이유: 우리 task 더 어려움? 우리 vision 이미 더 좋음? 데이터 적음?

**다음 방향 후보** (v2/v3 도 baseline 못 이기는 경우):
1. **Proprio goal**: trocar_world_mm을 proprio에 concat (overlay 대신 명시적 좌표 입력)
2. **Goal-conditioned diffusion**: trocar UV/dist를 action diffusion condition에 추가
3. **Multi-step refinement**: near-goal에서 별도 fine-tune된 작은 모델로 zoom-in
4. **Action representation 변경**: bin-quantized vs continuous diffusion

### 11.10 ⚠️ v2 (radius=20) 결과 — overlay 자체가 visual feature 손상

```
Model                  | close_2 | close_3 | close_5 | mean_min
-----------------------|---------|---------|---------|----------
v1 step3000 predicted  | 3.7%    | 25.9%   | 59.3%   | 5.04mm
v1 step3000 GT         | 0.0%    | 22.2%   | 55.6%   | 5.45mm
v1 step3000 off        | 3.7%    | 25.9%   | 59.3%   | 5.04mm  ← v1 무시
v2 step1000 predicted  | 0.0%    | 3.7%    | 51.9%   | 5.90mm  ← v2 손상
v2 step2000 predicted  | 0.0%    | 3.7%    | 55.6%   | 5.89mm
v2 step3000 predicted  | 0.0%    | 3.7%    | 51.9%   | 5.78mm
```

**중요 관찰**:
1. v1 predicted ≡ v1 off → 모델이 radius=3 overlay를 픽셀 노이즈로 무시 (radius invisible 확정)
2. v2 (visible radius=20)는 v1보다 **더 나쁨** (mean_min 5.78-5.90 > 5.04)
3. **새 가설**: visible overlay가 **trocar entry hole의 visual feature를 가림** (40px disk가 hole 위에 그려짐 → 모델이 hole pixel을 못 봄)

**구조적 결론**: SutureBot의 "opaque pixel marker" 접근은 우리 환경에서 부적합. 우리 trocar는 이미 카메라에 명확히 보이므로 marker가 redundant + 가림.

### 11.11 다음 방향: keypoint proprio injection

- 기존 코드 `use_keypoint_proprio=True` (proprio_dim=9 = ee_pose 6 + troc_uv 2 + dist_norm 1)
- Overlay 대신 **명시적 trocar UV/dist 좌표**를 proprio에 concat
- visual feature 가림 없이 goal signal 주입
- `project_keypoint_pipeline_0514` 학습된 head 그대로 사용 가능

대안:
- v3 (radius=30) 결과 확인 후 결정 (radius=30이 더 나으면 overlay 자체 가능성 재검토)
- v3도 v2와 유사하게 못 이기면 keypoint proprio로 pivot

### 11.12 v3 (radius=30) 초기 결과 + Pivot 결정

```
v3 (radius=30) step1000 predicted:
  close_once 1/2/3/5/8/10 mm: 0.0 / 0.0 / 0.0 / 51.9 / 81.5 / 88.9 %
  mean_min = 5.75mm
```

vs.
- v3 baseline: 5.04mm
- v1 step3000 pred: 5.04mm (overlay 무시)
- v2 step1000 pred: 5.90mm
- v3 step1000 pred: 5.75mm

→ **radius 클수록 더 나빠짐 (occlusion 가설 확정)**. SutureBot 접근법 우리 task에 부적합.

### 11.13 v4 keypoint proprio injection (12:10 launch)

- `config/sim_train_align_siglip2_kp_proprio_v4_config.yaml`
- `proprio_dim: 9`, `use_keypoint_proprio: true` (overlay 폐기)
- 명시적 `[troc_u, troc_v, dist_norm]` 3차원을 ee_pose(6)에 concat
- Visual feature 가림 없이 goal signal 주입
- pretrained_checkpoint: v3 SigLIP2_repro/b24_ft10mm_aux_strong/checkpoint_10000.pt (proprio_proj 6→9 random init)
- lr 5e-6 (v3 recipe), max_steps 3000, save 500
- 학습 후 auto-eval (watcher armed, GPU 2)

**예상**:
- Best case: proprio에 정확한 좌표 있으니 모델이 fine alignment 학습 쉬워짐, mean_min 4mm 이하 가능
- Worst case: proprio 신호도 무시 (champion 이미 vision만으로 잘하니 추가 signal 무의미) → 5mm 영역 정체
- Either case: overlay vs proprio 두 가설 결판

### 11.14 v3 (radius=30) 전체 ckpt 결과 — overlay 폐기 확정

```
v3 step1000 (radius=30, predicted):
  close 1/2/3/5/8/10mm: 0.0 / 0.0 / 0.0 / 51.9 / 81.5 / 88.9 %  | mean_min=5.75mm
v3 step2000:
  close 1/2/3/5/8/10mm: 0.0 / 0.0 / 0.0 / 55.6 / 81.5 / 88.9 %  | mean_min=5.62mm
v3 step3000:
  close 1/2/3/5/8/10mm: 0.0 / 0.0 / 0.0 / 48.1 / 81.5 / 88.9 %  | mean_min=5.66mm
```

vs v3 baseline (5.04mm, close_2mm=3.7%) — 모든 ckpt에서 baseline에 패배. 더 큰 overlay = 더 큰 occlusion. **overlay 접근법 결정적 폐기**.

### 11.15 Architecture 재검토 (user prompt 12:30)

User raised 3 questions:

1. **Proprio 묻힘 (1 / 289 tokens = 0.35%)**: 단일 proprio token이 256 vision token과 attention 경쟁 → 묻힐 가능성 높음
2. **Meta queries (32) — vision-only에서 무용**: VLM에서 cross-attend summarization 역할인데 우리는 LM 없음 → 그냥 32개 learnable register, 큰 도움 안 됨
3. **Vision encoder 거대 + 해상도 낮음**: SigLIP2-so400m native 512 vs 우리 256 입력 (encoder 능력 절반 활용). patch16 @ 256 = ~1.6mm/token (1mm precision 불가)

**HDF5 raw frames 640x480 검증됨** — `project_input_resolution_ceiling` 메모리 일부 오류 (HDF5 재생성 불필요, dataloader resize만 변경하면 됨)

### 11.16 v5 design plan (v4 oracle 결과 후 launch)

**v4 oracle test 추가**: `--oracle-kp` flag로 GT trocar 좌표 직접 주입 → proprio signal이 본질적으로 유효한지 확인. v4 watcher에 oracle eval 추가됨.

**v5 (v4 결과에 따라)**:
- v4 oracle >> predicted → proprio 효과 있음, KP head quality 개선 필요
- v4 oracle ≈ v3 baseline → proprio 1 token 흡수 안 됨 → 강화 필요 (multi-token, FiLM, replicated tokens)

**v5 후보 구성**:
| 변경 | 효과 |
|---|---|
| input_image_size 256→384 | tokens 256→576, 패치당 1.6→1.1mm |
| Multi-token proprio (ee_pose+goal_uv+goal_dist 분리) | proprio 토큰 1→3, attention 표면 3x |
| Remove meta_queries (or condition_type="tight") | 깔끔, vision token이 더 강조 |
| FiLM modulation (옵션) | goal coords로 vision feature 자체 변조 |
| smaller encoder (옵션, 효과 검증 후) | SigLIP-base or DINOv2-base 시 메모리 ↓ |

### 11.17 사용자 hard-won feedback (12:30)

| 시도 | 결과 |
|---|---|
| Sensor handoff | ❌ 효능 없음 (1D 신호 부족) — safety brake로만 |
| Proprio (ee_pose 6) | ✅ 효과 있음 |
| Proprio + sensor/KP | ❌ 떨어짐 |
| aux_distance_loss ↑ | ✅ **올릴수록 좋아짐** (v5a 1st-axis) |
| CNN encoder | ✅ but **unfreeze 필수** |

### 11.18 v4 (kp proprio) catastrophic failure (13:10 confirm)
- 24/27 eps ALL FAIL, dist 9~20mm
- 사용자 prediction 정확. proprio injection path 폐기.

### 11.19 v5 launches (13:25-)

**v5a (GPU 1)**: champion v3 base + aux_distance_loss boost (weight 0.5→1.0, max_boost 10→50). 1500 step finetune, lr 5e-6. 사용자 추천 1st-axis.

**v5b (GPU 2)**: ConvNeXtV2-base-384 + backbone_mode=finetune (full unfreeze). lr 1e-6 (보수적). 3000 step from-scratch effectively (pretrained=ImageNet only). 사용자 직관: CNN 쓸 거면 unfreeze 필수.

Watcher 자동 eval (v5a: ckpts 500/1000/1500, v5b: ckpts 1500/3000) on completion.

### 11.20 v5b (ConvNeXt unfreeze) 실패 (14:00)

- gnorm 11-42 throughout, loss never <1.5 (early start at 2.5+)
- eval 7/27 all FAIL, dist 19-21mm
- lr 1e-6 도 너무 높음 OR ConvNeXt full unfreeze 우리 데이터 size에 over-parameterized
- v5b.2 retry 보류 — v5a (frozen SigLIP2 + aux boost) 결과 우선

### 11.21 v5a.2 launch (14:00) — aux boost stress test

GPU 2 free → v5a.2 시작:
- weight 1.0 → **2.0**, max_boost 50 → **100** (사용자 "올릴수록 좋아짐" stress)
- 나머지 v5a 동일 (lr 5e-6, 1500 step, champion v3 base)
- GPU 1: v5a 3-ckpt eval 진행중 (ckpt 1500 ep 1/27)
- GPU 2: v5a.2 train 시작

만약 v5a > baseline AND v5a.2 > v5a → "올릴수록 좋아짐" confirmed, sweep aux weight 더 올려서 한계점 찾기 (v5a.3 weight 4.0?)

### 11.22 사용자 feedback: ckpt eval sparse (14:00)

"Eval을 그렇게 많이 할 필요없긴하거든? 지금 그런식이면 final이랑 중반 초반만 봐도 될 것 같기도"

→ memory `feedback_eval_workflow.md` 보강. watcher 작성 시:
- final 우선 (1500 step)
- final이 좋으면 mid 추가 (500 또는 1000)
- early 일반적으로 안 봄

**즉시 적용**:
- v5a.2 watcher: final 1500만 eval (500/1000 skip)
- v5a 현재 eval은 1500 진행중 → 끝나면 결과 보고 500/1000 cancel 검토

### 11.23 v5a (aux boost 1.0/50) 결과 + outlier 분석 (14:15)

**Raw distribution comparison (27 ep)**:

```
                       n  mean  median  p25   p10  best5  <2mm  <1mm
v3_baseline           27  5.42  5.33   2.96  2.41  2.26    1     0
v5a (boost 1.0/50)    27  5.44  5.10   3.17  2.82  2.56    1     0
                                ↑                          
                            median 0.23 ↑       best 0.30 ↓
```

**해석** (user outlier insight 적용):
- Mean 동등 → 평균만 보면 "별 효과 없음"
- Median 0.23mm 개선 (v5a 살짝 ↑)
- **하지만 best 10/5 cells 모두 후퇴** (best5 2.26→2.56) — fine alignment 손해!
- 결론: aux_boost는 **모든 cell을 5mm 영역으로 수렴**시키는 효과. outlier는 약간 개선, best cells는 후퇴 → fine precision 측면 net 손해

→ 사용자 가설 "올릴수록 좋아짐"이 이 config에서 confirm 안 됨. v5a.2 (2.0/100) 더 강화하면 더 안 좋을 가능성 큼.

**analyze_trajectory.py 보강**: 분포 metric 추가 (p25, p10, best5_mean, n_under_2mm/1mm).

### 11.24 v5a.2 (w=2.0, b=100) 결과 + aux_boost saturation (14:40)

```
                       n  mean  med   p25   p10  best5 <2 <1
v3_baseline           27  5.42  5.33  2.96  2.41  2.26  1  0
v5a (w=1.0, b=50)     27  5.44  5.10  3.17  2.82  2.56  1  0
v5a.2 (w=2.0, b=100)  27  5.46  5.05  3.15  2.78  2.59  1  0  ← v5a와 거의 동일
```

aux_boost weight 2x 차이로도 결과 동일 → **saturation**. 더 강화 의미 없음. axis 종결.

### 11.25 사용자 결정 (14:45)

- **GPU당 2 eval 동시 가능** (`feedback_gpu_concurrency` 신규) — 시간 절약 시 활용
- **Compact 준비**: 모든 v1~v5a.2 결과 + 사용자 dead-end 결정 + GPU concurrency를 memory에 강하게 저장

### 11.26 현재 진행 (compact 전)

**GPU 1**: v3 baseline + diff_steps=50 redo (Stage 1 sweep 결과 stale npz 오염 의심, 깨끗하게 재확인). ep ~5/27.

**다음 axes** (post-compact 진행 후보):
1. Action precision 분석 (diffusion noise floor)
2. Near-goal data 재생성 (HOLD step ↑, 0~5mm trajectory 보강)
3. CNN partial unfreeze v5b.2 (last 2 stages만, lr 1e-7)
4. Hierarchical fine-policy (별도 small near-goal model)

---

## 11.27 Action Variance Diagnostic (2026-05-20 post-compact)

**가설**: 5mm 천장 = diffusion sampling noise floor (action precision 한계)
**방법**: v3 ckpt1000 고정, 같은 obs에서 30 samples (×3 diff_steps × 3 phantom offsets) → per-dim std 측정

**Output**: `logs/action_variance_v3_ckpt1000.json`

| diff_steps | state    | dx     | dy     | dz     | rx     | ry     | rz     |
|------------|----------|--------|--------|--------|--------|--------|--------|
| 10         | far_10mm | 0.059  | 0.069  | 0.097  | 0.087  | 0.122  | 0.076  |
| 10         | mid_5mm  | 0.057  | 0.067  | 0.088  | 0.087  | 0.117  | 0.059  |
| 10         | near_3mm | 0.078  | 0.073  | 0.106  | 0.089  | 0.097  | 0.090  |
| 25         | far_10mm | 0.076  | 0.073  | 0.092  | 0.126  | 0.122  | 0.090  |
| 25         | mid_5mm  | 0.072  | 0.082  | 0.091  | 0.108  | 0.092  | 0.093  |
| 25         | near_3mm | 0.062  | 0.062  | 0.101  | 0.090  | 0.146  | 0.065  |
| 50         | far_10mm | 0.079  | 0.067  | 0.086  | 0.102  | 0.132  | 0.072  |
| 50         | mid_5mm  | 0.060  | 0.066  | 0.093  | 0.116  | 0.079  | 0.087  |
| 50         | near_3mm | 0.095  | 0.060  | 0.112  | 0.079  | 0.111  | 0.081  |

**해석** (action_max_sim ≈ 1mm/step normalized to [-1,1]):
- per-step std 0.06~0.13 = **0.06~0.13mm 또는 0.1° uncertainty/step** (sub-mm)
- closed-loop control, RMS over 100 steps = √100 × 0.1 ≈ 1mm 누적 변동
- **diff_steps 10 = 25 = 50: 거의 동일** (rx/ry 약간 변동 있으나 평균 차이 없음)

**결론** ⚠️:
1. ❌ **Action precision은 5mm 천장의 원인 아님**. Sampling noise는 sub-mm.
2. ❌ Diffusion step 늘리는 것도 효과 없음 (10 = 50). Stage 1 sweep 결과 (모든 cell 5mm 천장)와 일치.
3. ✓ 진짜 천장: **BC 모델이 near-goal에 정확한 mean action을 학습 못함** (데이터 + 표현력 한계)
4. ✓ 다음 axis 정당화: **near-goal 데이터 보강 (HOLD=60)** + hierarchical model

**Caveat**: phantom offset 10mm시에도 actual dist 33.7mm로 측정됨 — robot이 home pose. fine-align 영역 (≤5mm) state는 trained model 추론 후의 동적 state. 진단은 "approach far region" variance만 측정. 향후 actual fine-align state에서 측정 필요시 reset에 robot pre-align 추가.

## 11.28 Near-goal Data Regeneration (2026-05-20)

**가설**: 천장은 BC near-goal 데이터 부족. Hold trajectory 2배 보강 → fine alignment frame 학습량 ↑.

**구성**:
- `Sim/Save_dataset_align_NEARGOAL.py` (Save_dataset_align_HARD_unified copy)
- HOLD_RECORD_STEPS: 30 → **60** (hold trajectory 2배)
- ALIGN_HOLD_STEPS: 10 → **20** (성공 인정 더 엄격 — 진짜 정렬된 trajectory만)
- ALIGN_THRESHOLD_M: 0.002 그대로 (1mm 시도하면 성공률 망함)

**Launch** (GPU 영향 없음, CPU 4 worker):
```bash
python -u Sim/run_parallel.py --script align_neargoal \
  --workers 4 --episodes 250 \
  --base-dir dataset/fine_align/NEARGOAL_hold60_v1 \
  --randomize-phantom-pos --no-side-camera --seed 42
```

**예상**: ~70분 (HARD 30step hold 기준 +10% 시간). 완료 후 champion finetune cotrain mix에 추가.


---

## 11.29 v3 diff50 clean retest (2026-05-20)

**가설**: Stage 1 sweep contamination 의심 (stale npz로 모든 cell 5mm 천장). diff_steps=50으로 깨끗하게 retest 시 천장 깨질까?

**구성**: v3 ckpt1000, fresh eval dir (이전 stale dir 삭제), 27-cell @ retreat=2, diff50/exec1

**결과** (vs diff10 baseline):
| | diff10 baseline | diff50 retest |
|---|---|---|
| SR | 88.9% | **70.4% (-18pp)** |
| close_5mm | 88.9% | **48.1% (-41pp!)** |
| close_2mm | 3.7% | 0% |
| close_1mm | 0% | 0% |
| mean_min | 5.42 | 5.59 |
| median_min | 5.33 | 5.11 |
| best5_mean | 2.26 | 2.37 |

**결론**:
1. ❌ Stage 1 sweep contamination 의혹은 무효. Stage 1 결과대로 diff_steps 변경은 도움 안 됨
2. ⚠️ **학습-inference diffusion step mismatch는 해로움**. 학습 diff10 + inference diff50 → 48% SR 폭락
3. 학습 distribution과 inference scheduler 일관성 유지가 중요
4. action variance diagnostic도 동일 의미: diffusion step 자체는 천장과 무관

## 11.30 v3 final ckpt eval (2026-05-20)

**가설**: champion ckpt1000이 sweet spot인가, training 더 가면 좋아지는가?

**구성**: v3 checkpoint_final (5000 step), 27-cell, diff10/exec1

**결과** (vs ckpt1000):
| | ckpt1000 (champion) | checkpoint_final (5000) |
|---|---|---|
| SR | 88.9% | 74.1% |
| close_5mm | 88.9% | 47.1% |
| close_2mm | 3.7% | 2.9% |
| close_1mm | 0% | 0% |
| mean_min | 5.42 | 5.89 |
| median_min | 5.33 | 5.21 |
| best5_mean | 2.26 | 2.18 |

**결론**:
- **ckpt1000이 sweet spot 확정**. 5000 step까지 가면 mean/SR/close_5 모두 약간 worse
- best5는 살짝 better (overfit 가능성) 하지만 일관성 떨어짐
- v3 학습 schedule (max 5000 step)에서 best ckpt = 1000 stand firm

## 11.31 NEARGOAL Dual Track datagen launch (2026-05-20)

**가설** (사용자 인사이트):
1. 5mm 이내 정렬/유지 데이터 부족 → Track A
2. 각도 교정 약함 → Track B (angle-only specialized)

**구성**:
- Track A: `Save_dataset_align_only` script, phantom ±12/±29/0/±7°, perturb 5mm XY/±5mm Z/5° angle, hold 60, 3000 ep
- Track B: 동일 phantom, perturb XY=0/Z=0/angle=15° (angle-only), hold 60, 1000 ep
- 동시 launch (20 worker), CPU only

**코드 변경**:
- `Sim/run_parallel.py`: `--perturb-xy-mm/-z-min-mm/-z-max-mm/-angle-deg` flag 추가
- `Sim/6_neargoal_dual_track.sh`: dual-track 동시 launch

**진행 상황**:
- Track A worker 0: ~30 sec/ep (perturb 복잡) → 100 ep × 10 worker ~5h
- Track B worker 0: ~10 sec/ep (angle만이라 빠름) → 100 ep × 10 worker ~1.5h
- System load: 60/96 cores (24 worker total: Track A 10 + Track B 10 + 기존 NEARGOAL_v1 4). 정상

**Next** (~1.5h 후):
- Track B 완료 → sanity check (HDF5 ≥800, perturb metadata 검증)
- Track A 진행 중 (~5h)
- 둘 다 완료 → champion v3 finetune cotrain mix (lr 2.5e-6, max 2000 step)

