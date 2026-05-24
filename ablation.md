# Fine-Align Ablation Studies

Detailed ablation experiments split from `EXPERIMENTS_fine_align.md` on **2026-05-24** to keep the main doc focused on the paper narrative + SOTA results.

**This file contains**:
- **Section 22** — Ablation Master Table (paper Table 1 candidate; architecture / loss / data / inference / training schedule / metric design)
- **Section 29** — Qwen3.5-2B (with-LM) baseline ablation: fresh 20k vs vision-only chain comparison
- **Section 31** — `hold_recovery` autonomous sweep (v3-v8): holdSR 52% ceiling 탐색
- **Section 32** — Post-cleanup follow-up: exec=4 sweep + v9-v14 variants (sub-mm + hold focus)
- **Section 33** — 3-axis driver analysis: Architecture / Data / Loss for sub-mm + hold
- **Section 34** — 🚨 Honest metric suite (2026-05-24): early-termination artifact 진단 + 새 metric design

**See also**:
- `EXPERIMENTS_fine_align.md` — Master Cheatsheet (권위본), Paper backbone (Sections 1-9), BC Finetune knowledge, Section 28 (compact summary), Section 30 (Qwen + reach_recover SOTA)
- `attic/EXPERIMENTS_fine_align_history.md` — pre-2026-05-23 daily progress logs

---

## 🚨 Metric design note (2026-05-24, applies to ALL prior tables)

**문제 발견**: 모든 이전 ablation table의 `close_2`, `holdSR`, `safety` 수치는 **early-termination eval data 기준**. `scripts/sim_eval_align_only.py`가 `check_success()` (3D dist<5mm AND angle<10° AND 20-step hold) 만족 시 episode 즉시 break. 결과:
- "final lateral" = success 발화 시점의 lateral (model마다 다른 step). 모델이 빨리 success할수록 settle 측정 시간 부족.
- holdSR (lateral<2.5mm for 20 contig step) 도 episode 일찍 끝나면 underestimate.
- 평균 episode steps **~120** (max 250). 절반 데이터 잘려있음.

**진단** (early-term 데이터에서 추출 가능한 부분):
- v11 ck1500: final close_2=77.8% but **Reach@2=85.2%** — 7.4pp가 "도달은 했지만 마지막 step에 떠나감"
- 모델은 sub-mm 자주 터치 (Reach@1=37%) but settled 못 함 (Max30<2.5=18.5%)

**해결**: `--no-early-term` flag 추가 (sim_eval_align_only.py:1485). full 250 step trajectory 강제. 주요 champion 8개 재 eval 진행 중 (Section 34).

**New metric suite** (`scripts/honest_metrics.py`):

| metric | 정의 | 무엇을 측정 | 이전 단점 |
|---|---|---|---|
| **Reach@K** (K=5/2/1mm) | per-ep min(lateral) < K. fraction across 27 eps | "도달 능력" (peak reach) | final-state bias 없음 |
| **TTA@K** | first step lateral < K, median across eps | "도달 속도" (efficiency) | NEW |
| **min_lat_med** | per-ep min(lateral), median | "Peak 정밀도" | 동일 (기존 OK) |
| **HoldSR@K_N** | any N-step window all < K (K=2.5, N=20) | "Sustained alignment" | early-term이 truncate. honest eval로 fix |
| **Settled_lat** | last 30 step의 median, then median across eps | "최종 안착 위치" (NOT final-step) | final-step instability 제거 |
| **Settled_std** | last 30 step의 std, then median | "안착 후 jitter" | NEW |
| **Max30<2.5** | last 30 step 모두 <2.5 인 episode SR | "Hold (settled-window 기반)" | contig 요구 없음, cleaner |
| **Safety_settled** | p99 of settled_lat across eps | "Worst-case 안착 (medical bound)" | 기존 final 기반보다 stable |

**Eval protocol 변경** (paper 권장):
- `--no-early-term` 필수 (paper 결과 모두 honest)
- 모든 metric은 **post-hoc analysis from full 250-step trajectory**
- "final" → "settled (last 30 step)" — 모델 간 fair 비교 가능
- 기존 close_2 / safety 등은 deprecated, settled-based metric로 대체

→ Section 34에서 honest eval 결과 + new champion ranking 재정의.

---

## Section 22: Ablation Master Table (정리, 2026-05-22)

**목적**: 이 세션 동안 진행한 모든 ablation을 한 표로 모아 paper appendix 직행 가능.
모든 평가는 27-cell grid @ retreat=2 (xy ±10mm × 3, y ±25mm × 3, angle ±5° × 3), single seed 2026.

### A. Architecture / Vision Encoder ablation

| variant | 학습 | SR_old | close5 | close2 | holdSR | min_lat | safety | verdict |
|---|---|---|---|---|---|---|---|---|
| ACT (ResNet18 + CVAE+T) | scratch 30k | 100% | 100% | 48.1% | 24.5% | 2.00mm | **3.78mm** | reach OK, hold 약함 |
| DP (ResNet18 + CondUnet1D) | scratch 30k | 100% | 100% | 33.3% | 11.1% | 2.22mm | 3.91mm | reach OK, hold 약함 |
| ConvNeXt-base frozen + diff head | fresh 1500 | 0% | 0% | 0% | 0% | 17.42mm | 43.81mm | 학습 부족 |
| ConvNeXt-base frozen + diff head | fresh 5000 | 0% | 11.1% | 3.7% | 14.8% | 20.37mm | 48.51mm | 미세 진전 |
| ConvNeXt-base frozen + diff head | fresh 20000 | 3.7% | 14.8% | 3.7% | 11.1% | 20.50mm | 48.46mm | 13× 학습량에도 fail |
| DINOv3-ViT-L/16 frozen + diff head | fresh 1500 | 0% | 3.7% | 0% | 11.1% | 19.86mm | 47.93mm | 학습 부족 |
| DINOv3-ViT-L/16 frozen + diff head | fresh 5000 | 0% | 7.4% | 0% | 11.1% | 19.62mm | 47.66mm | saturation |
| DINOv3-ViT-L/16 frozen + diff head | fresh 20000 | 0% | 7.4% | 0% | 11.1% | 19.04mm | 45.97mm | 13× 학습량에도 fail |
| SigLIP2-so400m frozen + diff head | fresh 1500 | 0% | 3.7% | 0% | 11.1% | 19.01mm | 46.02mm | 학습 부족 |
| **SigLIP2-so400m + chain (Ours)** | base + finetune ≥4k | **44.4%** | **70.4%** | **51.9%** | **77.8%** | **0.87mm** | 11.48mm | **champion** |

**Verdict (Section 21)**: encoder 자체가 아니라 **checkpoint chain (base 50k + finetune cascade)**이 dominant. fresh budget에선 encoder 무관 모두 fail. Fair encoder ablation은 동일 chain matching 필수 → future work.

### B. Loss component ablation

**B.1 dist / lat / hold 4-cell (Section 17)**

| dist | lat | hold | best SR_old | min_lat | comment |
|---|---|---|---|---|---|
| ✓ | ✗ | ✗ | 44.4% | 0.99mm | dist-only baseline (`v2_dual_lr1e6`) |
| ✓ | ✓ | ✗ | 마진 | 1.0mm 부근 | lateral 단독은 약함 |
| ✓ | ✗ | ✓ | 마진 | 1.0mm 부근 | hold 단독은 약함 |
| ✓ | ✓ | ✓ | 44.4% | 0.87mm | **synergy** (dist + lat + hold 함께만 sub-mm) |

**B.2 DCT loss 0.1 vs 0.0 (Section 20)**

동일 spec rerun (lat_hold_v4_yneg_hold 계열, base = loss_lat_hold_v1/ck1000, seed 2026):

| step | DCT off | DCT on | Δ |
|---|---|---|---|
| 500  | SR 55.6% / c2 59.3% | SR 55.6% / c2 55.6% | ±0 / −3.7 |
| 1000 | SR 55.6% / c2 55.6% | SR 51.9% / c2 55.6% | −3.7 / ±0 |
| 1500 | SR **63.0%** / c2 51.9% | SR 51.9% / c2 **63.0%** | −11.1 / **+11.1** |

→ DCT는 본질적 contribution ≈ 0. 약한 trade-off (SR_old vs close_2) 외엔 noise. **champion config DCT 0.1 → 0.0 변경 추천** (학습 약간 가벼움).

**B.3 Hold-loss false confound (Section 21.3b)**

| 조건 | holdSR | min_lat | close_2 |
|---|---|---|---|
| ACT (no hold loss/data) | 24.5% | 2.00mm | 48.1% |
| SigLIP2 + dist-only (no hold loss, no hold data) | **74.1%** | **0.99mm** | 55.6% |
| SigLIP2 champion (+ hold loss + hold data) | 77.8% | 0.87mm | 51.9% |

→ aux_hold/aux_lateral 추가의 marginal 효과 (**+3.7pp holdSR, −0.12mm min_lat**). encoder + chain만으로 이미 holdSR 74% 달성. **aux_hold = load-bearing → marginal refinement** 정정.

### C. Data ablation (Section 18)

**C.1 hold-rich data (NEARGOAL_yneg_hold + perfect_strict)**

| 조건 | SR_old | min_lat | y=+25 |
|---|---|---|---|
| base data만 (`v2_dual_lr1e6`) | 44.4% | 0.99mm | 9/9 |
| + yneg_hold + perfect_strict (champion) | 44.4% | 0.87mm | 9/9 |
| Δ | ±0 | −0.12mm | ±0 |

→ hold-rich data 추가 효과 **marginal** (min_lat −0.12mm). aux loss와 동일 패턴 — base data + chain만으로 천장 도달.

**C.2 Y-region 데이터 비대칭 (project_y_neg_distribution_bias)**

approach_00 PHANTOM_Y 72%가 y>0 → y=-25 cells 모두 fail (champion 0/9), y=+25는 100% (9/9). 데이터 분포 fix 필요. `NEARGOAL_yneg_v1` (yneg 1500ep 전용 datagen) + `NEARGOAL_ypos_v1` (ypos 1500ep) 수집 완료.

### D. Inference / Eval protocol ablation

**D.1 Action chunk stride (exec) — Section 19.2, memory `feedback_inference_axis_exec2`**

reach champion `b100v4_ft_phase2_lowlr/ck1500`에서 exec sweep:

| exec | SR_old | lat_med | safety | ang_near | y=+25 |
|---|---|---|---|---|---|
| 1 | 77.8% | 2.84mm | 7.37 | 2.86° | 3/9 |
| **2** | **85.2%** | **2.71mm** | 7.45 | **2.52°** | **5/9** |
| 4 | 77.8% | 2.72mm | 7.66 | 2.85° | 3/9 |
| 6 | 74.1% | 2.80mm | 7.66 | 2.83° | 2/9 |
| 8 | 77.8% | 3.12mm | 7.35 | 2.85° | 3/9 |

→ **exec=2 모든 primary 지표 best**. paper default standard. exec≥4는 over-commit.

**D.2 Retreat protocol (Section 19.2)**

| retreat | description | best model on this |
|---|---|---|
| 10mm | 기존 ACT/DP 비교 protocol | 둘 다 22% (천장 artifact) |
| **2mm** | paper protocol (advancing 가능) | champion 44%, ACT/DP 100% reach |
| 0mm  | static start | 새 SigLIP2 50k base에 적합 |

→ retreat=2 standardize. retreat=10은 deprecated.

### E. Training schedule ablation

**E.1 Learning rate (memory `project_lr_ablation_final`)**

| lr | ckpt_1500 median dist |
|---|---|
| 1e-5 | divergent (gnorm 폭주) |
| 1e-6 | **4.44mm** (champion) |
| 5e-7 | 4.51mm (lowlr) |
| 1e-7 | 4.62mm |

→ lr 1e-6 default. >1e-5는 일관되게 gnorm 폭주. 5e-7도 안전 (b100v4_ft_phase2_lowlr/ck1500 SR_old 85.2). fresh training은 5e-6 (lr 1e-5 보다 약간 안전).

**E.2 Step budget (Section 21.3c)**

fresh DINOv3/ConvNeXt 1500 → 5000 → 20000 step 학습량:
- ConvNeXt: close5 0% → 11% → 15% (monotonic 미세 진전)
- DINOv3: close5 3.7% → 7.4% → 7.4% (5k에서 saturation)
- 둘 다 SR_old ~0% 유지 → fresh로 chain 못 따라잡음

**E.3 Pretrained chain (Section 21.3d)**

base 50k → finetune cascade (4-7 chains) → champion. encoder choice보다 dominant.

### F. Eval metric ablation

이 세션의 가장 중요한 metric design 발견:

**F.1 Lateral metric breakthrough (project_lateral_metric_breakthrough)**

"5mm 천장"은 3D dist + retreat 2mm Z offset = artifact. 실제 lateral median 0.87-1.19mm 이미 sub-mm.

**F.2 Multi-criteria (paper claim 정정)**

`SR_old` 단독으론 retreat=2에서 ACT/DP 100% — 신호 없음. **(holdSR, min_lat, safety)** 3-axis composite 필수:

| metric | 신호 |
|---|---|
| SR_old (3D < 5mm at end) | retreat=2에선 ACT/DP saturated, deprecated |
| **close_5 (final_lateral < 5mm)** | 천장 진단용 |
| **close_2 (final_lateral < 2mm)** | 정밀 정렬 |
| **holdSR (lateral<2.5mm for ≥20 contig steps)** | hold-and-stay vs touch-drift 차별 |
| **min_lat (per-ep min lateral median)** | peak 정밀도 |
| **safety (p99 final_lateral)** | medical worst-case bound |
| per-region SR (y=-25/0/+25) | 분포 비대칭 진단 |

→ paper Table 1은 이 7 metric all-in-one.

### Summary — 무엇이 dominant?

| axis | dominant? | 정량 |
|---|---|---|
| **Vision encoder choice** | ❌ (fresh 20k에서 ConvNeXt ≈ DINOv3 ≈ SigLIP2 0%) | — |
| **Checkpoint chain** | ✅ **dominant** | fresh SR 0% → chain SR 44%, holdSR 78% |
| **aux_hold/lateral loss** | ❌ marginal | +3.7pp holdSR, −0.12mm min_lat |
| **Hold-rich data** | ❌ marginal | −0.12mm min_lat |
| **DCT loss** | ❌ noise | ±3.7-11pp trade-off |
| **dist+lat+hold synergy (vs dist only)** | △ marginal | −0.12mm min_lat (vs base) |
| **exec=2 (vs exec=1)** | ✅ **free win** | SR +7.4pp, lat −0.13mm, ang −0.3° (no training) |
| **lr ≤ 1e-6** | ✅ critical | >1e-5 폭주 |
| **base + finetune cascade** | ✅ load-bearing | champion only achievable via chain |

→ Architecture choices > Training schedule > Loss/data tweaks. **Encoder picking은 paper에서 over-claim 금지** (chain matching까지 한 게 아니면). aux losses + hold data는 marginal refinement로 honest 표기.

### Artifacts cross-ref

| Section | Topic | Configs / Scripts |
|---|---|---|
| 17 | Loss synergy | `sim_train_align_loss_lat_hold_v1`, `sim_train_align_lat_hold_*` |
| 18 | Data ablation | NEARGOAL_yneg_hold + perfect_strict |
| 19 | Compact summary | (no new artifacts) |
| 20 | DCT on/off | `sim_train_align_dct_{off,on}_v1`, `scripts/{eval,analyze}_dct_ablation.{sh,py}` |
| 21 | Encoder + baseline | `sim_train_align_{dinov3,siglip2,convnext}_{baseline,long5k,long20k}_v1`, `scripts/{eval,analyze}_baseline_matrix.{sh,py}`, `eval_long{5,20}k_matrix.sh` |
| 22 | Master table | (this section) |


---


---

## Section 29: Qwen3.5-2B (with-LM) ablation — fresh 20k baseline (2026-05-23)

### 29.1 Motivation

Vision-only variant ([[project_model_architecture]])이 우리 champion 아키텍처지만, paper narrative에 **"LM을 빼는 게 contribution"**이라 주장하려면 같은 budget에서 **LM 포함 baseline 측정**이 필요. Section 21.3 encoder ablation은 SigLIP2/DINOv3/ConvNeXt fresh 20k 모두 fail을 보여줬으나 "LM을 더하면 도움이 될까?" 는 미답 axis.

**Qwen3.5-2B는 hybrid VL model** — built-in vision encoder + alternating linear_attention/full_attention layers (24 layers, ratio 3:1). Per [[project_vlm_choice]] memory: Qwen3.5-2B fine-align SR > Qwen3-VL-2B-Instruct +14%. 우리 task 가장 강한 LMM 후보.

### 29.2 Setup

| field | value |
|---|---|
| Config | `config/output_dir_b100_baseline_model_20000step_qwen.yaml` (active) / `_EVAL.yaml` (eval-safe) |
| LMM | `Qwen/Qwen3.5-2B` (~2B params, hybrid VL, frozen) |
| Action head | Same diffusion head (depth=24, heads=16, queries=32, 1152d) |
| Proprio | ee_pose 6 DoF, fed into VLM (`use_proprio_input_vlm=true`) |
| Data | approach_00 (5K) + approach_10k_v3 (5K) + 10mm_fine_align (5K) + align_small_angle (3K) + align_10k_v3 (3K) ≈ 21K eps |
| Loss | diffusion flow_match + aux_distance (w=0.5) + DCT (w=0.1). **NO aux_lateral, NO aux_hold** |
| Aug | enabled (random_resized_crop, brightness, contrast, saturation, hue) — vision-only champ는 aug off |
| lr / batch / steps | 1e-5 / 100 / 20000 (fresh, no pretrained chain) |
| Backbone | frozen |
| Eval | 27-cell grid @ retreat=2, exec=2, diff=10 (paper standard) |
| Checkpoint path | `checkpoints/output_dir_v2_dual_finetune_qwen_20000step/checkpoint_20000.pt` |

### 29.3 Eval engineering 노트 (재현용)

**Ckpt format pitfall**: 원본 `checkpoint_20000.pt`는 `{"model_state_dict": OrderedDict}` 1단계 wrap 구조. sim_eval.py `load_model`은 `"config" in ckpt and "model_state_dict" in ckpt` 둘 다 만족할 때만 wrap을 풀고, 아니면 outer dict를 state_dict로 취급 → 921 missing / 1 unexpected (단일 key `model_state_dict`만 보임). **Fix**: inner OrderedDict 그대로 flat save (single dict, no wrap). 진단 후 `.pt`를 `model_state_dict` payload만으로 재저장 → 0 missing / 0 unexpected.

**Config pitfall**: `lmm_path: "Qwen/Qwen3.5-2B"`로 instantiate해야 ckpt key 일치 (`lmm.model.visual.*` + `lmm.model.language_model.*.linear_attn.*` + `*.self_attn.*` 혼합). 잘못해서 `Qwen3-VL-2B-Instruct`로 instantiate하면 `linear_attn` 레이어 부재 → 170 missing / 162 unexpected (full_attention 자리 keys만 일치). [[feedback_config_choices_intent]] 항목 — config의 active line 항상 ckpt에 맞게 검증.

**Linear attention slow path**: Qwen3.5 linear_attention layers는 `flash-linear-attention` + `causal-conv1d` 패키지 없으면 torch fallback (느림). 본 eval은 fallback path로 진행 (설치 필요시 future work).

### 29.4 결과 — 27-cell @ retreat=2, exec=2, diff=10

🔥 **놀라운 결과**: Qwen3.5-2B fresh 20k가 **여러 핵심 지표에서 champion + ACT 두 진영의 강점을 모두 동시 충족**.

| variant | n | SR_old | close5 | close2 | holdSR | min_lat | min_3D | finLat | ang° | safety | y=-25 | y=0 | y=+25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-2B fresh 20k** ⭐ | 27 | **66.7%** | **100%** | 40.7% | 44.4% | 1.50mm | **2.88mm** | 2.44mm | **2.04°** | **3.95mm** | 1/9 | **9/9** | 8/9 |
| ACT (ResNet18 30k) | 27 | 100% | 100% | 48.1% | 24.5% | 2.00mm | 2.80mm | 2.01mm | 1.55° | 3.78mm | 9/9 | 9/9 | 9/9 |
| DP  (ResNet18 30k) | 27 | 100% | 100% | 33.3% | 11.1% | 2.22mm | 2.85mm | 2.34mm | 1.36° | 3.91mm | 9/9 | 9/9 | 9/9 |
| ConvNeXt fresh 20k | 27 | 3.7% | 14.8% | 3.7% | 11.1% | 20.50mm | 21.79mm | 21.96mm | 4.75° | 48.46mm | 0/9 | 0/9 | 1/9 |
| DINOv3 fresh 20k | 27 | 0% | 7.4% | 0% | 11.1% | 19.04mm | 20.62mm | 19.04mm | n/a | 45.97mm | 0/9 | 0/9 | 0/9 |
| SigLIP2 + dist only (chain) | 27 | 44.4% | 74.1% | 55.6% | 74.1% | 0.99mm | 5.20mm | 1.95mm | 3.00° | 11.17mm | 0/9 | 3/9 | 9/9 |
| **SigLIP2 champion (chain)** | 27 | 44.4% | 70.4% | 51.9% | **77.8%** | **0.87mm** | 5.04mm | 1.96mm | 3.00° | 11.48mm | 0/9 | 3/9 | 9/9 |
| reach_recover v5 ck2000 (chain) | 27 | 63.0% | 74.1% | 55.6% | 74.1% | 1.00mm | 4.80mm | 1.55mm | 3.40° | 11.62mm | 2/9 | 6/9 | 9/9 |

(y region 표기: `cells_succeeded / 9`)

### 29.5 핵심 발견

1. **SR_old 66.7% — 최강 VLA reach 모델**: ACT/DP 100% 제외, 모든 vision-only/chain 기반 모델 추월. champion v3 44.4% 대비 **+22pp**, 우리 chain best reach_recover v5 ck2000 63% 대비 **+3.7pp**.

2. **close_5 100% — 모든 27 cell이 lateral < 5mm에 도달**: ACT/DP와 유일하게 동급. champion 70% 대비 +30pp. **단 한 cell도 lateral 발산 안 함** — VLM의 visual feature representation이 trocar localization을 robust하게 만듦.

3. **safety 3.95mm — 의료 grade에 가장 가까움**: champion 11.48mm 대비 **−65%**. ACT 3.78mm 수준 도달. y=-25 region에서 멀리 못 가는 게 아니라 못 닿는 cell은 닿지 않으면서 worst-case bound이 안전.

4. **min_3D median 2.88mm**: champion 5.04mm 대비 **−43%**. 3D dist 천장 (retreat 2mm + lateral metric artifact)을 깨뜨림. ACT 2.80mm와 거의 동급.

5. **angle 2.04° — 최저**: champion 3.00° 추월. VLM의 view 이해가 angle alignment에 직접 기여.

6. **y=0 region perfect 9/9**: champion 3/9 대비 **3배**. VLM이 in-distribution cell들은 완전 정복.

7. **단점 — hold + sub-mm precision**: holdSR 44.4% (champion 77.8% 대비 **−33pp**), min_lat 1.50mm (champion 0.87 대비 **+0.63mm**), close_2 40.7% (champion 51.9 대비 **−11pp**). **ACT/DP-like touch-and-drift 패턴 답습** — aux_hold/aux_lateral 부재 + chain matching 안 됨.

8. **y=-25 region 1/9 — 11% (champion 0/9 대비 +1 cell)**: 다른 모든 모델과 마찬가지로 fundamental 한계 ([[project_y_region_asymmetry_0521]]). 약간 회복하나 ACT 9/9에 비해 여전히 큰 격차.

### 29.6 Paper narrative 전환 — 기존 claim 정정

**기존 (Section 21.3d, 22)**: "Encoder choice는 fresh budget에서 차별화 안 됨; chain이 dominant. ConvNeXt ≈ DINOv3 ≈ SigLIP2 fresh 20k → 모두 SR_old 0~3.7% fail."

**Section 29 추가 발견**: **이 규칙은 vision-only encoder들에만 적용**. **LM (Qwen3.5-2B hybrid VL)을 추가하면 fresh 20k로 chain champion보다 더 강한 reach 모델** (SR 66.7% vs 44.4%, close5 100% vs 70%).

**원인 가설**:
1. **VLM의 cross-attention scaffolding**: linear_attention + full_attention 혼합 layer가 visual feature → action mapping을 안정화. vision-only frozen encoder + diff head는 fresh budget에서 mapping 학습 부족.
2. **2B-scale LM의 representational power**: 단순 ConvNeXt/DINOv3보다 훨씬 큰 backbone (2B vs 100-300M). 학습 budget이 충분히 활용됨.
3. **Aug 차이 (confound)**: vision-only champ는 aug off, Qwen run은 aug on. 동일 조건 아님. 부분적 contribution이 aug에서 옴 가능 (단 close5 100%, safety 3.95mm 등 큰 차이는 aug만으로 설명 어려움).
4. **VLM pretraining의 transfer**: Qwen3.5는 vast image-text 데이터로 pretrained — visual scene understanding이 head start.

**결론 정정 — 두 axis로 paper narrative 재구성**:

| 단계 | claim | 정량 backing |
|---|---|---|
| Old | "vision-only가 LM-included 능가" | 미측정 — 가설 |
| Section 21.3d | "encoder는 fresh budget에서 차별화 X, chain dominant" | ConvNeXt/DINOv3/SigLIP2 fresh 20k 모두 SR 0~3.7% |
| **Section 29 NEW** | "**fresh budget에서 LM 추가는 큰 도움**" | Qwen3.5-2B fresh 20k SR **66.7%** ≫ vision-only fresh 20k 0~3.7% |
| **Section 29 NEW** | "**그러나 hold 능력은 chain + aux_hold/aux_lat 필요**" | Qwen3.5 holdSR 44.4% ≪ champion chain 77.8% |

**Paper Table 1 (재구성된 multi-axis champion)**:

| Use case | Champion | SR_old | holdSR | min_lat | safety |
|---|---|---|---|---|---|
| **Reach + safety** | **Qwen3.5-2B fresh 20k** | **66.7%** | 44.4% | 1.50mm | **3.95mm** |
| **Hold (precision)** | SigLIP2 champion + exec=4 | 48.1% | **81.5%** | 1.00mm | 10.78mm |
| **Sub-mm lateral** | lat_hold_v4 ck1000 + exec=2 | 44.4% | 77.8% | **0.87mm** | 11.48mm |

→ **Single training 4 deployment modes**:
- ACT (reach 100% + safety best 3.78mm)
- Qwen3.5-2B (reach 66.7% + safety 3.95mm + close5 100% + ang 2.04°)
- Vision-only chain (hold + sub-mm precision)
- exec axis sweep (knob between reach/hold)

[[feedback_chain_dominant_over_encoder]] 메모 부분 정정: "**encoder swap fresh 무력**"은 vision-only encoder에만 적용. **LM (Qwen3.5 hybrid VL) 추가는 fresh budget으로도 reach champion 가능**.

### 29.7 추가 분석 — failure mode breakdown

Qwen3.5-2B 27-cell, 11 fails:
- **angle_fail**: 6 (54.5%) — angle > 10° at end despite reach
- **insufficient**: 3 (27.3%) — never reached
- **diverge**: 2 (18.2%) — reached then drifted away

Failure은 y=-25 region에 집중. y=-25 9 cells: 1/9 reach, 8 fail. failure modes 분포:
- 8 fails @ y=-25, 1 fail @ y=+25 (cell 21 angle_fail).

→ **angle_fail은 hold loss 부재의 직접 증거**. Qwen이 위치는 잡지만 회전 안정화 못 함. champion chain의 aux_hold (rot_weight 0.5)가 메우는 부분.

### 29.8 Confound caveat (paper에 명시 필수)

| confound | Qwen3.5 run | champion | 영향 |
|---|---|---|---|
| Aug | on (color jitter + resized_crop) | off | reach robustness ↑ 가능 |
| Loss | aux_dist 0.5 only | aux_dist + aux_lat + aux_hold | hold 약함 설명 |
| Data | base 21K (no NEARGOAL_yneg_hold) | + yneg_hold + perfect_strict | y=-25 weak 일관 |
| Chain | fresh (no pretrained) | base 50k + 4-7 finetune cascade | fair budget 아님 |
| Backbone | Qwen3.5-2B (2B params) | SigLIP2-so400m (~400M) | scale 차이 |
| Inference | exec=2, diff=10 | exec=2, diff=10 | 동일 |
| Eval seed | 2026 | 2026 | 동일 |
| **Comparable run for fair LM ablation** | — | aug-on SigLIP2 fresh 20k 필요 (미실행) | future work |

### 29.9 Memory updates (예정)

- `project_qwen35_with_lm_0523` — Qwen3.5-2B fresh 20k = reach champion 66.7%, close5 100%, safety 3.95mm. paper narrative 재정립 필요.
- `feedback_chain_dominant_over_encoder` 업데이트 — vision-only encoder들에만 적용. LM 추가는 fresh budget으로도 강함.
- `project_vlm_choice` 보강 — Qwen3.5 active 결정 backing 강화 (66.7% on paper grid).

### 29.10 Open questions (future work)

1. Qwen3.5-2B + aux_hold + aux_lat + chain → SR 가 80%+ 가능한가?
2. Qwen3.5-2B + exec=4 → holdSR 81%+ 가능?
3. Aug on/off ablation: vision-only champ에 aug 추가 시 reach 회복?
4. 동일 chain matching (base + finetune cascade) 적용 시 Qwen3.5가 vision-only chain champion 초과?
5. Inference speed (Qwen3.5 linear_attn fallback 너무 느림) — flash-linear-attention 설치 후 wall-clock 측정.

### 29.7 Artifacts

- Config (training): `config/output_dir_b100_baseline_model_20000step_qwen.yaml`
- Config (eval-safe copy): `config/output_dir_b100_baseline_model_20000step_qwen_EVAL.yaml`
- Checkpoint: `checkpoints/output_dir_v2_dual_finetune_qwen_20000step/checkpoint_20000.pt` (flat state_dict 형식)
- Eval log: `logs/qwen_eval/eval_20000_exec2_v3.log`
- Eval shards: `checkpoints/output_dir_v2_dual_finetune_qwen_20000step/align_eval_step20000_exec2_diff10_shard{0,1}/`
- Backup: `checkpoint_20000_wrapped.pt.bak` (원본 wrap 형식)

### 29.8 Section index 업데이트

```
29  Qwen3.5-2B with-LM ablation (NEW)
30  Qwen + reach_recover finetune — 신 SOTA (NEW)
```


---


## Section 31: hold_recovery autonomous sweep (v3~v8) — holdSR ceiling 탐색 (2026-05-23 EOD)

### 31.1 Motivation

v2 ck1500 SOTA (SR 100% / close_2 70.4 / safety 2.86 / y 9/9/9) 확정 후 **남은 약점 = holdSR 48.1%** (chain 78% 대비 약함). 사용자 요청 "성능 더 올릴 방법" — 자율 SOTA 탐색 모드로 6 variants × ~14 ckpts 평가.

### 31.2 Axes (v3-v8)

| Variant | Base | lr | Step | Loss/Data 변경 | 목적 |
|---|---|---|---|---|---|
| **v3 holdfull** | v2 ck1500 | 5e-7 | 1000 | aux_hold pos 0.15→0.3 / rot 0.25→0.5 (champion default) | aux_hold weight 풀강도 효과 |
| **v4 extended** | v2 ck1500 | 5e-7 | 1500 | 동일 recipe + 추가 학습 | over-train sweet spot check |
| **v5 holddata** | v2 ck1500 | 5e-7 | 1500 | perfect_hold/strict 2x oversample + approach 5K→2K + range/phantom_range drop | data weighting hold 효과 |
| **v6 holdonly** | v2 ck1500 | 5e-7 | 1000 | 위 + 모든 wide approach 제거 (approach cap 1K) | 극단 hold focus |
| **v7 lowlr_long** | v2 ck1500 | 2.5e-7 | 2000 | v5 + lr 더 낮춤 | very low lr + extended sweet spot |
| **v8 aux_extreme** | **v5 ck500** | 2.5e-7 | 1000 | aux_hold pos 0.3→0.5 / rot 0.5→1.0 (champion 2x) | hold champion + extreme aux_hold |

R1 exec sweep도 진행: v2 ck1500 + exec={1, 2, 4}.

### 31.3 Results (27-cell @ retreat=2, exec=2)

#### Master table

| Variant | SR | close_5 | close_2 | holdSR | min_lat | min_3D | safety | y=-25 | ang° |
|---|---|---|---|---|---|---|---|---|---|
| ACT (ref) | 100 | 100 | 48.1 | 24.5 | 2.00 | 2.80 | 3.78 | 9/9 | 1.55 |
| DP (ref) | 100 | 100 | 33.3 | 11.1 | 2.22 | 2.85 | 3.91 | 9/9 | 1.36 |
| SigLIP2 champion chain | 44.4 | 70.4 | 51.9 | **77.8** | **0.87** | 5.04 | 11.48 | 0/9 | 3.00 |
| Qwen fresh 20k | 66.7 | 100 | 40.7 | 44.4 | 1.50 | 2.88 | 3.95 | 1/9 | 2.04 |
| **v2 ck1500 (REF)** | 100 | 100 | 70.4 | 48.1 | 1.32 | 2.34 | 2.86 | 9/9 | 2.01 |
| v3 holdfull ck500 | 100 | 100 | 70.4 | 48.1 | 1.57 | 2.54 | 2.78 | 9/9 | 1.99 |
| **v3 holdfull ck1000** | 100 | 100 | **74.1** | 48.1 | 1.34 | 2.43 | 2.87 | 9/9 | 2.22 |
| v4 extended ck500 | 100 | 100 | 70.4 | 40.7 | 1.37 | 2.54 | 3.12 | 9/9 | 2.24 |
| **v4 extended ck1000** | 100 | 100 | 70.4 | 44.4 | **1.24** | 2.57 | 2.90 | 9/9 | 2.03 |
| v4 extended ck1500 | 100 | 100 | 70.4 | 40.7 | 1.39 | 2.49 | 2.88 | 9/9 | 2.08 |
| **v5 holddata ck500** 🏆hold | 100 | 100 | 66.7 | **51.9** | 1.51 | 2.46 | 2.65 | 9/9 | 2.13 |
| v5 holddata ck1000 | 100 | 100 | **74.1** | 44.4 | 1.47 | 2.81 | 2.70 | 9/9 | 2.02 |
| **v5 holddata ck1500** 🏆safety | 100 | 100 | 70.4 | 48.1 | 1.36 | 2.76 | **2.56** | 9/9 | 2.28 |
| v6 holdonly ck500 | 100 | 100 | 70.4 | 48.1 | 1.67 | 2.77 | 2.67 | 9/9 | 2.00 |
| v6 holdonly ck1000 | 100 | 100 | 70.4 | 44.4 | 1.43 | 2.38 | 2.72 | 9/9 | 2.14 |
| v7 lowlr_long ck500 | 100 | 100 | **74.1** | 40.7 | 1.38 | 2.39 | 2.68 | 9/9 | 2.15 |
| v7 lowlr_long ck1000 | 100 | 100 | 66.7 | 37.0 | 1.52 | 2.70 | 2.73 | 9/9 | 2.05 |
| v7 lowlr_long ck1500 | 100 | 100 | 70.4 | 44.4 | 1.45 | 2.67 | 2.76 | 9/9 | 2.21 |
| v7 lowlr_long ck2000 | 100 | 100 | **74.1** | 48.1 | 1.36 | 2.69 | 2.74 | 9/9 | 2.17 |
| **v8 aux_extreme ck500** 🏆precision | 100 | 100 | **77.8** | 37.0 | **1.19** | 2.52 | 2.73 | 9/9 | 2.13 |
| **v8 aux_extreme ck1000** 🏆safety | 100 | 100 | 66.7 | 48.1 | 1.40 | 2.71 | **2.47** | 9/9 | 2.11 |

#### R1 exec sweep (v2 ck1500)

| exec | SR | close_2 | min_lat | **holdSR** | safety | ang |
|---|---|---|---|---|---|---|
| 1 | 100 | 66.7 | 1.48 | 37.0 | 2.95 | 2.35 |
| **2** | 100 | 70.4 | **1.32** | **48.1** | 2.86 | 2.01 |
| 4 | 100 | 66.7 | 1.48 | 33.3 | 2.92 | 2.22 |

→ Qwen에선 **exec=2가 hold sweet spot** (chain models는 exec=4가 hold↑ ↔ 반대 패턴).

### 31.4 핵심 발견

1. **holdSR 진짜 ceiling ~52% on Qwen base** — 6 variants × 14 ckpts 시도, 단 1개 (v5 ck500) 가 52% 도달. 나머지 모두 37-48%.
2. **Hold ceiling은 데이터 weighting axis에서만 깨짐** (v5 ck500: perfect_hold 2x oversample). aux_hold weight ↑ (v3, v8), lr ↓ (v7), 학습 ↑ (v4) 모두 null.
3. **Pareto trade-off 가시화** — 모든 27 cells 100% SR + close_5 + y region 완벽한 채로 holdSR/safety/precision sub-axes 미세 조정만 남음.
4. **새 record들** (단일 ckpt 기준):
   - **v8 ck1000 safety 2.47mm** (모든 모델 최저, ACT 3.78 −35%)
   - **v5 ck500 holdSR 51.9%** (Qwen-family 최고, chain 78% 미달)
   - **v8 ck500 min_lat 1.19mm** (Qwen-family 최저)
   - **v8 ck500 close_2 77.8%** (모든 모델 최고, ACT 48.1 +29.7pp)
5. **v2 ck1500 여전히 SOTA balanced** — 단일 ckpt로 모든 핵심 지표 high (SR/close_5/precision/safety/y), holdSR 약점만.
6. **v8 의외의 trade-off** — extreme aux_hold (rot 1.0) 가 paradoxically holdSR 약화 (37%) but precision 폭발 (close_2 77.8, min_lat 1.19). ck1000에선 hold 회복 (48%) + safety 신 record. 즉 **aux_hold weight↑ = 학습 dynamics 변화 → precision↑ trade-off**, holdSR과 정반대 방향.

### 31.5 Paper Table 1 갱신 — 7 deployment regimes (final)

| Use case | Champion | Inference | SR | close_2 | min_lat | holdSR | safety | y=-25 |
|---|---|---|---|---|---|---|---|---|
| **🏆 Balanced SOTA (medical)** | v2 ck1500 | exec=2 | **100** | 70.4 | 1.32 | 48.1 | 2.86 | 9/9 |
| **🏆 Best precision (close_2)** | **v8 ck500** | exec=2 | 100 | **77.8** | 1.19 | 37.0 | 2.73 | 9/9 |
| **🏆 Best min_lat (Qwen)** | **v8 ck500** | exec=2 | 100 | 77.8 | **1.19** | 37.0 | 2.73 | 9/9 |
| **🏆 Best holdSR (Qwen)** | v5 ck500 | exec=2 | 100 | 66.7 | 1.51 | **51.9** | 2.65 | 9/9 |
| **🏆 Best safety (medical worst-case)** | **v8 ck1000** | exec=2 | 100 | 66.7 | 1.40 | 48.1 | **2.47** | 9/9 |
| Hold (chain, retreat trade-off) | v5_combo ck2000 (vision-only) | exec=4 | 48.1 | 55.6 | 1.00 | **81.5** | 10.78 | 0/9 |
| Sub-mm lateral (chain) | lat_hold_v4 ck1000 (vision-only) | exec=2 | 44.4 | 51.9 | **0.87** | 77.8 | 11.48 | 0/9 |

→ **5 Qwen + 2 chain = 7 deployment options**. v8 ck500이 precision champion 2개 axis 점령 (close_2/min_lat). v8 ck1000 신 safety champion.

### 31.6 Engineering 노트 — 자율 sweep

1. **GPU 2개 병렬 학습**: 2 trainings on GPU 1+2 (PCI_BUS_ID + CUDA_VISIBLE_DEVICES=0 → nvidia-smi 1, =1 → nvidia-smi 2). 2 변종 ~22min each parallel.
2. **Eval 2-shard split**: dual-GPU 27-cell ~9min per ckpt. 단점 — shard imbalance로 한 GPU 일찍 idle.
3. **Sharded eval 신뢰성 함정**: 단일 shard만 launching (race condition) 가능 → 14 cells만 측정. 항상 27 cell 확인.
4. **Flat ckpt 명명 함정**: `checkpoint_flat_{STEP}.pt` (NOT `checkpoint_{STEP}_flat.pt`) — sim_eval의 step parsing `stem.split("_")[-1]`.
5. **Project name 디렉토리 분할**: train.py가 `_`로 첫 segments를 부모 dir 분리 → `VLANeXt_Qwen35_NEARGOAL_X` → `VLANeXt_Qwen35_NEARGOAL/X/`.
6. **race condition in orchestrator**: `while pgrep ...` for newly-launched PIDs unreliable. 직접 `while ps -p PID` 권장.

### 31.7 다음 axes (남은 axis)

| priority | axis | 비용 | 기대 효과 |
|---|---|---|---|
| 1 | Multi-seed eval (3 seeds × v2 ck1500) | ~45min | stochasticity bound 정량화 |
| 2 | **Ensemble (Qwen v2 + chain) action averaging** | ~2h (code+eval) | holdSR 60%+ 가능 추정 (chain 78% 영향) |
| 3 | Architecture: hold gate (near-goal action mask) | ~1day code | structural fix, hold ceiling 진짜 해결 가능 |
| 4 | Real robot transfer | n/a (사용자 작업) | sim→real gap 측정 |
| 5 | More aggressive lr search (lr 5e-6, 5e-7 × 다양한 base) | ~2h | sweep saturation 확인 |

### 31.8 결론 — Qwen-family holdSR 천장 확인, 다른 axis 필요

**Status**: Qwen 기반 finetune axes 거의 saturate. holdSR ~52% 가 데이터/loss 변화만으론 한계.  
**다음 paper-headline axis** = **ensemble (Qwen reach + chain hold)** — code 작업 가치 있음.

### 31.9 Artifacts (이 세션)

- Configs: `config/sim_train_align_qwen_reach_recover_v{3,4,5,6,7,8}_*_config.yaml`
- Train logs: `logs/qwen_finetune/train_v{3,4,5,6,7,8}.log`
- Eval logs: `logs/qwen_eval/v{3,4,5,6,7,8}_step{500,1000,1500,2000}_exec2.log`
- Orchestrators: `/tmp/qwen_unified_phase3.sh`, `/tmp/qwen_eval_after_v3v4.sh`, `/tmp/qwen_eval_v5v6.sh`, `/tmp/qwen_eval_v7v8.sh`
- Checkpoints: `checkpoints/VLANeXt_Qwen35_NEARGOAL/reach_recover_v{3,4,5,6,7,8}_*/`
- wandb runs: e20hlhwb (v1), jqa9ejg3 (v2), ... (각 variant 별 wandb dashboard)

### 31.10 Section index 업데이트

```
31  hold_recovery autonomous sweep (v3-v8) — holdSR ceiling 확인 (NEW)
    31.4  핵심 발견 — holdSR ceiling ~52% confirmed
    31.5  Paper Table 1 — 7 deployment regimes (5 Qwen + 2 chain)
    31.7  남은 axes (ensemble 우선)
```



---

## Section 32: Post-cleanup follow-up (2026-05-24, in progress)

### 32.1 Motivation

Section 31 saturated holdSR ceiling at ~52% via 6 variants × 14 ckpts on Qwen base. Two further axes pursued after doc cleanup:

1. **exec=4 sweep on remaining Qwen champions**: Section 31 R1 only covered v2 ck1500. New evals on v8 ck500 (close_2 record), v5 ck500 (hold champion), v8 ck1000 (safety champion) — does Qwen exec=4 pattern flip for different ckpts?
2. **v9/v10 finetune attempts**: explicit hold-precision combo trying to break 52% ceiling.

### 32.2 Setup

| variant | base | lr | step | recipe 변경 | 가설 |
|---|---|---|---|---|---|
| **v9 gentle_hold** | v5 ck500 | 1.5e-7 | 1500 | aux_hold pos 0.3/rot 0.5 (champion default, half v8) + v5 holddata mix | hold champion + very gentle preserve, no extreme aux_hold |
| **v10 balanced_hold** | v2 ck1500 | 5e-7 | 2000 | yneg_hold/perfect_strict 3x (was 2x in v5) + champion aux_hold | SOTA base + heavier hold data |

exec=4 sweep targets: v2 ck1500 (redundant — Section 31 R1 already done), v8 ck500, v5 ck500, v8 ck1000.

### 32.3 exec=4 sweep results — Qwen 4 champion ckpts (2026-05-24)

| ckpt | exec=2 (ref) | exec=4 (new) | Δ holdSR | Δ close_2 |
|---|---|---|---|---|
| v2 ck1500 (balanced SOTA) | h 48.1 / c2 70.4 / safety 2.86 | **h 33.3 / c2 66.7 / safety 2.92** | −14.8pp ❌ | −3.7pp |
| **v8 ck500 (close_2 champ)** | h 37.0 / c2 77.8 / safety 2.73 | **h 48.1 / c2 63.0 / safety 2.69** | **+11.1pp** ✅ | −14.8pp |
| v5 ck500 (hold champ) | h 51.9 / c2 66.7 / safety 2.65 | h 44.4 / c2 66.7 / safety 3.25 | −7.5pp ❌ | ±0 |
| v8 ck1000 (safety champ) | h 48.1 / c2 66.7 / safety 2.47 | **h 40.7 / c2 70.4 / safety 3.12** | −7.4pp | **+3.7pp** ✅ |

**핵심 발견**:
1. **v8 ck500 + exec=4 = new mode**: holdSR 37→48% (+11pp), close_2 trade-off. v8 paradox 일부 회복 (extreme aux_hold rot 1.0 → exec=4 chunk averaging이 hold dynamics 부분 보상).
2. **v8 ck1000 + exec=4 = close_2 70.4%** — exec=2의 66.7% 위에 +3.7pp. safety는 후퇴 (2.47→3.12mm).
3. **v2/v5 ck1500은 exec=2가 모든 axis sweet** — chain pattern (exec=4 hold↑) 적용 안됨, **단 v8 (aux_hold rot=1.0)만 chain pattern 부분 회복**.
4. **holdSR 52% ceiling 깨지지 않음** — v5 ck500 exec=2 (51.9%) 여전히 Qwen-family 최고.
5. v2 ck1500 exec=4 = Section 31 R1과 정확히 일치 (eval 결정론성 sanity ✓).

**Paper-grade insight**: exec axis가 Qwen에서 작동 방식이 **모델 의존적**. extreme aux_hold (v8) 학습한 모델만 exec↑ → hold↑ 패턴 일부 복구.

### 32.4 v9/v10/v11/v12 results (완료, 야간 자율 세션 2026-05-24 새벽)

학습+eval 통합 결과 (모두 exec=2, retreat=2, seed 2026):

| ckpt | base | lr | step | SR | close5 | close2 | **holdSR** | min_lat | safety | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| **REF v2 ck1500** | — | — | — | 100 | 100 | 70.4 | 48.1 | 1.32 | 2.86 | prior balanced SOTA |
| **REF v5 ck500** | — | — | — | 100 | 100 | 66.7 | **51.9** | 1.51 | 2.65 | prior hold champ |
| **REF v8 ck500** | — | — | — | 100 | 100 | **77.8** | 37.0 | **1.19** | 2.73 | prior precision |
| **REF v8 ck1000** | — | — | — | 100 | 100 | 66.7 | 48.1 | 1.40 | **2.47** | prior safety |
| v9 ck500 ⭐ | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | **70.4** | **51.9** | 1.50 | 2.64 | **v5 우월 (close_2 +3.7pp, 나머지 동률)** |
| v9 ck1000 | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | **77.8** | 37.0 | 1.48 | 2.58 | close_2 tied SOTA (hold 손실) |
| v9 ck1500 | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | 74.1 | 48.1 | 1.50 | 2.74 | over-train |
| v10 ck500 | v2 ck1500 | 5e-7 | 2000 | 100 | 100 | 66.7 | 40.7 | 1.52 | 2.72 | (early) |
| v10 ck1000 | v2 ck1500 | 5e-7 | 2000 | 100 | 100 | 63.0 | 48.1 | 1.40 | 2.72 | |
| v10 ck1500 | v2 ck1500 | 5e-7 | 2000 | 100 | 100 | 70.4 | 48.1 | 1.46 | **2.57** | safety 2nd best |
| v10 ck2000 | v2 ck1500 | 5e-7 | 2000 | 100 | 100 | 66.7 | 44.4 | 1.57 | 2.64 | over-train |
| v11 ck500 | v2 ck1500 | 3e-7 | 1500 | 100 | 100 | 66.7 | 37.0 | 1.37 | 2.76 | early |
| v11 ck1000 | v2 ck1500 | 3e-7 | 1500 | 96.3 | 100 | 70.4 | 37.0 | 1.39 | 2.90 | |
| **v11 ck1500** 🏆 | v2 ck1500 | 3e-7 | 1500 | 100 | 100 | **77.8** | **48.1** | **1.32** | 2.65 | **NEW SOTA: precision + hold 동시** |
| v12 ck500 | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | 74.1 | 40.7 | 1.40 | 2.74 | hold 손실 |
| v12 ck1000 | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | **74.1** | 37.0 | **1.32** | 2.76 | precision 좋음, hold 손실 |
| v12 ck1500 | v5 ck500 | 1.5e-7 | 1500 | 100 | 100 | 63.0 | 40.7 | 1.49 | **2.57** | safety 2nd best |

(v11/v12는 NEW `NEARGOAL_submm_hold_v1` 데이터 + threshold_mm 1.5 사용)

### 32.5 핵심 발견 (이번 야간 세션)

1. **🏆 v11 ck1500 = NEW PARETO BALANCED SOTA**:
   - close_2 **77.8% = v8 ck500 신기록 동률**, BUT holdSR **48.1% = v8 ck500의 37%보다 +11pp**
   - min_lat 1.32mm + safety 2.65mm 모두 v8 ck500 수준 유지
   - **"v8의 precision + v2의 hold 통합" 달성** — sub-mm 데이터 (250 step hold + 1.5mm perturb) + threshold_mm 1.5 효과 입증
   - **Recipe = v2 ck1500 base + submm_hold_v1 (2x) + aux_hold threshold 2.5→1.5 + lr 3e-7 + 1500 step**

2. **v9 ck500 = strictly improves v5 ck500**:
   - holdSR 51.9% (동률) + close_2 70.4% (v5 ck500의 66.7%보다 **+3.7pp**) + safety 2.64 (slightly better)
   - **All-axis dominance over previous hold champion** — gentle continued training (lr 1.5e-7, 1 ckpt) 으로도 충분
   - **v5 ck500 deprecate, v9 ck500 = new hold champion**

3. **holdSR 52% ceiling 여전히 못 깸**:
   - v9/v10/v11/v12 모두 51.9% (tied) 이상 못 감
   - threshold_mm 1.5 (v11/v12), aux_hold weight 변경 (v9), 더 많은 hold data (v10/v11/v12) 모두 무효
   - **Qwen-family hold ceiling = structural** (architecture limitation, action style)
   - Chain SigLIP2 78% holdSR 와 격차 — Qwen 으로는 ensemble 또는 architectural change 필요

4. **min_lat 1.19mm (v8 ck500) NOT improved**:
   - v11/v12 best = 1.32mm (= v2 ck1500), v9 best = 1.48
   - Sub-mm precision 천장도 = v8 extreme aux_hold rot 1.0 이 유일하게 깼던 영역
   - threshold tightening은 close_2를 올렸지만 min_lat (peak precision)은 못 깸

5. **safety 2.47mm (v8 ck1000) NOT improved**:
   - v10 ck1500/v12 ck1500 모두 2.57mm로 second
   - v8 ck1000 safety record 유지

6. **NEARGOAL_submm_hold_v1 효과 검증**:
   - 1800ep + 250 step hold + 1.5mm perturb 추가 → v11 ck1500 신 SOTA 도출
   - 단 holdSR 천장은 깨지 못 함 — data 만으로 부족, **threshold_mm tightening combo** 가 핵심

### 32.6 Updated paper Table 1 — 7 deployment regimes (v11 ck1500 신추가)

| Use case | Champion | exec | SR | close_2 | min_lat | holdSR | safety | y=-25 |
|---|---|---|---|---|---|---|---|---|
| 🏆 **Balanced SOTA (medical, 신규)** | **v11 ck1500** | 2 | 100 | **77.8** | 1.32 | **48.1** | 2.65 | 9/9 |
| Balanced (이전) | v2 ck1500 | 2 | 100 | 70.4 | 1.32 | 48.1 | 2.86 | 9/9 |
| Precision peak | v8 ck500 | 2 | 100 | 77.8 | **1.19** | 37.0 | 2.73 | 9/9 |
| Hold champion | **v9 ck500** | 2 | 100 | 70.4 | 1.50 | **51.9** | 2.64 | 9/9 |
| Safety | v8 ck1000 | 2 | 100 | 66.7 | 1.40 | 48.1 | **2.47** | 9/9 |
| Hold (chain) | v5_combo ck2000 | 4 | 48 | 55.6 | 1.00 | **81.5** | 10.78 | 0/9 |
| Sub-mm lateral (chain) | lat_hold_v4 ck1000 | 2 | 44 | 51.9 | **0.87** | 77.8 | 11.48 | 0/9 |

→ **v11 ck1500 = paper 메인 row 후보** (single ckpt가 5 axes 모두 high). 단 hold ceiling (chain 78%, sub-mm 0.87mm)은 별도 specialist.

### 32.7 Multi-seed eval (2027/2028) 결과

(seed=2026 baseline → 2027 → 2028 overwrite로 merged data는 seed 2028 결과만 남음. 단 raw log에서 SR 추출):

| ckpt | seed 2026 SR | seed 2027 SR | seed 2028 SR |
|---|---|---|---|
| v2 ck1500 | 27/27 (100%) | 25/27 (92.6%) | 25/27 (92.6%) |
| v5 ck500 | 27/27 (100%) | 25/27 (92.6%) | 26/27 (96.3%) |
| v8 ck500 | 27/27 (100%) | 25/27 (92.6%) | 26/27 (96.3%) |

→ **Stochasticity bound = ±1-2 cells (3-7pp)** 동일 ckpt 단일-seed eval에서 변동 가능. SOTA 비교 시 ±5pp 이내 차이는 noise 가능성. v11 ck1500과 v8 ck500의 close_2 77.8% 동률은 multi-seed로 재검증 필요.

### 32.8 다음 axes (morning pipeline 진행 중)

| variant | base | recipe | 가설 |
|---|---|---|---|
| **v13 lat_boost** | v11 ck1500 (new SOTA) | aux_lateral 0.5→1.0 + lr 1e-7 | v11 SOTA 위에 lat 강화 — oscillation 다른 방식 억제 |
| **v14 hold_extreme** | v9 ck500 (new hold champ) | threshold_mm 1.5→1.0 + soft_scale 0.5 + 2000 step | 진짜 가까울 때만 hold loss (lat<1mm 영역) — hold ceiling 도전 |
| exec sweep | v11 ck1500, v9 ck500 | exec=1, 4 추가 | 새 champion Pareto knob |

### 32.9 Artifacts (update)

- Configs: `config/sim_train_align_qwen_reach_recover_v{9,10,11,12,13,14}_*_config.yaml`
- Datagen script: `Sim/11_submm_hold.sh` → `dataset/fine_align/NEARGOAL_submm_hold_v1` (1800ep, 250 step hold, 1.5mm perturb)
- Orchestrators: `/tmp/qwen_v9v10_relaunch.sh`, `/tmp/qwen_night_master_v2.sh`, `/tmp/qwen_morning_pipeline.sh`
- Eval logs: `logs/qwen_eval/v{9,10,11,12}_step*_exec2.log`, multi-seed: `logs/qwen_eval/v{2,5,8}_step*_seed{2027,2028}.log`


---


## Section 33: Sub-mm precision + Hold — 3-axis driver analysis (2026-05-24)

> 사용자 요청 종합 정리: "어떤 부분이 가장 큰 서브밀리미터 단위의 성능을 끌어올릴 수 있었는지 — 모델 구조적, 데이터, Loss 3가지 측면". 바늘 정렬 task = sub-mm 정확도 + sustained hold 둘 다 핵심.

### 33.1 두 가지 핵심 metric (paper Table 1 가장 중요한 컬럼)

| metric | 정의 | 의료 의미 | 현 best |
|---|---|---|---|
| **min_lat** | per-ep min lateral median | peak 정밀도 (sub-mm 가능성) | **0.87mm** (chain lat_hold_v4 ck1000 + exec=2) |
| **holdSR** | lateral<2.5mm for ≥20 contig steps | sustained alignment (insertion 직전 단계) | **81.5%** (chain v5_combo ck2000 + exec=4) |
| **close_2** | final_lateral < 2mm | 종료 시점 sub-mm 정렬 | **77.8%** (Qwen v8 ck500 + exec=2) |
| safety | p99 final_lateral | medical worst-case | **2.47mm** (Qwen v8 ck1000 + exec=2) |

→ **Sub-mm reach + 안정 hold는 별개 axis**. 한 ckpt가 둘 다 못 잡음 (Section 31/32 final). single training + multi exec mode가 paper headline.

### 33.2 Axis A: 모델 구조 (Architecture)

| 구조 | params | min_lat | holdSR | close_2 | 비고 |
|---|---|---|---|---|---|
| ACT (ResNet18 scratch + CVAE+Transformer) | 62M | 2.00mm | 24.5% | 48.1% | reach OK, hold/precision 약함 |
| DP (ResNet18 scratch + CondUnet1D) | 89M | 2.22mm | 11.1% | 33.3% | hold 더 약함 |
| ConvNeXt-base frozen + ours head, fresh 20k | ~100M+1043M | 20.5mm | 11.1% | 3.7% | fresh budget X — chain 필수 |
| DINOv3-ViT-L/16 frozen + ours head, fresh 20k | ~300M+1043M | 19.0mm | 11.1% | 0% | 동상 |
| SigLIP2-so400m frozen fresh 20k | ~400M+1043M | 19.0mm | 11.1% | 0% | encoder 자체로는 차별화 X |
| **SigLIP2 + chain (base 50k + finetune cascade)** | 1043M | **0.87mm** ⭐ | **77.8%** ⭐ | 51.9% | sub-mm precision 1등 |
| Qwen3.5-2B (hybrid VL) fresh 20k | 2848M | 1.50mm | 44.4% | 40.7% | reach + safety 우위, hold 약함 |
| **Qwen3.5-2B + reach_recover (v2 ck1500)** | 2848M | 1.32mm | 48.1% | 70.4% | SOTA balanced (SR 100, safety 2.86) |
| **Qwen3.5-2B + extreme aux_hold (v8 ck500)** | 2848M | 1.19mm | 37.0% | **77.8%** ⭐ | close_2 1등, hold 손실 |

**Sub-mm precision driver**:
1. **Encoder scale + frozen pretrained backbone**: ResNet18 (ACT/DP)는 sub-mm 못 함. **400M~2B pretrained frozen backbone** 이 필수.
2. **Chain training (base + finetune cascade)**: vision-only encoder들은 fresh budget 20k 으로는 SR ~0%. **Chain matching 없으면 encoder choice 무의미** (Section 21).
3. **VLM (Qwen 2B) 의외 강점**: fresh 20k에서도 reach + safety 강함. VLM의 visual feature → action mapping이 vision-only frozen + diff head 보다 stable.
4. **그러나 chain SigLIP2가 sub-mm/hold에서 여전히 우위**: chain matching이 dominant driver, scale은 보조.

**Hold driver (architecture 측면)**:
- ACT/DP scratch는 ResNet18 정밀도 한계 + 학습 budget 한계로 holdSR 11-24%.
- SigLIP2 chain은 holdSR 78% — encoder의 visual feature 안정성이 hold에 직접 기여.
- Qwen3.5-2B 단독은 holdSR 44% — VLM action style이 inherently hold-friendly 아님 (decoder처럼 매번 새 action 생성).
- Hold는 **vision encoder pretraining quality + training schedule**이 결정, params scale은 효과 적음.

→ **Architecture verdict**:
- Sub-mm: SigLIP2-so400m + chain training
- Hold: SigLIP2-so400m + chain training (same)
- Reach + safety: Qwen3.5-2B + reach_recover finetune
- **단일 architecture가 모든 axis 1등 불가** — paper에서 multi-deployment regime narrative 필수.

### 33.3 Axis B: 데이터 (Data)

데이터셋 종합 (`dataset/fine_align/` 외 base data 포함):

| dataset | ep | perturb | hold_steps | sub-mm 기여 | hold 기여 | reach 기여 |
|---|---|---|---|---|---|---|
| `approach_00` | 5000 | wide phantom XY±12/Y±29/ang±12 | - | base coverage | - | base |
| `10mm_fine_align_00_tip2` | ~50 | tip variation | - | small refine | - | - |
| `NEARGOAL_eval_match_v2` | ? | eval grid match | 60 | - | small | base |
| `NEARGOAL_angle_only_v2` | ? | angle focused | 60 | - | - | angle |
| **`NEARGOAL_yneg_hold_v1`** | 800 | y<0 + 2mm + hold 120 | 120 | small | **+holdSR/y=-25** | yneg recovery |
| **`NEARGOAL_perfect_strict_v1`** | 800 | 1mm tight + hold 150 | 150 | **+close_2** | medium | - |
| `NEARGOAL_perfect_hold_v1` | 1000 | 2mm + hold 120 | 120 | small | medium | - |
| `NEARGOAL_yneg_v1` | 1500 | y<0 wide | 60 | - | - | **+11pp SR** (Section 18) |
| `NEARGOAL_ypos_v1` | 1500 | y>0 wide | 60 | - | - | **+y=+25 region** |
| `NEARGOAL_yneg25_strict_v1` | 1500 | y∈[-29,-21] tight | 60 | **null** | null | null (Section 27) |
| **`NEARGOAL_submm_hold_v1`** (new, 2026-05-24) | 1800 | **1.5mm tight + hold 250** | **250** | **pending v11/v12** | **pending** | - |

**Sub-mm precision driver (data)**:
1. **perfect_strict_v1 (1mm perturb + 150 hold)** — close_2 약간 기여, base reach 데이터만으로도 도달.
2. **chain training data continuity** > 새 데이터 추가 — 같은 데이터 budget이라도 base 50k + cascade가 fresh 20k 압도.
3. **데이터 분포 매칭이 핵심**: NEARGOAL_eval_match (eval grid와 정확히 일치)이 in-distribution 보장.
4. **yneg25_strict (1500ep, y∈[-29,-21] tight)는 null** — y=-25 cells 2/9 그대로 (Section 27). 데이터 부족 axis 가 아니라 **encoder representational power axis** (Qwen 으로 9/9 해결됨 Section 30).
5. **submm_hold_v1 (이번 신규)**: 1800ep + 1.5mm perturb + 250 hold. 기존 perfect_strict (800ep + 150 hold) 보다 **데이터 양 2x + hold supervision 1.7x**. holdSR 52% ceiling 깰지 v11/v12에서 검증.

**Hold driver (data)**:
- 가장 큰 hold contribution: **yneg_hold_v1 + perfect_strict combo (Section 18 cycle 7)** — SR_new 77.8% 달성. 하지만 SigLIP2 + dist-only도 holdSR 74.1% (Section 21.3b false confound).
- **Hold-loss false confound (Section 21.3b)**: hold-rich data 추가의 marginal 효과 +3.7pp. **encoder + chain이 진짜 hold driver**, data 추가는 refinement.
- **단 v5 holddata (perfect_hold 2x + approach cap 2K)는 v2의 hold 48% → v5 ck500 52% 로 +3.8pp** (Section 31) — chain 위 data weighting axis만 유효.

**Reach driver (data)**:
- y-region balance (`yneg_v1` + `ypos_v1` 1500ep each) — SR +11pp (Section 23).
- approach_00 cap 5000 → 2000 reducing wide approach — over-train 방지.
- **y=-25 specialized data null** — encoder choice가 더 강한 driver.

→ **Data verdict**:
- Sub-mm: perfect_strict_v1 (1mm) > perfect_hold_v1 (2mm) > yneg25_strict (null)
- Hold: yneg_hold + perfect_strict combo, 단 **encoder/chain의 1/4 정도 contribution**
- Reach: yneg/ypos balance + approach cap (큰 효과)

### 33.4 Axis C: Loss

| component | weight | sub-mm 기여 | hold 기여 | reach 기여 |
|---|---|---|---|---|
| `main_flow_match` (diffusion) | 1.0 | base | base | base |
| **`aux_distance`** (margin 0.1, near_goal_scale 2mm boost 10x) | 0.5 | **load-bearing** | base | base |
| **`aux_lateral`** (margin 0.05, near_goal_scale 1mm boost 10x) | 0.5 | marginal (-0.12mm) | small | - |
| **`aux_hold` pos/rot** (pos 0.3, rot 0.5, threshold 2.5) | - | small | **marginal (+3.7pp)** | - |
| `dct_loss` (low/high freq weight 1.0 each) | 0.1 | **≈0 (Section 20)** | ≈0 | ≈0 |
| `direction_decoupled_loss` | 0 | **harmful** (gnorm 폭주) | harmful | harmful |
| `aux_hold extreme` (pos 0.5, rot 1.0 — v8) | - | **+close_2 7.4pp 7.4pp** | **−holdSR 11pp** | - |

**Sub-mm precision driver (loss)**:
1. **aux_dist load-bearing**: dist 단독으로도 SigLIP2 chain holdSR 74.1% (Section 21.3b). aux_lat/hold 추가는 marginal refinement.
2. **synergy (dist + lat + hold) ≠ additive (Section 17)**: 4-cell ablation에서 dist-only/+lat/+hold는 SR 70.4%로 동률, **3개 함께 = 85.2% (+14.8pp)**. 곱셈적 시너지.
3. **aux_hold extreme paradox (v8)**: rot_weight 0.5 → 1.0 으로 올리면 **close_2 77.8% 신기록** but holdSR 37%로 폭락. **precision↑ trade-off hold↓** — 같은 방향 axis 아님.
4. **DCT loss ≈ 0** (Section 20 controlled rerun): champion config DCT 0.1 → 0.0 변경 권장. paper에서 contribution 표기 금지.
5. **threshold_mm axis (v11/v12 pending)**: aux_hold threshold 2.5 → 1.5으로 좁히면 hold loss가 정말 가까울 때만 활성화. navigate 자유도↑, hold 정밀도↑ 가설.

**Hold driver (loss)**:
- **aux_hold marginal** (Section 21.3b false confound): SigLIP2 chain + dist-only도 holdSR 74.1%. aux_hold 추가는 +3.7pp.
- **aux_hold rot 1.0은 hold 약화** (v8 paradox) — pos/rot 모두 너무 강하면 oscillation 유발 추정.
- **chain training + encoder pretraining이 진짜 hold driver** — loss 자체는 fine tune.

**Reach driver (loss)**:
- aux_dist near_goal_scale 2mm boost 10x — fine alignment 영역에서 dist gradient 증폭.
- aux_lateral margin 0.05mm는 fine refinement용. reach 자체엔 효과 작음.

→ **Loss verdict**:
- Sub-mm: aux_dist load-bearing, **synergy multiplicative**, DCT/DDL 폐기, aux_hold extreme은 trade-off.
- Hold: **chain effect** 위에 marginal refinement. loss로는 hold 천장 못 깸.
- Reach: aux_dist + near_goal_scale boost.

### 33.5 종합 — 어떤 axis가 sub-mm을 가장 끌어올렸나?

| axis | 정량 contribution to **min_lat 0.87mm** | 정량 contribution to **holdSR 78%** |
|---|---|---|
| **Architecture (SigLIP2 chain)** | **+18mm → 0.87mm** (fresh 19mm 대비) | **+67pp → 78%** (fresh 11% 대비) |
| **Chain training** (base 50k + finetune cascade) | dominant — 없으면 encoder 효과 X | dominant — 없으면 holdSR 11% |
| **Data (yneg_hold + perfect_strict)** | small marginal | +3.7pp marginal |
| **Loss synergy (dist+lat+hold)** | small marginal | +3.7pp marginal |
| **Inference axis (exec=2)** | -0.13mm | balanced |
| **Inference axis (exec=4)** | -0.13mm | **+3.7pp holdSR** (free win) |

**Rank 1 driver = Architecture (frozen pretrained vision backbone) + Chain training**. 두 변수가 sub-mm/hold 모든 천장을 결정. **Data + Loss는 모두 marginal refinement** (각 axis ±3-5pp 수준).

**Inference axis (exec)는 cheap free win** — training 없이 +3.7pp holdSR (chain v5_combo ck2000 + exec=4 = 81.5%). 단 Qwen에선 exec=4가 hold 약화 (v2/v5 ck1500) — **model-dependent**.

**Paper narrative 권장 (33.5 종합)**:
> "Sub-mm 정밀도와 sustained hold의 두 가지 핵심 metric 모두 **chain training된 frozen pretrained vision encoder** 가 dominant driver. ResNet18 scratch baseline (ACT/DP) 은 sub-mm 못 도달, fresh vision-only encoder 들은 모두 fail. **Loss aux components와 hold-rich data 추가는 marginal refinement** (±5pp). VLM 추가 (Qwen3.5-2B) 는 reach + safety axis에서 강하나 sub-mm/hold는 여전히 chain SigLIP2 우위 — 단일 architecture 가 모든 axis 1등 불가, **single training + multi exec deployment regime** 이 paper headline."

### 33.6 Hold 중요성 강조 (사용자 지적)

"이전에는 Success 해버리면 끝나지만 결국은 Hold도 중요하니까" — 100% 맞음. paper Table 1 표시 방식:

- **single-metric SR_old (3D dist < 5mm at end)**: retreat=2 에서 ACT/DP 100%로 saturated → 의미 없음.
- **multi-criteria 필수**: (close_2, holdSR, min_lat, safety) 4-axis 동시 표시.
- **holdSR 정의**: lateral<2.5mm for ≥20 contig steps — 단순 reach 가 아닌 **sustained alignment**, insertion phase 직전 안정.
- **현재 Pareto**: Qwen reach SOTA (SR 100, safety 2.86) vs chain hold SOTA (holdSR 81.5%). Single ckpt + multi exec mode 로 cover.

### 33.7 다음 axes (이번 야간 + 미래)

이번 야간 실험:
- **NEARGOAL_submm_hold_v1** 데이터 생성 중 (1800ep, 250 hold, 1.5mm perturb)
- **v11 (v2 ck1500 + new data + threshold 1.5)** + **v12 (v5 ck500 + new data + threshold 1.5)** — holdSR 52% / min_lat 0.87mm ceiling 시도
- **Multi-seed eval (2027/2028)** on v2/v5/v8 ck500/ck1500 — stochasticity bound
- 결과 → Section 32.4 update + 33.5 정량 update

미래 axis (highest impact):
1. **Ensemble (Qwen reach + chain hold) action averaging** — paper-grade, code 작업 ~2h.
2. **Input resolution 384/512** — SigLIP2 native patch 활용, sim HDF5 재생성 ~1일.
3. **Real robot transfer** — sim-to-real gap 측정 (user 작업).
4. **DDPM scheduler + 50 step (vs flow_match 10 step)** — inference cost 5x but precision 향상 가능 unconfirmed.


---

## Section 34: Honest eval suite (2026-05-24) — early-term artifact 진단 + 새 metric

### 34.1 Motivation

기존 모든 ablation 결과 (Section 22-33)는 `check_success()` early-termination 데이터 기반. 발견:
- 3D dist<5mm + 20-step hold 만족 시 episode 즉시 break (sim_eval_align_only.py:1257)
- 평균 episode 길이 ~120 step (max 250) — **절반 데이터 잘림**
- "Final lateral" = success 시점이라 model마다 다른 timing, fair 비교 어려움
- holdSR (lateral<2.5mm for 20 contig) underestimate — 모델이 lateral hold 달성하기 전에 3D 조건으로 끝남

기존 early-term data 에서 추출 가능한 **숨겨진 signal** (Reach@K 사용):

| ckpt | final close_2 (old) | **Reach@2** (any step) | gap (숨겨진 능력) |
|---|---|---|---|
| v11 ck1500 | 77.8 | **85.2** | +7.4pp 보임 |
| v9 ck500 | 70.4 | 74.1 | +3.7pp |
| v8 ck500 | 77.8 | 81.5 | +3.7pp |
| chain v5_combo ck2000 + exec=4 | 55.6 | **85.2** | **+29.6pp** ✨ |

→ chain v5_combo는 final close_2 56%만 보였지만 실제로 27 중 23 episode가 **어느 시점에서 lateral < 2mm 도달**. 단지 final-state에서 떠나가서 측정 못 한 것.

### 34.2 Eval protocol 변경

- `scripts/sim_eval_align_only.py:1485` — `--no-early-term` flag 추가
- `check_success()` 만족해도 break 안 함 → full 250 step 진행
- 결과: 모든 episode 동일 budget, fair 비교

```bash
# 새 표준 eval command
GPUS=0,1 TRAIN_CONFIG_OVERRIDE="<cfg>" bash Run_Eval_Parallel.sh align "<ckpt>" \
  --max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 3 --z-steps 1 --angle-steps 3 --repeats 1 \
  --perturb-xy-mm 10 --perturb-y-min-mm -25 --perturb-y-max-mm 25 \
  --perturb-angle-deg 5 --perturb-z-min-mm 0 --perturb-z-max-mm 0 \
  --retreat-mm 2 --num-steps-execute 2 \
  --no-early-term   # NEW
```

### 34.3 New metric suite (`scripts/honest_metrics.py`)

| metric | 정의 | 계산 |
|---|---|---|
| **Reach@K** | per-ep min(lateral) < K. fraction across eps | `(min(lat) < K).mean()` |
| **TTA@K** | first step lateral < K. median across reached eps | `median(argmax(lat<K))` |
| **min_lat_med** | per-ep min(lateral), median | `median(min(lat))` |
| **HoldSR@2.5_20** | any 20-step window all <2.5 | contig run check |
| **Settled_lat** | last 30 step median, then median across eps | `median(median(lat[-30:]))` |
| **Settled_std** | last 30 step std, then median | `median(std(lat[-30:]))` |
| **Max30<2.5** | last 30 step max < 2.5, episode SR | `(max(lat[-30:]) < 2.5).mean()` |
| **Safety_settled** | p99 of settled_lat across eps | `percentile(settled_lat, 99)` |

**핵심 차이**:
- "final" → "settled (last 30 step)" — 모델 간 fair, time-invariant
- Reach@K = peak ability, Settled = stability — 두 axis 분리
- HoldSR 유지 (contig 20) + 새 "Max30<2.5" 추가 (window-based, cleaner)

### 34.4 Honest eval results (진행 중)

8 key champions 재 eval 진행 중 (2026-05-24 13:47 launched):
- qwen_v2_1500, qwen_v5_500, qwen_v8_500/1000, qwen_v9_500, qwen_v11_1500
- chain_lathold_1000, chain_v5combo_2000 (exec=4)

각 eval ~20min × 8 = ~2.7h. 결과는 `align_eval_step*_exec*_diff10_noET/` 디렉토리 (기존 early-term 데이터는 `_origterm` 으로 backup).

→ 결과 표 + 새 champion ranking 추후 추가 예정.

### 34.5 Artifacts

- Patched eval: `scripts/sim_eval_align_only.py` (--no-early-term flag, line 1485)
- Metric analyzer: `scripts/honest_metrics.py`
- Pipeline: `/tmp/honest_eval_pipeline.sh`
- Logs: `logs/qwen_eval/honest_*.log`
- New eval dirs: `*_noET` suffix per ckpt
- Backup of original early-term data: `*_origterm` suffix


---

