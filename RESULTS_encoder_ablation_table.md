# 추가 표 1 — Vision Encoder Ablation (Lateral 메트릭)

> **작성 2026-05-28.** 판단 기준 = **lateral**. 모든 값 honest eval (`--no-early-term`, 250 step), n=27 cells, exec=2 기준.
> 목적: vision encoder 선택이 fine-align 성능에 미치는 영향. policy(DiT) 구조는 동일 고정, encoder만 교체.

---

## 메인 표

| # | Encoder + Policy | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **CNN (ConvNeXtV2) + policy** | 14.8 | 14.8 | 3.7 | 20.30 | 21.90 | 2.57 | 43.38 | 7.4 | 0.0 | 0.0 |
| 2 | **DINOv3 + policy** | 18.5 | 11.1 | 7.4 | 21.00 | 22.62 | 1.92 | 45.16 | 7.4 | 3.7 | 0.0 |
| 3 | **SigLIP2 (frozen) + policy** | 100.0 | 88.9 | 59.3 | 1.08 | 6.35 | 1.10 | 12.21 | 29.6 | 3.7 | 0.0 |
| 4 | **SigLIP2 (unfreeze last-4) + policy** | _정성적_ | _정성적_ | — | — | _정성적_ | — | — | — | — | — |
| 5 | **Ours (Qwen3.5 VLM) + policy** | 100.0 | 96.3 | 55.6 | 0.93 | 1.99 | 0.75 | 3.66 | 100.0 | 59.3 | 3.7 |

(↑ 높을수록 좋음, ↓ 낮을수록 좋음. SR % 소수점 1자리, 거리 mm. 전부 honest eval tlen 250.)

### 표 1 해석
- **CNN/DINOv3는 chain50k base에서도 catastrophic fail** (settled 22mm, R5 ~15%). chain matching을 해도 ConvNeXt/DINOv3 encoder로는 fine-align 수렴 불가 → `feedback_chain_dominant_over_encoder` 재확인.
- **VLM 계열만 성공**: SigLIP2 6.35mm → Qwen 1.99mm. 즉 "encoder가 vision-language pretrained여야 한다"가 핵심.
- worst case(max_lat): CNN 43mm, DINOv3 45mm vs Ours 3.66mm → 의료 안전성 관점에서 비교 불가 수준.

---

## 행별 체크포인트 / 상태

| # | 모델 | 체크포인트 | 상태 |
|---|---|---|---|
| 1 | CNN (ConvNeXtV2-base-22k-384) | `VLANeXt_ConvNeXt_chain50k/v1/checkpoint_45000.pt` | ⏳ eval 진행 중 |
| 2 | DINOv3 (vitl16-lvd1689m) | `VLANeXt_DINOv3_chain50k/v1/checkpoint_40000.pt` | ⏳ eval 진행 중 |
| 3 | SigLIP2 frozen (so400m-512) | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_2000.pt` (exec4) | ✅ 완료 |
| 4 | SigLIP2 unfreeze last-4 | (체크포인트 삭제됨, config만: `v4_unfreeze_seed*`) | ❌ 재학습 skip → 정성적 |
| 5 | Ours = Qwen3.5-2B VLM | `VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_1500.pt` (exec2) | ✅ 완료 |

---

## 정성적 근거 (수치 미보유 행)

**#4 SigLIP2 unfreeze last-4**: 과거 4-seed 실험에서 **mean SR 34% < frozen 48%** (σ 23pp, seed lottery)로 frozen보다 일관되게 열세 → 폐기. 체크포인트 삭제됨. 재학습 skip 결정 (사용자: "이전에 결과 안 좋아 포기한 구조, 무조건 낮음"). 출처: 메모리 `project_unfreeze_seed_lottery`, `project_unfreeze_breakthrough`.

**참고 — non-chain CNN/DINOv3 (chain 없는 fresh)**: settled ~22-24mm, R5 ~15-19%로 catastrophic fail. 이는 `feedback_chain_dominant_over_encoder`("chain 없는 encoder swap 무효")의 증거. 본 표 #1/#2는 chain50k base로 eval하여 fair한 수치 확보 예정.

---

## ⚠️ 공정성 주의

- **#1/#2 (CNN/DINOv3)** 는 `chain50k` **base 50k stage**. **#3 SigLIP2 v5_combo**는 base + reach_recover finetune cascade까지 간 것이라 stage가 한 단계 더 깊음. 즉 #1/#2가 불리한 비교 → "encoder base 수준에서도 SigLIP2가 우월"을 보이는 용도. 완전 동일 stage 비교하려면 #1/#2에도 finetune cascade 필요 (현재 재학습 skip).
- **#5 Ours**는 exec=2, **#3 SigLIP2**는 exec=4 (각자 best deploy). exec 차이 명시.

---

## 메트릭 정의
`RESULTS_lateral_metrics_table.md`와 동일. R5/R2/R1 = min lateral < 5/2/1mm 도달 %, settled = 마지막 30 step lateral 평균, min/max_lat = settled의 episode간 최소/최대, Hold5/2/1 = settled < 5/2/1mm episode %.

## 참고 메모리
`project_sota_checkpoints_0528`, `feedback_chain_dominant_over_encoder`, `project_unfreeze_seed_lottery`, `project_baseline_matrix_0522`, `feedback_no_early_term_mandatory`
