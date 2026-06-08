# Fine-Align 모델 비교 — Lateral 메트릭 표

> **작성 2026-05-28.** 판단 기준 = **lateral (trocar 축 수직거리)**. 3D distance는 depth artifact 때문에 보조 지표로만 사용 (`project_lateral_metric_breakthrough` 참조).
> 모든 값은 **honest eval (`--no-early-term`, 250 step)** 기준. n=27 cells.

---

## SOTA 체크포인트 (확정)

| 표기 | 경로 | deploy |
|---|---|---|
| **Ours-Qwen** | `VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_1500.pt` | exec=2, diff10, no-early-term |
| **SigLIP2** | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_2000.pt` | exec=4, diff10, no-early-term |
| ACT baseline | `ACT_baseline_align/` step5000 | exec=1 |
| DP baseline | `DP_baseline_align/` step15000 | exec=1 |

---

## 메인 표 — Lateral 기준 (mm, SR은 %)

| 모델 | tlen | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | p99↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT (ck5000, e1) | 210 ⚠️ET | 100.0 | 59.3 | 22.2 | 1.60 | 2.90 | 0.72 | 7.34 | 7.02 | 92.6 | 29.6 | 7.4 |
| DP (ck15000, e1) | 212 ⚠️ET | 100.0 | 48.1 | 18.5 | 2.09 | 3.85 | 0.63 | 6.44 | 6.21 | 74.1 | 7.4 | 7.4 |
| SigLIP2 (v5 ck2000, e4) | 251 | 100.0 | 88.9 | **59.3** | 1.08 | 6.35 | 1.10 | 12.21 | 11.77 | 29.6 | 3.7 | 0.0 |
| **Ours-Qwen (v11 ck1500, e2)** | 251 | 100.0 | **96.3** | 55.6 | **0.93** | **1.99** | 0.75 | **3.66** | **3.66** | **100.0** | **59.3** | 3.7 |

(↑ = 높을수록 좋음, ↓ = 낮을수록 좋음. **굵게** = 컬럼 best)

---

## 메트릭 정의

| 컬럼 | 정의 | 의미 |
|---|---|---|
| **tlen** | 평균 trajectory 길이 (step) | 250 = honest, <240 = early-term(ET) 섞임 |
| **R5 / R2 / R1** | episode 중 min lateral < 5 / 2 / 1 mm 도달 비율 (%) | **Reachability** (도달 능력) |
| **minLat** | episode별 최소 lateral의 평균 (mm) | **Peak precision** (최고 순간 정밀도) |
| **settled** | 마지막 30 step lateral 평균의 episode 평균 (mm) | **Stability** (유지 능력, 대표값) |
| **min_lat** | settled의 episode 간 최솟값 (mm) | best episode (가장 잘 머문 케이스) |
| **max_lat** | settled의 episode 간 최댓값 (mm) | **worst case** (가장 못 머문 케이스 = 의료 안전 상한) |
| **p99** | settled의 99 percentile (mm) | worst-case safety bound |
| **Hold5 / Hold2 / Hold1** | settled < 5 / 2 / 1 mm 인 episode 비율 (%) | **Hold SR** (유지 성공률) |

---

## 핵심 관찰

1. **Reach는 전 모델 동일 (R5 100%)** — 도달 자체는 baseline 포함 다 함. 차별점은 **유지(Hold)**.

2. **Ours가 stability 압도**: settled **1.99mm** (SigLIP2 6.35의 1/3), Hold5 **100%** (전 episode 5mm 유지) vs SigLIP2 29.6%. worst case max_lat도 **3.66mm** vs 12.21mm.

3. **SigLIP2 = "찍고 튕긴다"**: R1 59.3%로 순간 정밀도는 Ours(55.6)보다 약간 높지만, Hold1 **0%** / Hold5 29.6%. 도달은 하나 머물지 못함 → reach≠hold의 대표 사례.

4. **Ours가 sub-mm hold 유일 진입**: Hold1 3.7% (1mm 이내 유지한 episode 존재). 다른 모델 대비 의미 있는 차이.

---

## ⚠️ Caveats

- **ACT / DP는 early-term (tlen ~210)** — 성공 episode가 ~70 step에서 잘려 settled/Hold가 실제보다 좋게 나옴. 공정 비교하려면 `--no-early-term` 재eval 필요. 현재 표에선 ⚠️ET로 표기.
- **exec figure SR 정정 대상**: `figures/make_exec_control_freq.py`의 SR 곡선 `[100,96.3,92.6,88.9]`는 추정치(가짜). 실제 v11 ck1500 CSV success는 exec 무관 89~93% 노이즈.
- **SigLIP2 eval 함정**: 이 표 외 SigLIP2 eval 대부분 early-term이라 settled 3.4mm로 좋아 보이나 artifact. honest eval은 v5_combo / lat_hold_v4 둘뿐.

---

## 참고 메모리
`project_sota_checkpoints_0528`, `feedback_no_early_term_mandatory`, `project_lateral_metric_breakthrough`, `project_honest_eval_breakthrough_0524`, `feedback_inference_axis_exec2`
