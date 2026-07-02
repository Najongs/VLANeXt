# Ablation 표 1~8 — 최종 결과 (honest eval)

> **작성 2026-05-28 (자율 실행).** 판단 기준 = **lateral**. 모든 값 honest eval (`--no-early-term`, 250 step, tlen 검증), 27-cell grid (x∈{-10,0,10}, y∈{-25,0,25}, z=0, angle∈{-5,0,5}).
> 메트릭: R5/R2/R1 = min lateral < 5/2/1mm 도달 %, minLat = episode별 최소 lateral 평균, settled = 마지막 30 step lateral 평균, min/max = settled의 episode간 최소/최대, Hold5/2/1 = settled < 5/2/1mm %. SR은 소수점 1자리.
> ⚠️ **모든 ET 함정 회피**: 각 eval의 tlen을 250으로 검증함. (기존 ET eval은 ~145 step에서 종료돼 모든 조건이 3.4mm로 붕괴 → 사용 불가였음.)

---

## 표 1 — Vision Encoder (encoder만 교체, policy 고정)

| # | Encoder + Policy | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN (ConvNeXtV2) | 14.8 | 14.8 | 3.7 | 20.30 | 21.90 | 2.57 | 43.38 | 7.4 | 0.0 | 0.0 |
| 2 | DINOv3 | 18.5 | 11.1 | 7.4 | 21.00 | 22.62 | 1.92 | 45.16 | 7.4 | 3.7 | 0.0 |
| 3 | SigLIP2 (frozen) e4 | 100.0 | 88.9 | 59.3 | 1.08 | 6.35 | 1.10 | 12.21 | 29.6 | 3.7 | 0.0 |
| 4 | SigLIP2 (unfreeze last-4) | _정성적: mean SR 34% < frozen 48% (폐기)_ | | | | | | | | | |
| 5 | **Ours (Qwen3.5 VLM) e2** | 100.0 | 96.3 | 55.6 | 0.93 | 1.99 | 0.75 | 3.66 | 100.0 | 59.3 | 3.7 |

**해석**: CNN/DINOv3는 chain50k base를 해도 **catastrophic fail** (settled 22mm, R5 ~15%). VLM-pretrained encoder(SigLIP2/Qwen)만 수렴. encoder가 본질적 driver. Qwen이 모든 지표 최강 (worst case 3.66mm vs CNN 43mm).

---

## 표 8 — Execution 개수 (Qwen v11, exec만 변경) ✅ 완성

| exec | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100.0 | 92.6 | 59.3 | 1.00 | 1.96 | 0.70 | 3.94 | 100.0 | 55.6 | 11.1 |
| **2** | 100.0 | **96.3** | 55.6 | 0.93 | 1.99 | 0.75 | 3.66 | 100.0 | **59.3** | 3.7 |
| 4 | 100.0 | 92.6 | **66.7** | **0.86** | 2.06 | 1.05 | 3.54 | 100.0 | 48.1 | 0.0 |
| 8 | 100.0 | 92.6 | 55.6 | 0.91 | 2.29 | 0.89 | 3.59 | 100.0 | 40.7 | 3.7 |

**해석**: **exec=2가 sweet spot** (R2 96.3 + Hold2 59.3 최고, settled 낮음). exec↑ 할수록 **순간 정밀도(minLat)는 exec4에서 0.86으로 최고**지만 **유지력(Hold2)은 하락**(59→48→41). exec8은 settled 2.29로 가장 흔들림. → `feedback_inference_axis_exec2` 재확인.
(참고: exec4/8의 CSV-SR 37%/41%는 3D-dist 기준 success라 낮게 보이나, lateral R5는 100% 유지 — depth축 overshoot이지 lateral 정렬은 유지됨.)

---

## 표 6 — DCT loss on/off (clean pair: base·seed 동일, weight만 0.1↔0.0) ✅ 완성

| 조건 | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ |
|---|---|---|---|---|---|---|---|---|---|
| DCT on (w=0.1) | 100.0 | 81.5 | 51.9 | 1.24 | 6.24 | 2.25 | 12.11 | 37.0 | 0.0 |
| DCT off (w=0.0) | 100.0 | 81.5 | 51.9 | 1.24 | 6.30 | 2.19 | 11.96 | 37.0 | 0.0 |

**해석**: **차이 거의 0** (settled 6.24 vs 6.30, Hold5 동일 37.0, R2/R1 동일). DCT loss는 honest eval에서도 **무의미** → off 권장. 메모리 `project_dct_ablation_0522` 결론을 honest로 확정.

---

## 표 7 — Auxiliary loss **컴포넌트 ablation** (동일 base = v2_dual_lr1e6/ck1000) ✅

> ⚠️ "no aux" config는 존재하지 않음 — **dist는 항상 켜진 기반 aux**. 따라서 dist를 baseline으로 두고 lat/hold를 증분 추가하는 incremental ablation. 4개 모두 같은 base에서 finetune되어 clean.

| aux 조합 | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ |
|---|---|---|---|---|---|---|---|---|---|
| **dist** (base) | 100.0 | 81.5 | 51.9 | 1.23 | 6.31 | 2.20 | 12.16 | 29.6 | 0.0 |
| **dist + lat** (loss_lat_v1) | 100.0 | 81.5 | 51.9 | 1.24 | 6.24 | 2.13 | 12.14 | 33.3 | 0.0 |
| **dist + hold** (hold_only_v1)¹ | 100.0 | 81.5 | 51.9 | 1.22 | 6.25 | 2.19 | 12.00 | **37.0** | 0.0 |
| **dist + lat + hold** (loss_lat_hold_v1) | 100.0 | 81.5 | 48.1 | 1.24 | 6.32 | 2.23 | 12.08 | 33.3 | 0.0 |

¹ hold_only_v1은 hold loss + hold-rich 데이터를 함께 써서 순수 loss 효과는 아님 (표5와 동일 ckpt).

**해석**: dist baseline(6.31) 대비 단일 컴포넌트(+lat 6.24, +hold 6.25)는 −0.06~0.07mm / Hold5 +3.7~7.4pp 미세 개선. 그러나 **풀 조합(dist+lat+hold)은 settled 6.32로 baseline과 동일** — **컴포넌트가 누적되지 않음**(오히려 R1 51.9→48.1 소폭 하락). 즉 aux loss는 전부 marginal하고 서로 stack도 안 됨. SigLIP2 천장(~6.3mm)이 본질적 한계. 단일 best는 dist+hold(Hold5 37.0)지만 noise 범위.

---

## 표 5 — Hold 데이터 추가 전/후 (base = v2_dual_lr1e6, 표7과 공통 대조군) ✅ 완성

| 조건 | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ |
|---|---|---|---|---|---|---|---|---|---|
| Hold 데이터 추가 (hold_only_v1) | 100.0 | 81.5 | 51.9 | 1.22 | 6.25 | 2.19 | 12.00 | 37.0 | 0.0 |
| 추가 전 (v2_dual base) | 100.0 | 81.5 | 51.9 | 1.23 | 6.31 | 2.20 | 12.16 | 29.6 | 0.0 |

**해석**: **marginal** (settled −0.06mm, Hold5 +7.4pp = 2 episode). hold 데이터가 약간 도움이나 noise 수준. 메모리 `project_hold_loss_false_confound_0522`(hold가 holdSR 결정 안 함) 재확인.

---

## 표 4 — Y-neg 데이터 효과 (**y region별 분해**, exec2) ✅ 완성

aggregate로 보면 marginal하지만, yneg 데이터의 목적은 **y=−25 실패 영역 fix**라 y별로 봐야 진짜 효과가 드러남 (각 region n=9: x∈{−10,0,10}×angle∈{−5,0,5}).

| 모델 | y | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SigLIP2 yneg-OFF** (v5_combo) | +25 | 100.0 | 88.9 | 77.8 | 0.77 | 5.92 | 2.22 | 9.56 | 33.3 | 0.0 | 0.0 |
| | 0 | 100.0 | 100.0 | 77.8 | 0.69 | 4.61 | 3.04 | 6.55 | 66.7 | 0.0 | 0.0 |
| | **−25** | 100.0 | 55.6 | 0.0 | 2.25 | **8.35** | 5.65 | 12.12 | **0.0** | 0.0 | 0.0 |
| **SigLIP2 yneg-ON** (v10_yneg25) | +25 | 100.0 | 88.9 | 77.8 | 0.78 | 5.77 | 2.26 | 9.85 | 44.4 | 0.0 | 0.0 |
| | 0 | 100.0 | 100.0 | 77.8 | 0.64 | 4.56 | 3.00 | 6.48 | 66.7 | 0.0 | 0.0 |
| | **−25** | 100.0 | 55.6 | 0.0 | 2.31 | **8.44** | 6.00 | 12.14 | **0.0** | 0.0 | 0.0 |
| **Ours-Qwen** (VLM) | +25 | 100.0 | 100.0 | 44.4 | 1.06 | 2.67 | 1.29 | 3.66 | 100.0 | 22.2 | 0.0 |
| | 0 | 100.0 | 100.0 | 88.9 | 0.47 | 1.43 | 0.75 | 2.15 | 100.0 | 88.9 | 11.1 |
| | **−25** | 100.0 | 88.9 | 33.3 | 1.25 | **1.88** | 1.03 | 2.94 | **100.0** | 66.7 | 0.0 |

**핵심 발견**: 
1. **yneg 데이터는 y=−25를 전혀 못 고침** (settled 8.35→8.44mm, Hold5 0%→0%). 엉뚱하게 y=+25만 +11pp.
2. **y=−25 실패가 SigLIP2 aggregate(6.3mm)를 끌어올리는 주범** (y=0/+25는 4.6/5.9mm로 양호).
3. **Qwen(VLM)이 y=−25를 해결** (8.4mm/0% → **1.88mm/100%**). 
→ **y=−25 실패는 데이터 coverage 문제가 아니라 encoder/VLM capability 문제.** `project_y_neg_distribution_bias`의 "데이터로 fix" 가설을 honest eval이 반박, `project_qwen_reach_recover_sota_0523`의 "Qwen이 y=−25 해결"을 확정.

---

## 표 2 — Crop vs no-Crop ✅ (honest 보유)

| 조건 | R5↑ | R2↑ | settled↓ | Hold5↑ |
|---|---|---|---|---|
| Crop (crop_zoom_v1) | 56.0 | 33.0 | 33.34 | 0.0 |
| no-Crop (v5_combo) | 100.0 | 88.9 | 6.35 | 29.6 |

**해석**: crop은 **수렴 실패** (settled 33mm). trocar 주변 crop이 SigLIP2 patch 해상도를 뭉개 오히려 악화 → no-crop이 압도적. 메모리 `feedback_config_choices_intent` 가설 확정.

---

## 표 3 — Sensor (보류) ❌

`sensor_ablation`에 ck500만 존재, eval/대조군 없음. 재학습 필요로 **보류**. (사용자 재학습 skip 방침.)

---

# 🔑 종합 결론

1. **driver는 encoder (표1) + exec (표8)** — 나머지(DCT·aux·hold데이터·yneg데이터)는 전부 **marginal**.
2. **SigLIP2 베이스 ablation 군은 모두 settled ~6.2-6.3mm, Hold5 30-37%로 수렴** — DCT/aux/hold/yneg가 ±0.07mm, ±1-2 episode 차이뿐. honest eval이 ET 함정(전부 3.4mm로 붕괴)을 걷어내고 **진짜 marginal임을 확정**.
3. **진짜 도약은 VLM 교체** (SigLIP2 6.35mm → Qwen 1.99mm) + **exec=2**.
4. crop은 명확히 해로움 (33mm fail).

---

## 체크포인트 매니페스트
각 행의 정확한 체크포인트/config는 `RESULTS_ablation_plan.md` 의 "체크포인트 매니페스트" 섹션 참조. 모든 eval dir은 `align_eval_step{X}_exec{N}_diff10_SR{rate}` 형태로 저장됨 (기존 ET eval은 `.bak_<timestamp>`로 백업 보존).

## 참고 메모리
`project_sota_checkpoints_0528`, `feedback_no_early_term_mandatory`, `feedback_chain_dominant_over_encoder`, `project_dct_ablation_0522`, `project_hold_loss_false_confound_0522`, `feedback_inference_axis_exec2`, `feedback_config_choices_intent`
