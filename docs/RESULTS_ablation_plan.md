# Ablation 표 마스터 플랜 (표 1~8)

> **작성 2026-05-28.** 판단 기준 = **lateral**, honest eval (`--no-early-term`, 250 step), 27-cell grid (x∈{-10,0,10}, y∈{-25,0,25}, z=0, angle∈{-5,0,5}), exec=2 기본.
> 메트릭: R5/R2/R1 = min lateral < 5/2/1mm %, settled = 마지막 30 step lateral 평균, min/max = settled 범위, Hold5/2/1 = settled < 5/2/1mm %.

---

## 🚨 핵심 문제 — 기존 ablation eval은 전부 early-term

표 4·5·6·7의 기존 eval(dct_on/off, hold_only, loss_lat, yneg 등)은 **거의 전부 early-term (traj ~145 step)**. early-term은 drift 시작 전에 종료돼서 **모든 조건이 settled ~3.4mm / R5 100 / R2 81 / H5 74로 동일하게 붕괴** → ablation 차이를 전혀 구분 못 함.

→ 메모리의 "DCT/hold/aux marginal" 결론은 **early-term eval 기반**이라 honest로 재검증 필요. 각 조건을 `--no-early-term`으로 재eval해야 의미 있는 표가 나옴.

---

## 표별 상태 요약

| 표 | 비교축 | 체크포인트 | 상태 | 필요 작업 |
|---|---|---|---|---|
| 1 | Encoder (CNN/DINOv3/SigLIP2/unfreeze/Ours) | 대부분 보유 | 🔄 CNN/DINOv3 eval 진행 중 | 완료 대기 |
| 2 | Crop vs no-Crop | crop_zoom_v1/v2 + v5_combo | ⚠️ honest 보유(crop=fail) | 채움 가능 (crop 실패) |
| 3 | Sensor vs no-Sensor | sensor_ablation (ck500만) | ❌ eval 없음, 대조군 없음 | train + eval 필요 (최난) |
| 4 | Y-neg 데이터 추가 전/후 | yneg_* + 대조군 | ❌ ET only | honest 재eval 쌍 |
| 5 | Hold 데이터 추가 전/후 | hold_only_* + 대조군 | ❌ ET only | honest 재eval 쌍 |
| 6 | DCT on/off | dct_on_v1, dct_off_v1 | ❌ ET only (체크포인트 보유) | honest 재eval 쌍 (clean) |
| 7 | Aux loss on/off | loss_lat_v1 + 대조군 | ❌ ET only | no-aux baseline + 재eval |
| 8 | Execution 개수 | Qwen v11 exec 1/2/4/8 | 🟡 exec1,2 honest 보유 | exec4,8 honest 재eval |

---

## 표 8 — Execution 개수 (부분 완성) ✅🟡

Qwen v11 ck1500 단일 모델, exec만 변경. **exec1/2는 honest 측정 완료**, exec4/8 재eval 필요.

| exec | R5↑ | R2↑ | R1↑ | minLat↓ | settled↓ | min_lat↓ | max_lat↓ | Hold5↑ | Hold2↑ | Hold1↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100.0 | 92.6 | 59.3 | 1.00 | 1.96 | 0.70 | 3.94 | 100.0 | 55.6 | 11.1 |
| 2 | 100.0 | 96.3 | 55.6 | 0.93 | 1.99 | 0.75 | 3.66 | 100.0 | 59.3 | 3.7 |
| 4 | _재eval_ | | | | | | | | | |
| 8 | _재eval_ | | | | | | | | | |

관찰(잠정): exec1→2에서 R2 92.6→96.3↑ (reach 약간↑), Hold1 11.1→3.7↓. exec↑ 효과는 메모리 `feedback_inference_axis_exec2`(exec=2 minLat free win) 재확인 필요.

---

## 표 6 — DCT on/off (재eval 대상, 가장 clean) ⭐

`dct_on_v1` vs `dct_off_v1` — 동일 recipe, DCT weight만 0.1 vs 0.0. 체크포인트 보유. **honest 재eval만 하면 즉시 완성**.

| 조건 | R5 | R2 | R1 | settled | Hold5 | Hold2 |
|---|---|---|---|---|---|---|
| DCT on (w=0.1) | _재eval_ | | | | | |
| DCT off (w=0.0) | _재eval_ | | | | | |

(기존 ET 결과: 둘 다 set 3.4, H5 74 — 구분 불가. 메모리 `project_dct_ablation_0522`: ET 기반 "marginal" 결론.)

---

## 표 2 — Crop vs no-Crop (honest 보유, crop=fail) ⚠️

| 조건 | R5 | R2 | settled | Hold5 | 비고 |
|---|---|---|---|---|---|
| Crop (crop_zoom_v1) | 56 | 33 | 33.34 | 0 | honest, **catastrophic fail** |
| Crop (crop_zoom_v2) | 59 | 30 | 61.15 | 0 | honest, fail |
| no-Crop (v5_combo) | 100 | 89 | 6.35 | 30 | honest 정상 |

→ crop은 honest eval에서 **수렴 실패** (settled 33-80mm). 메모리 `feedback_config_choices_intent`의 "patch 해상도 뭉개짐" 가설과 연결. 단 crop_zoom은 별도 base에서 학습돼 stage 불일치 — 깔끔한 비교 원하면 동일 base에서 crop on/off 재학습 필요.

---

## 표 4 / 5 / 7 — Yneg / Hold / Aux loss (재eval 대상)

전부 ET only → honest 재eval 쌍 필요. 각 비교의 체크포인트 후보:
- **표4 Yneg**: `reach_recover_v10_yneg25` (yneg 추가) vs yneg 추가 전 base. 대조군 식별 필요.
- **표5 Hold**: `hold_only_v1/v2/v3` (hold 데이터) vs base. 대조군 식별 필요.
- **표7 Aux loss**: `loss_lat_v1` (aux on, base=v2_dual_lr1e6/ck1000) vs `v2_dual_lr1e6/ck1000` (aux off) ← 대조군 명확.

---

## 표 3 — Sensor (최난, 보류) ❌

`sensor_ablation`에 ck500만 존재, eval 없음. no-sensor 대조군도 매칭 안 됨. proprio sensor 추가는 train부터 다시 해야 함 (메모리 `project_sensor_proprio`). **재학습 skip 방침이면 보류 또는 정성적 처리.**

---

## 📋 재eval 큐 (우선순위)

GPU1/GPU2 가용. 각 eval ~10-15min (honest 250 step, exec2). EGL = Mesa software (race 없음).

1. **표1 마무리**: CNN ck45000, DINOv3 ck40000 (진행 중)
2. **표6 DCT**: dct_on_v1, dct_off_v1 — 가장 clean, 즉시 가능
3. **표8 exec**: Qwen v11 exec4, exec8
4. **표7 aux**: loss_lat_v1 (on) + v2_dual_lr1e6/ck1000 (off)
5. **표4 yneg**: reach_recover_v10_yneg25 + base
6. **표5 hold**: hold_only_v* + base
7. **표2 crop**: (honest 이미 보유, 동일 base 재학습은 선택)
8. **표3 sensor**: 보류 (재학습 필요)

---

---

## 📌 체크포인트 매니페스트 (각 표 행 → 정확한 ckpt + config)

> 모든 SigLIP2 모델: encoder `google/siglip2-so400m-patch16-512`, policy 동일. eval: exec2, no-early-term, 27-cell(z=0) grid.

### 표 1 — Encoder
| 행 | checkpoint | train-config | exec |
|---|---|---|---|
| CNN | `VLANeXt_ConvNeXt_chain50k/v1/checkpoint_45000.pt` | `sim_train_align_convnext_chain50k_v1_config.yaml` | 2 |
| DINOv3 | `VLANeXt_DINOv3_chain50k/v1/checkpoint_40000.pt` | `sim_train_align_dinov3_chain50k_v1_config.yaml` | 2 |
| SigLIP2 frozen | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_2000.pt` | `sim_train_align_reach_recover_v5_combo_config.yaml` | 4 |
| SigLIP2 unfreeze | (삭제됨, config `v4_unfreeze_seed123`) — 정성적 | last_n_unfreeze n=4 | — |
| Ours (Qwen) | `VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_1500.pt` | `sim_train_align_qwen_reach_recover_v11_submm_tight_config.yaml` | 2 |

### 표 6 — DCT (clean pair, base 동일·weight만 차이)
| 행 | checkpoint | dct_weight |
|---|---|---|
| DCT on | `VLANeXt_SigLIP2_NEARGOAL/dct_on_v1/checkpoint_1500.pt` | 0.1 |
| DCT off | `VLANeXt_SigLIP2_NEARGOAL/dct_off_v1/checkpoint_1500.pt` | 0.0 |

### 표 7 — Aux loss (base = v2_dual_lr1e6/ck1000)
| 행 | checkpoint | aux |
|---|---|---|
| Aux on | `VLANeXt_SigLIP2_NEARGOAL/loss_lat_v1/checkpoint_1500.pt` | dist+lateral+hold |
| Aux off (base) | `VLANeXt_SigLIP2_NEARGOAL/v2_dual_lr1e6/checkpoint_1000.pt` | 없음 |

### 표 5 — Hold 데이터 (base = v2_dual_lr1e6/ck1000, 표7과 공통 대조군)
| 행 | checkpoint |
|---|---|
| Hold 데이터 추가 | `VLANeXt_SigLIP2_NEARGOAL/hold_only_v1/checkpoint_1500.pt` |
| 추가 전 (base) | `VLANeXt_SigLIP2_NEARGOAL/v2_dual_lr1e6/checkpoint_1000.pt` |

### 표 4 — Y-neg 데이터 (base = v5_combo/ck2000, 이미 honest 보유)
| 행 | checkpoint |
|---|---|
| Yneg 추가 | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v10_yneg25/checkpoint_1500.pt` |
| 추가 전 (base) | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_2000.pt` |

### 표 8 — Exec (단일 ckpt, exec만 변경)
| 행 | checkpoint | exec |
|---|---|---|
| exec 1/2/4/8 | `VLANeXt_Qwen35_NEARGOAL/reach_recover_v11_submm_tight/checkpoint_1500.pt` | 1/2/4/8 |

### 표 2 — Crop (별도 base, stage 불일치 주의)
| 행 | checkpoint |
|---|---|
| Crop | `VLANeXt_SigLIP2_NEARGOAL/crop_zoom_v1` (honest fail) |
| no-Crop | `VLANeXt_SigLIP2_NEARGOAL/reach_recover_v5_combo/checkpoint_2000.pt` |

### 표 3 — Sensor (보류)
| 행 | checkpoint |
|---|---|
| Sensor | `VLANeXt_SigLIP2_NEARGOAL/sensor_ablation/checkpoint_500.pt` (eval 없음) |
| no-Sensor | 미정 |

---

## ✅ 완료 (2026-05-28 자율 실행)
10개 honest eval 완료 → **표 1·4·5·6·7·8 전부 채움**. 결과는 **`RESULTS_ablation_tables_FINAL.md`** 참조.
- 종합: driver = encoder(표1) + exec(표8). DCT/aux/hold데이터/yneg데이터는 전부 marginal (settled ±0.07mm).
- 표2 crop = honest fail(33mm) 확정. 표3 sensor = 보류(재학습 필요).
- ⚠️ eval 출력 dir은 `_SR{rate}` 접미사로 저장됨. 기존 ET eval은 `.bak_<ts>`로 백업 보존.
- 표4 yneg clean pair 완성 (v5_combo exec2 재eval 완료, marginal 확정).

---

## 참고 메모리
`project_sota_checkpoints_0528`, `feedback_no_early_term_mandatory`, `project_honest_eval_breakthrough_0524`, `project_dct_ablation_0522`, `project_ablation_master_0522`, `feedback_config_choices_intent`, `feedback_inference_axis_exec2`, `feedback_gpu_concurrency`
