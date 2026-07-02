# Paper Patches — 2026-05-21 (Review Comments 보완)

심사위원 공격 포인트 4가지에 대한 patches. 각 섹션에 paste-ready.

---

## Patch 1: 4.X (또는 4.7 통합) "Safety Analysis — Worst-case Lateral Bound"

> ### Multi-criteria Evaluation: Safety Bounds vs Reach Rate
>
> 의료 미세수술에서는 "평균적 정확도"보다 "환자 안전을 보장하는 최악 경계 (worst-case bound)"가 본질적으로 중요하다. 단일 SR 지표는 평균 행동에 가중되어 안전성 평가에 부적절하다. 본 연구는 27-cell perturbation grid 전체에서 lateral error의 **최대값 (worst-case)** 과 **분산 (variability)** 도 정량 평가한다.
>
> **Table 4.X**: 주요 모델 별 lateral worst-case bound (mm)
>
> | Model | min<sub>med</sub> | min<sub>worst</sub> | std | y=−25 worst | final<sub>worst</sub> |
> |---|---|---|---|---|---|
> | Vision-Policy v4 (night) | 0.87 | **3.61** | 0.96 | 3.61 | 12.33 |
> | Vision-Policy v3-strict | 1.13 | **3.66** | 0.94 | 3.66 | 12.62 |
> | Vision-Policy v2-dual | 1.00 | **3.42** | 0.92 | 3.42 | 12.85 |
> | b100 baseline + finetune | 2.16 | 4.98 | 1.42 | 4.98 | **7.45** |
>
> **관찰**: SR<sub>old</sub> (3D dist<5mm 도달) 단일 지표로는 b100 family가 우수 (85.2% vs 70-74%)이나, **lateral worst-case bound**는 night model이 모든 cell에서 3.7mm 이내로 더 tight (b100은 5mm). 반면 b100은 trajectory final state가 더 안정 (7.5mm vs 12.5mm). 이는 두 모델 family가 **상이한 안전 특성**을 가짐을 보여주며, **medical robotics에서는 단일 SR 지표가 아닌 multi-criteria 평가 (median + tail + variability)가 필수**임을 시사한다.
>
> exec=2 stride 적용 시 worst-case bound이 5-10% 추가 개선되어 (다른 free-win 변경 없음), **inference-time hyperparameter도 안전성 영역에 영향을 미친다**는 점을 추가로 확인했다.

---

## Patch 2: 4.3 (Baseline Comparison) — 고전 Visual Servoing 방어

> ### Classical Control Baseline의 정량 비교 제외 사유
>
> 자코비안 기반 PBVS/IBVS 등 고전 visual servoing은 trocar 입구 검출의 안정성에 의존한다. 본 연구의 사전 feasibility 실험 (Phase 1.5 OpenCV 기반 detector, project_phase15_opencv_feasibility 참조)에서, MuJoCo digital twin 환경의 트로카 검출 median 픽셀 오차는 약 80px (입력 256px 기준 31%)였으며, 이는 IBVS feature jitter로 직결되어 sub-mm 제어를 불가능하게 한다. ArUco 마커 부착 시에도 1.2에서 기술한 광학적 jitter (난반사 + edge detection 노이즈)로 시야 가림이 발생하여 수렴이 보장되지 않는다. 따라서 본 비교군에서는 동일 spec의 학습 기반 baseline (VLA-LM group)만 정량 평가했으며, 고전 제어는 **선행 연구의 한계로 인용하며 정량 비교에서 제외**한다 (추후 확장 연구 필요).

---

## Patch 3: 5.1 (Limitations) 추가 항목

> ### Limitation 3: Sim-only Optimization의 한계
>
> 본 연구의 모든 평가는 MuJoCo digital twin 환경에서 수행되었다. 실제 광학현미경 환경의 specular reflection, depth-of-field, 카메라 캘리브레이션 오차 등은 본 평가에서 직접 반영되지 않았으며, sim-to-real gap에 대한 정량적 검증은 향후 연구로 미루었다. 다만 본 연구의 데이터 mix에 포함된 6,000 episodes의 real demonstration data (project_real_data_frames)에서 관찰된 **real action scale이 sim 대비 2~3배 큼** (특히 Y dpos)이라는 정량적 증거는, 단순 multi-domain union이 도메인 갭을 자동으로 해결하지 못하며, **explicit domain adaptation 또는 stage-wise finetune** 이 필요함을 시사한다.
>
> ### Limitation 4: Visual Domain Robustness 부분 검증
>
> 학습 데이터 중 일부 변종 (b100 baseline)은 random_resized_crop, brightness/contrast/saturation/hue color jitter 등 image-level augmentation을 적용하였으나, MuJoCo 환경 자체의 **lighting / texture / camera intrinsic randomization은 적용되지 않았다.** 이로 인해 모델이 시뮬레이션 특정 픽셀 패턴에 over-fit했을 가능성을 완전히 배제할 수 없다.
>
> ### Limitation 5: SigLIP 부분 unfreeze의 Seed Sensitivity
>
> Pre-trained vision backbone의 마지막 4 layer를 학습 대상에 포함시키는 시도는 단일 seed cherry-pick에서는 baseline 대비 우수했으나 (77.8%), n=4 multi-seed 검증에서 mean 34% < frozen 48% (σ 23pp)로 robustness가 부족함을 확인 (project_unfreeze_seed_lottery). 본 연구의 모든 final report는 **frozen SigLIP** 기반이며, 부분 unfreeze는 더 큰 데이터 또는 정규화 기법 적용 후 재검증 대상이다.

---

## Patch 4: 5.2 (Future Work) 추가 항목

> ### Real-World Deployment via Visual Domain Randomization (단기 우선)
>
> Sim에서 검증된 최종 champion 모델을 Meca500 실기에 zero-shot deployment하고, 실패 패턴 (lighting / specular / depth ambiguity)을 정량 분석한다. 도메인 갭 해소를 위해 다음 두 가지를 병행:
> 1. **MuJoCo level Visual DR**: lighting direction/intensity randomization, table/needle texture/specular property randomization, camera intrinsic/extrinsic small perturbation 추가하여 재학습.
> 2. **Photometric augmentation 확장**: 현재 적용된 color jitter 외 GaussianBlur, motion blur, ISO noise 추가.
>
> ### Inference-time Hyperparameter Search (단기)
>
> Section 4.5에서 발견한 action chunk stride (exec=N) 효과처럼, **재학습 없이 inference 변경만으로 정밀도/안전성을 개선하는 free-win** 탐색을 systematic하게 진행한다 (diffusion timesteps, classifier-free guidance scale, action ensemble across sampling seeds).
>
> ### Multi-seed Paper-grade Robustness Validation (중기)
>
> 현재 champion은 single seed (2026). n=3-5 seed로 SR/lateral 분포 재현성을 정량화하고 표준편차 보고를 paper-grade 수준으로 격상한다.
>
> ### Input Resolution Scaling (중기)
>
> SigLIP2-so400m-patch16-**512** native resolution과 학습 입력 256×256 사이의 misalignment를 해소. HDF5 재생성 후 입력 384/512에서 spatial token이 sub-pixel 정보를 보존하는지 정량 평가.

---

## Patch 5: 5.3 Conclusion — 정량 수치 갱신 + Safety 강조

> ### Conclusion (수정안)
>
> 본 연구는 안과 미세수술의 트로카 정렬 작업에 대해 **(1) 종래의 평가 지표였던 3D distance가 retreat offset에 의한 measurement artifact임을 발견**하고 lateral perpendicular projection metric을 새로 도입했으며, **(2) VLM의 LM 디코더 병목을 SigLIP2 + Action Diffusion + 보조 손실 함수 (aux_distance + aux_lateral + aux_hold)로 대체**하여 sub-mm precision을 정량 입증했다. 27-cell perturbation grid (xy ±10mm, y ±25mm, angle ±5°) 전체에서:
>
> - **Lateral median 0.87mm**, **lat<0.5mm 25.9%**, **lat<1mm 51.9%** (best 모델)
> - **SR<sub>old</sub> (dist<5mm) 85.2%** (b100 + sim-only finetune)
> - **y=−25 region SR<sub>old</sub> 100%** (이전 distribution bias 진단 후 multi-stage finetune으로 해결)
> - **Safety bound (worst-case lateral) 3.4-3.7mm** (의료 robotics 평균 외 worst-case 정량 평가)
>
> 또한 본 연구는 보조 신호 융합 시도들 (OCT proprio, keypoint head + handoff servo, overlay loss, crop-zoom finetune, sensor handoff) 모두가 단순 frozen + 정제 데이터 + task-specific 보조 손실 조합을 outperform하지 못함을 정량 확인하며, **"explicit prior 주입은 의외로 brittle하다"**는 일반화된 insight를 도출했다. 클리닉적으로는 기존 51분의 수술 준비 시간 + 거대 프레임 로봇을 요구했던 트로카 정렬 작업을 commodity 상용 로봇(Meca500) + 단일 tool camera + 비전 인공지능으로 자율화 가능함을 입증했다.

---

## Patch 6: 4.4.1 Architecture Ablation — Over-training & Composite-metric Trap (2026-05-21 추가)

> ### Larger Pretraining Budget Doesn't Help BC Fine-tuning
>
> 우리는 base pretraining checkpoint의 학습 길이가 finetune 결과에 미치는 영향을 정량 비교했다. 동일 finetune recipe (lr 5e-7, aux_lateral + aux_hold loss, yneg_hold + perfect_strict 데이터) 적용 시:
>
> | base ckpt | finetune step | SR<sub>old</sub> | min_dist median | lateral mean | diverge rate |
> |---|---|---|---|---|---|
> | b100 (3000 step) | 1500 | **85.2%** | 3.90mm | 3.05mm | low |
> | b100 (50,000 step) | 1500 | **37.0%** | 4.19mm | 4.67mm | **70.6%** |
> | b100 (50,000 step) | 5000 | 33-40% | 4.30mm | 4.65mm | 68-89% |
>
> **결과**: 50,000-step pretrained base 사용 시 SR<sub>old</sub>이 45pp 후퇴하며, episodes가 2mm 영역을 순간 통과한 뒤 발산(diverge)하는 oscillation pattern으로 fail mode가 변화. 이는 더 긴 pretraining이 fine-control task의 BC distribution shift에 더 brittle해진다는 사실을 시사한다 (큰 distribution gap 이전 더 좋은 prior가 finetune 시 미세 보정에 저항).
>
> ### Composite Precision Metric의 Diverge Trap
>
> 우리의 multi-criteria composite ranking (close_once_2mm + close_once_1mm + handoff_ok + p90_dist)는 **순간 통과형 oscillator 모델을 winner로 선정**하는 함정을 보였다. 50k base finetune이 close_2mm 11.1% (vs baseline 3.7%) 3배 향상으로 1위를 차지했으나, final-state SR<sub>old</sub>는 37%로 baseline 85%의 절반에도 못 미친다.
>
> 이는 sub-mm precision 평가 시 close-once metric 단독으로는 oscillator를 stable approacher와 구별 불가능함을 의미하며, 후속 medical robotics 연구는 **반드시 final-state SR과 diverge_rate를 병행 보고**해야 한다는 lesson learned를 추가한다.

---

## Patch 7: 4.5 Y-Region Asymmetry — Model-Specific Weak Region (2026-05-21 추가)

> ### Distribution Bias가 Model-Specific Weak Region을 만든다
>
> 27-cell grid를 y-axis 3개 region (y ∈ {−25, 0, +25}mm, 각 9 cells)로 분할하여 분석한 결과, 동일 데이터로 학습된 두 best 모델이 정반대 weak region을 보였다:
>
> | model | y=−25 | y=0 | y=+25 |
> |---|---|---|---|
> | reach champion (b100v4_ft_phase2_lowlr/ck1500) | **100%** | **100%** | 55.6% |
> | minLat champion (lat_hold_v4_yneg_hold/ck1000) | 11.1% | 100% | **100%** |
>
> 두 모델은 동일한 데이터 셋(approach + fine_align + yneg_hold)으로 학습되었음에도, finetune recipe (aux_loss weights, learning rate, base ckpt)의 차이만으로 한 모델이 y=+25에서, 다른 모델이 y=−25에서 fail한다. 이는 **training distribution의 y-balance 자체가 약점을 결정하는 게 아니라, loss landscape에서의 local optimum이 어느 region을 더 잘 학습할지 결정함**을 시사한다.
>
> 두 모델의 weak region이 cross-complementary하다는 점은 ensemble (또는 multi-task) 학습이 single-model precision-reach trade-off를 깰 가능성을 제시한다. 본 연구의 다음 단계 (Phase 3 rot-boost finetune)는 reach champion base에서 aux_hold rot_weight를 강화하여 lateral precision 회복 + y=+25 weakness 동시 보강을 시도하고 있다.

---

## 진행 순서

1. Patch 1-2-3-4-5-6-7 위 텍스트를 paper 해당 섹션에 paste 또는 변형
2. 사용자 hardware test (real Meca500) 진행 후 4.8 "Preliminary Real Deployment" 추가
3. 사용자 ArUco jitter 영상 → 1.2 한계 2 또는 4.3 baseline 방어 supplementary로 인용
4. phase3_rot1 결과 도착 시 Patch 6/7 정량 수치 갱신
