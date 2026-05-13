# Fine-align 실험 로그 (2026-05-12 ~ 05-13)

VLANeXt 기반 needle/trocar 정렬. 본 task의 **최종 목표 = Hybrid pipeline**:
- **VLA (이 문서의 대상)**: trocar 근처(≤5mm)까지 데려가고, 그 영역에서 머무르려 노력
- **Sensor grid search**: 마지막 mm은 contact sensor 5mm 격자로 잡음 (별도 트랙)

따라서 **VLA의 평가 기준은 SR이 아니다**. SR은 hold-까지 다 충족해야 하는 sensor-역할 metric. VLA가 진짜 잘 했는지는 **"trocar 근처에 닿고 머무르려 함"** 의 trajectory 신호로 판정.

Eval: **exec=1 closed-loop** (매 step 새 vision으로 chunk 재예측, 첫 1만 실행). Grid: xy×z×angle×repeats.

---

# ⭐ EXECUTIVE SUMMARY (2026-05-13 update)

## 🎯 평가 framing 변경 — SR 단독 판단 금지

VLA를 SR로만 보면 **세 종류 오해**가 발생:
1. **SR=0인데 좋은 모델**: trocar 근처까지 잘 데려가고 머무르지만 strict 5mm+10°+hold 못 함 → 우리 hybrid에선 충분히 좋음
2. **SR↑인데 운**: 적은 ep에서 hold 한두 개 운 좋게 박힘 (#8 sensor_lr5e5 SR 16% — close-once<5mm는 #1보다 나쁨)
3. **SR=0이면서 진짜 실패**: 안 움직이거나 다른 곳 헤맴 → trajectory 신호로만 구별 가능

→ **모든 새 실험은 multi-metric으로 평가**. SR은 보조 지표.

## 📊 Primary metric: Handoff readiness (sensor에 인계 가능한가?)

| metric | 의미 | 합격선 (잠정) |
|---|---|---|
| **close_once ≤5mm** | 한 번이라도 5mm 안 도달 | ≥ 40% |
| **time_near_5mm** | 250 step 중 5mm 이내 비율 | ≥ 20% (sensor가 grid 찍을 시간) |
| **retreat** = max(dist after t_min) − min_dist | 도달 후 도망 거리 | ≤ 3 mm |
| **approach_signal** = approaching% + holding% − fleeing% | trocar 방향성. + = 보고 있음, 0 = 랜덤 wander | ≥ 0.2 |
| **HANDOFF_OK** = close_once ≤5mm ∧ time_near_5mm ≥20% ∧ retreat ≤3mm | 종합 | ≥ 40% |

**Secondary**: SR (strict 5mm+10°+hold), min_dist median, final_angle. **참고 only**.

분석 도구: `python -m scripts.analyze_trajectory <eval_dir>` (per-episode npz → 위 metric)

## 🏆 Best (잠정) 모델

**`/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_fine/align/checkpoint_10000.pt`** (wandb atsde7s4, 2026-05-11)
- SigLIP2-SO400m@512 frozen + VLANeXt head, **B=16 / lr 1e-4 / 10k step / commit 6a6bd1e + 미커밋 vision_only 분기**
- wandb 실측: loss/total **0.04**, gnorm **1~7 clean**
- **05-13 partial 153ep eval (HANDOFF metric 포함)**:
  - SR strict: **3%** | **HANDOFF_OK: ~4%** (목표 ≥40% ❌)
  - close_once ≤5mm: 16% | ≤10mm: 37%
  - min_dist median: 11.5mm | **retreat median: 10mm** ❌ | **time_near 5mm: 0%** ❌
  - approach_signal **+0.11** (approaching% 49%, fleeing 35%) — trocar 방향성은 있음
  - **결론**: VLA가 trocar 방향으로 움직임은 있지만 5mm 근처 머무름 거의 없음 (Mode A 확정). sensor 인계 사실상 불가능. **hold-aware aux 필요 정당화**

## 🏆 핵심 발견 (2026-05-13)

**`#26 repro_b24` = batch 16→24만으로 명백한 도약** (50ep eval):

| Metric | #1 baseline (153ep) | **#26 repro_b24 (50ep)** | 변화 |
|---|---|---|---|
| SR strict | 3% | **22%** | **7×** ✅ |
| close_once ≤5mm | 16% | **38%** | 2.4× ✅ |
| close_once ≤10mm | 37% | **64%** | 1.7× ✅ |
| min_dist median | 11.5mm | **7.5mm** | 1.5× 가까움 ✅ |
| approach_signal | +0.11 | **+0.27** | 2.5× 더 강한 의도 ✅ |
| approaching% | 49% | **61%** | 더 trocar 방향 ✅ |
| **retreat median** | 10mm | **20mm** | **2× 나빠짐** ❌ |
| **HANDOFF_OK** | 4% | 4% | 동일 ❌ |
| **time_near 5mm** | 0% | 0% | **미해결** ❌ |

**핵심 해석**:
- batch ↑ → **"찾아가는 능력" 강화** (도달, 의도, 거리 모두)
- 그러나 **"머무르는 능력"은 못 잡음** — 오히려 더 강하게 다가갔다가 더 멀리 튕겨남
- Mode A (도착 후 도망) **batch만으로는 미해결** — demo 자체에 "머무름" 신호 없으니 BC가 학습 못 함 (구조적 한계)

## ❌ 실패한 방향 (시간 절약용 기록)

| 시도 | 결과 | 교훈 |
|---|---|---|
| **#27 holdaux hard threshold** (`where` margin 5mm 경계) | step 3183부터 gnorm 폭주, 4914에서 44k → kill | hinge loss의 binary threshold는 batch 경계 진동 → 불연속 gradient |
| **#27v2 holdaux soft ramp + lr 5e-5** | 학습 완료 but eval 14ep: SR 0%, retreat **23mm**, approach **−0.43** | 학습 자체 약함 (loss 0.55 vs #26의 0.04). aux 변형이 main MSE 방해. 게다가 페널티만 줘봐야 demo에 없는 "머무름" 행동은 BC가 학습 못 함 |

## 🚧 진행 중 (2026-05-13)

| Track | GPU | 상태 |
|---|---|---|
| **#28 `repro_b24_ft10mm`** | GPU 1 (PID 1954391) | 13:04 launch (OOM 후 expandable_segments 추가 재시도). `repro_b24/ckpt_10000.pt`에서 resume, **데이터 = approach_00 + 10mm_fine_align (9818 ep, 2배)**, `reset_optimizer_scheduler=true`, lr **5e-6**, warmup 200, max 5000 step. 가설: 10mm 영역 demo가 5mm 근처 신호 풍부 → 머무름 학습 가능성. ETA ~1.5h |

## 🧭 다음 방향성 (시간 1일 남음)

| 후보 | 본질 | 비용 | 예상 leverage |
|---|---|---|---|
| **#28 ft10mm 진행 중** | 데이터 추가 + continue training | 1.5h | 데이터 부족 root cause라면 직접 fix |
| **Hybrid pipeline 통합**: VLA close_once≤5mm 38% → 닿는 순간 sensor 인계 | 시스템 설계 | sensor 모듈 점검 | 머무름 학습 불필요 — VLA 도달, sensor 머무름 |
| **Inference clamp**: 5mm 도달 감지 → action norm 작게 | 추론 시 강제 정지 | 30분 코드 | 학습 불필요, 즉시 검증 |
| **Real demo 추가 학습 (real_align)** | domain transfer | 2h, 위험 | sim → real gap 해소 (시간 risk 큼) |

## ⚠️ 회귀 / 환경 노트
- **GPU 0 dead** (NVML 죽음). GPU 1/2 (3090 24GB) 만 사용
- MuJoCo eval 필수 env: `MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`
- **lr 1e-4 회귀 의심 → 해소**: `repro_b24`가 step 7781까지 loss 0.04 gnorm 2-4 깨끗. 이전 #24 폭주는 working tree 상태/seed 이슈로 보임. lr 1e-4 자체는 atsde7s4-style config에서 OK.
- **단, hold-aware aux + lr 1e-4 조합은 위험**: #27 hard threshold가 step 3200부터 폭주. 5mm 경계 불연속이 batch 내 sample mix와 만나 gradient surface 진동. **soft ramp + lr 5e-5 조합으로 우회 (#27v2)**.

## 🔬 왜 baseline 성능이 구린지 (2026-05-13 정리)
1. **Vision encoder 픽셀해상도 한계**: SigLIP2-SO400m@512 patch 16px → mm 단위 미세 거리 추정 약함
2. **Single view + no depth**: tool_camera 하나로 3D 5mm 정밀도는 본질적 ill-posed
3. **Demo distribution gap**: demo가 5mm 도달 직후 종료 → "도착 후 머무름" 신호 부족 → Mode A (도착 후 도망)이 SigLIP2/ACT/DP 공통으로 발생
4. **→ hybrid pipeline 정당화 근거**. VLA로 mm까지 가는 건 구조적 한계, sensor 인계 필요

---

# 📝 paper contribution framing (논문 방향)

## 가설
**"VLA-only로 mm 정밀도 달성은 어렵다. Classical pose-est + IK도 perception ambiguity (occlusion/변형/조명)에서 실패. 두 도구의 적절한 결합이 정답."**

## 우리 contribution = **Hybrid pipeline 정당화 + 인계 지표 표준화**

1. **VLA의 역할 명시**: approach + coarse alignment (5-10mm 영역까지). hold/mm-precision은 sensor에 인계
2. **인계 지표 (Handoff metric) 정의**: close_once + time_near + retreat + approach_signal. SR 단독 평가의 위험성을 정량적으로 제시 (#8 vs #1 trade-off 사례)
3. **Cross-architecture parity**: SigLIP2 / ACT / DP 모두 Mode A 동일 실패 → VLA family의 공통 천장
4. **Sensor grid search**: 5mm 영역에서 contact-based final-mm 도달 (별도 트랙)

## 방어 포인트
- "VLA의 mm 실패"는 알려진 한계이며, 본 paper의 **입력 조건** (분석된 한계 + 합리적 인계 기준)
- Classical pose-est 비교: short experiment (template matching SR under occlusion) 추가 시 hybrid 우위 명확
- 도달했다가 도망 ≠ trocar를 봄. **머무름이 진짜 grounding 증거** — `approach_signal`, `retreat` metric이 이를 측정

## 약점 (인지)
- Sensor grid 파트가 아직 미실측 → SR 50%+ 보여줄 수 있어야 contribution 완결
- "VLA 인계 후 sensor가 마무리"의 end-to-end SR을 보여줘야 함

---

# 📋 1. Run 결과 표 (#1 ~ #26)

## 데이터 표시 규약 (⚠️ 중요)
- **모든 SR 단독 표기는 "incomplete metric"** — 그 시점 결정 근거로만, 모델 capability 판정 X
- **재평가 필요**: 옛 SR-only 결론에 ❓ 표시. handoff_ok % 측정 안 된 모델은 `--` 로 둠
- 새 run은 반드시 `analyze_trajectory.py`로 handoff_ok / time_near / retreat / approach_signal 산출

| # | Config | Backbone / Adapt | lr | 학습 신호 | eval 신호 (handoff 관점) | 결론 |
|---|---|---|---|---|---|---|
| **1** | siglip2_config (atsde7s4) | SigLIP2-SO400m@512 / frozen | 1e-4 | loss 0.04, gn 1~7 clean | (62ep 부분) close-once<5mm 16%, min_dist 12.1, retreat 8.5, angle 10.6° / handoff_ok **미측정** | **잠정 BEST.** 250ep 정식 eval 진행 중 |
| 2 | dinov3_vits | DINOv3-vits16@224 / frozen | 1e-5 | gnorm noisy | Mode B (progress 3.5mm — 거의 안 움직임) | head 동반 축소 confound. **CLOSED** |
| 3 | dinov3_vitb | DINOv3-vitb16@512 / frozen | 1e-5 | gnorm noisy | progress 0.4mm (안 움직임) | head 동반 축소. **CLOSED** |
| 4 | dinov3_vitl | DINOv3-vitl16@512 / frozen | 1e-4 | gnorm 폭주 | — | lr too high. **CLOSED** |
| 5 | siglip2_lora | SigLIP2 / LoRA r=16 | 1e-5 | loss 정체 KILL@6500 | — | **LoRA axis CLOSED** |
| 7 | siglip2_sensor | SigLIP2 + sensor / frozen | 1e-4 | DIVERGE@3500 | — | proprio scale mismatch |
| **8** | siglip2_sensor_lr5e5 | SigLIP2 + sensor / frozen | 5e-5 | 안정 | close-once<5mm **4%** (#1 16%) / retreat 2.5 / SR<15mm 16% | ⚠️ **SR↑은 운**. mean approach 떨어짐. **sensor axis CLOSED** |
| 9 | siglip2_last4_unfreeze | SigLIP2 last-4 / partial | 1e-5 | 정체 KILL@3278 | — | **backbone adapt CLOSED** |
| 10 | siglip2_ddl | SigLIP2 + DDL / frozen | 1e-4 | — | progress 6.8mm, retreat **84mm** (도망) | position 약화. **DDL CLOSED** |
| 11 | siglip2_sensor_cont | SigLIP2 + 연속 sensor | 5e-5 | KILL (user pivot) | — | **sensor axis CLOSED** |
| 12 | dinov3_vitl_match | DINOv3-vitl16@512 / frozen | 1e-5 | — | min_dist 36mm | encoder swap fail |
| 13 | siglip2_neargoal | SigLIP2 + neargoal 3× boost / frozen | 1e-4 | DIVERGED@4000 | mid-range #1↑ but jitter | **near-goal naive CLOSED** |
| 14 | dinov2_large_match | DINOv2-large@518 / frozen | 1e-5 | — | min_dist 34mm | **encoder swap CLOSED** |
| 15 | siglip2_neargoal_lr5e5 | #13 lr↓ | 5e-5 | underfit | min_dist 36-40mm (#1의 3×) | lr-stability trade-off. **CLOSED** |
| 16 | resnet50_match | ResNet50 conv / frozen | 1e-4 | — | close-once<5mm **0%**, mode collapse | **Mode D**. conv naive CLOSED |
| 17 | resnet50_actstyle | ResNet50 + 2D sincos + 4L TFenc | 1e-4 | 보류 | — | #16 결과로 leverage 작음 추정 |
| 18 | siglip2_in768 | SigLIP2@768 / frozen | 1e-4 | KILL@2545 gn 9000+ | — | token-heavy + lr 1e-4. **CLOSED** |
| 19 | siglip2_localcrop | SigLIP2 + center crop | 1e-4 | KILL@2784 gn 7000+ | — | 동일 패턴 |
| 20 | siglip2_in768_lr5e5 | SigLIP2@768 / frozen | 5e-5 | 10k 완료 (loss 0.94 gn 45) | 영상 "답없" → eval kill | retreat 측정 안 됨, 일단 보류 |
| 21 | siglip2_localcrop_lr5e5 | SigLIP2 + crop / frozen | 5e-5 | 10k 완료 (loss 0.93 gn 25) | 영상 판단 kill | 동일 |
| 22 | siglip2_in768_v2 | SigLIP2@768 / frozen | 5e-5 | 학습 15k+ 진행 중이었음 → KILL (회귀 base) | — | 회귀 base 위라 의미 없음, kill |
| 23 | ~~siglip2_localcrop_v2~~ | — | — | — | — | 처음부터 무산 |
| 24 | siglip2_v2 | SigLIP2@512 / frozen | 1e-4 | KILL@3500 (gnorm 942→49k) | — | **회귀 의심 (#1 정확 config 폭주)** |
| 24b | siglip2_v2_lr5e5 | SigLIP2@512 / frozen | 5e-5 | 학습 완료 15k (loss 0.51 gn 16 clean) | (62ep) **SR 0%, min_dist 17-22mm — 완전 실패** | ⚠️ 회귀 base + lr↓ 조합이 #1 회복 못 함. **KILL eval** |
| 25 | siglip2_v3_holdaux | SigLIP2 + hold-aware aux | 5e-5 | 시작 안 함 (회귀 base 위라 무의미 판정) | — | **CANCELLED** |
| **26** | **siglip2_repro_b24** (GPU 2) | SigLIP2@512 / frozen | 1e-4 | **step 4012/10000 loss 0.22 gn 3-6 clean** | — | atsde7s4 1:1 + B=24. **#24 폭주 지점 통과 → 회귀 의심 해소** |
| **27** | **siglip2_holdaux** (GPU 1) | SigLIP2@512 + hold-aware aux / frozen | 1e-4 | step 19 시작, loss 2.4 gn 11 정상 | — | atsde7s4 1:1 + `hold_threshold_mm=5` **단일 변수**. Mode A (도착 후 도망) 직접 fix |
| 28 | siglip2_resume25k (config 준비) | SigLIP2@512 / frozen | 1e-4 | pending — `resume_path=#1 ckpt_10000`, max_steps 25000 | — | undertrained 가설 검증. 4.5hr |

## 표 보는 법
- **잠정 결론 = 1번 best**. 나머지는 다 진행/CLOSED
- handoff metric 측정 안 된 옛 row는 결론 잠정 — `analyze_trajectory.py`로 재평가 가능 (npz 신포맷 있어야)
- 새로 학습할 때는 step 2200 통과까지가 critical zone

---

# 📊 2. 실패 모드 + 효과 데이터

## 실패 모드 (handoff 관점 재정의)
| Mode | 증상 | trajectory 신호 | 대표 |
|---|---|---|---|
| **A** approach OK + hold fail | 다가가는데 정렬 안 됨, 도착 후 도망 | close_once<5mm 있지만 time_near 낮음, retreat 큼 | #1, ACT, DP |
| **B** approach fail | 거의 안 움직임 | min_dist 30+mm, close-once 0% | DINOv3 시리즈 |
| **C** divergence / jitter | 학습 중 gnorm 폭주 → 영상 떨림 | min_dist 들쑥날쑥, retreat 큼 | #13 |
| **D** mode collapse | 평균 방향만 미세 움직임 | close-once<5mm 0%, approach_signal ≈ 0 | #16 ResNet50 |

⚠️ **B mode 함정**: 안 움직이면 perturbation 그대로 → angle 만 보면 정답 분포 → angle-only SR 100% 착시 (#2, #3). pos-aware metric (close_once, min_dist) 필수

## #1 SigLIP2 vs DINOv3 4-way (compare_eval, 부분 eval)
| 지표 | #1 SigLIP2 | DINOv3_vits | DINOv3_vitb | DINOv3_vitl |
|---|---|---|---|---|
| progress median (mm) | **33.0** | 3.5 | 0.4 | 10.7 |
| close-once <5mm | **16%** | 0% | 0% | 14% |
| close-once <15mm | **64%** | 15% | 14% | 43% |
| final_dist median | **24.9** | 36.3 | 36.7 | 30.2 |
| retreat median | 8.5 | 3.6 | 8.9 | 7.9 |
| strict SR <5mm+10° | 0% | 0% | 0% | 0% |
| pos-only SR <5mm (참고) | 6% | 0% | 0% | 7.1% |
| angle-only SR (참고) | 44% | 100%* | 92.9%* | 0% |

\*안 움직여 perturbation 분포 그대로 — angle SR 함정

→ #1 SigLIP2가 close-once / progress 모든 면에서 dominant. retreat는 DINOv3_vits가 작지만 그건 **애초에 안 움직였기 때문**

## #1 vs 주요 변형 (handoff 관점 재해석)

**#8 sensor_lr5e5**: SR<15mm 4→16% (4×) **but close-once<5mm 16→4% (1/4×)**. retreat 8.5→2.5 (좋음) / angle 10.6→8.0 (좋음). → **운 좋게 한두 ep에서 hold 박힌 sensor 효과로 SR↑, 정작 접근 능력은 1/4로 하락**. SR-only로 보면 best로 오해 가능. **sensor axis CLOSED**

**#10 DDL**: close-once<15mm 64→17%, progress 33→6.8mm, retreat 11→**84mm** (대도주). DDL position 약화 시그니처. **CLOSED**

**#13 neargoal**: SR<15mm 4→12%, close-once<8mm 20→38%, min_dist 12.1→10.8. but close-once<5mm 16→10%. divergence 후 후퇴 발생. **CLOSED**

## #27 Hold-aware aux: 가설 + 메커니즘

**문제 (Mode A)**: #1은 도착 후 도망 → aux_distance_loss가 hold-state에서도 progress 요구하기 때문

**기존 (atsde7s4, hold_thr=0)**:
```python
aux = relu(pred_dist - cur_dist + 0.1mm)
```
cur_dist=2mm 이미 가까이 → pred=1.9mm 더 가야 loss=0. **머무름이 패널티** → 오버슈팅 → 도망

**#27 (hold_thr=5)**:
```python
margin_eff = 0 if cur_dist ≤ 5mm else 0.1mm
aux = relu(pred_dist - cur_dist + margin_eff)
```
5mm 영역에서 margin=0 → pred=cur 일 때 loss=0 (= 머물러도 OK). 멀어질 때만 패널티

**예상 효과**: retreat ↓ (도망 안 함), time_near_5mm ↑ (머무름), close_once<5mm 유지 또는 ↑. SR도 부수적으로 ↑ 가능

**예상 실패 시나리오**: 5mm 영역에서 멈춤 → trocar 외곽에서 정지 (정렬 정밀도 부족). 그러면 hold OK여도 strict SR 안 오름. handoff 관점에서는 좋음

---

# 🚫 3. CLOSED 축 (재시도 금지)

| 축 | 시도 | 결론 |
|---|---|---|
| Encoder swap (head 동일) | #12 vitl, #14 large | min_dist >30mm. SigLIP2가 dominant |
| Backbone adaptation | #5 LoRA, #9 last-4 | loss 정체. frozen이 정답 |
| Sensor proprio | #7, #8, #11 | SR↑은 운, mean approach 떨어짐 |
| DDL (Direction Decoupled Loss) | #10 | position magnitude 약화 |
| Head 축소 | #2, #3 | head capacity가 결정 인자 |
| Low input resolution | #2 vits@224 | receptive field 부족 |
| Conv backbone naive | #16 ResNet50 | Mode D collapse |
| Near-goal naive | #13 lr1e-4 발산, #15 lr5e-5 underfit | lr-stability trade-off |
| Vision spatial naive @ lr 1e-4 | #18 in768, #19 localcrop | token-heavy + lr 1e-4 폭주 |

---

# 📁 4. 부록

## 4-1. Best #1 wandb 실측 (atsde7s4, 2026-05-11 16:43)
```
Config: B=16, max_steps=10000, grad_acc=1, lr 1.0e-4, warmup 500
seed: 2026, commit: 6a6bd1e + 미커밋 변경(vision_only path)
vision_only + SigLIP2-SO400m@512 frozen
aux_distance(w=0.2, margin 0.1, near_goal boost 5×), DCT 0.1
proprio_dim 6 (no sensor), view_mode single
```

### wandb history (gnorm/loss 시계열)
| step | grad_norm | loss/total |
|---|---|---|
| 541 | 5.95 | 0.87 |
| 1568 | 4.67 | 0.38 |
| 2722 | **3.03** | **0.08** |
| 4155 | 1.64 | 0.040 |
| 7526 | 1.26 | 0.018 |
| 9441 | 2.43 | 0.031 |

**gnorm 1~7 clean**, loss/total 0.04 수렴. SR 4%는 학습 부족이 아니라 **모델/태스크 capability ceiling** 가능성.

### 회귀 의심 (2026-05-13)
- #24: 같은 commit/config/seed로 재현 → step 2200 폭주
- #26 (현재 진행): 동일 condition + B=24 — 만약 clean하게 끝나면 #24가 outlier, 또 폭주하면 코드/환경 회귀 확정

## 4-2. lerobot 자산 (Phase II 트랙)
- `dataset/convert_to_lerobot.py` + `dataset/lerobot_sim/` (4916 ep)
- `Run_Train_Lerobot.sh`, `scripts/sim_eval_lerobot.py`, `Run_Eval_Parallel.sh lerobot_act|lerobot_dp`
- **ACT 80k**: `outputs/train/lerobot_act_align_20260510_2222/checkpoints/080000/pretrained_model` ✅
- **DP 80k**: pretrained_model **삭제됨** (eval mp4만 남음). re-train ~17hr 필요

### 과거 부분 eval (참고용, SR-only)
| Eval | step | ep | SR | 주의 |
|---|---|---|---|---|
| ACT eval_act_20k v1 | 20k | 28/50 | 17.9% | SR-only, retreat 미측정 |
| ACT eval_act_20k v2 | 20k | 39/50 | 2.6% | 분산 큼 → 운 영향 큼 |
| DP shard0+1 | 80k | 30/50 | 17% | incomplete |

→ SR 기준 ~17%. 단 retreat/handoff 미측정 → cross-architecture parity 결론은 영상 인상 기반 (정량 미확정)

## 4-3. 데이터 규모
- 4916 ep × 평균 128 frame = ~**596k 샘플** (window 적용)
- 1 epoch step: B=12 → 50k, B=16 → 37k, B=24 → 25k
- #1 (10k step) ≈ **0.2 epoch만** 학습됨

## 4-4. Hardware
- GPU 0 **dead** (NVML 죽음). GPU 1/2 (3090 24GB) 만 사용
- VRAM cap: B=16 @ 1024 tok ≈ 14GB / **B=24 @ 1024 tok ≈ 23.4GB (tight, expandable_segments 필요)** / B=12 @ 2304 tok = OOM, B=10 = OK

---

# ⚙️ 5. 운영 룰 + 명령어

## 새 실험 체크리스트
- [ ] #1 baseline yaml에서 시작
- [ ] **딱 한 가지만** 바꿔라 (두 개 이상이면 두 run으로 분리)
- [ ] head depth/hidden/queries 안 건드림 (#2~#3 실수)
- [ ] lr: default 5e-5 (안전), 1e-4는 새 code에서 폭주 가능
- [ ] grad_acc=1, B 최대 fit
- [ ] eval은 **반드시** 250ep 정식 + `analyze_trajectory.py` handoff metric 산출
- [ ] **SR 단독 비교 금지** — multi-metric (close-once / time_near / retreat / approach_signal) 같이 기록

## 핵심 명령어
```bash
# Train (UUID로 GPU 명시)
CUDA_VISIBLE_DEVICES=GPU-<uuid> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /home/yohan/miniconda3/envs/VLANeXt/bin/python -m scripts.train \
  --config config/<yaml> > train_<name>.log 2>&1 &

# Eval single shard (250ep grid)
CUDA_VISIBLE_DEVICES=GPU-<uuid> MUJOCO_GL=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  /home/yohan/miniconda3/envs/VLANeXt/bin/python -m scripts.sim_eval_align_only \
  --config config/sim_eval_align_config.yaml --checkpoint <ckpt.pt> \
  --train-config config/sim_train_align_config.yaml \
  --shard-id 0 --num-shards 1 \
  --max-steps 250 --eval-seed 2026 --perturb-mode grid \
  --xy-steps 5 --z-steps 2 --angle-steps 5 --repeats 1

# Handoff-aware analysis (per-step npz 필요 — 신포맷)
python -m scripts.analyze_trajectory <eval_dir>

# Compare 두 모델 (multi-metric)
python -m scripts.compare_eval <dir1> <dir2> ... --names ... --out report.md
```

## 운영 가이드
- **Eval sanity**: 10 ep로 빠르게 abort 가능. 단 최종 결론은 250 ep + handoff metric
- **무조건 multi-metric**: SR 4% vs 6% 단일 비교 금지. close_once / retreat / time_near 같이 봄
- **Skip-fast**: step <2k 에 sustained gnorm > 500 → 즉시 kill
- **분기 parity**: train.py deepspeed/non-deepspeed 두 model() 호출에 인자 동시 추가 (memory: `feedback_train_branch_parity`)
- **새 npz 포맷**: 2026-05-13 이후 eval부터 traj_ep*.npz에 per-step `dist_mm / lateral_mm / angle_deg` 포함. 이전 eval은 trajectory analyzer가 skip
