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

## ✅ #28 / #29 ft10mm continue-training 결과 (2026-05-13 16:00)

**결론: ft10mm + continue-training 유효성 확인. lr 1e-5 ≈ lr 5e-6 (거의 동급)**. Mode A (도착 후 도망) **확연히 개선**되나 5mm 안에서 **머무름은 여전히 못 함** → sensor 인계 필요 정당화 강화.

### 학습 설정 공통
- resume from `repro_b24/ckpt_10000.pt` (loss 0.04 atsde7s4급)
- 데이터 = `approach_00` + `10mm_fine_align_00_tip2/collected_data_merged` (9818 ep)
- `reset_optimizer_scheduler=true`, warmup 200, max_steps 5000, B=24
- #28: lr **5e-6** (보수적), #29: lr **1e-5** (천장)

### 3-way handoff 비교 (25ep, repeats=1 grid 동일)

| Metric | **#26 baseline** | **#28 (lr 5e-6, 34ep mixed)** | **#29 (lr 1e-5, 25ep clean)** | 비고 |
|---|---|---|---|---|
| SR strict | 22% | **50%** | 36% | strict는 angle/hold 다중 통과 — 영상상 정렬돼도 fail로 빠지는 케이스 있음 |
| Handoff_OK | 4% | 0% | **4.0%** | 3개 동시 통과 (5mm + time_near 20% + retreat≤3mm) — 둘 다 time_near 미통과로 거의 0 |
| close_once ≤5mm | 38% | 47% | 44% | 한 번이라도 5mm 진입 |
| min_dist median | 7.5mm | 8.0mm | **6.3mm** ✨ | 가장 근접 |
| **retreat median** | **20mm** | **5.4mm** ✨ | 6.7mm | **Mode A 도망 거의 사라짐** |
| time_near_5mm | 0% | 0% | 0% | ❌ 5mm 안 머무름 여전 미해결 |
| time_near_8mm | — | 5.6% | **19.6%** ✨ | 8mm 영역에선 머무르기 시작 |
| approach_signal | +0.27 | +0.55 | +0.42 | 정량 모두 trocar 방향 의도 강함 |

### 핵심 해석
- **batch 24 + ft10mm continue가 retreat을 20mm → 5mm로 줄임** = 가장 큰 진전. Mode A 도망 패턴 거의 해소.
- **lr 5e-6과 1e-5 거의 차이 없음** (#29가 min_dist/time_near_8mm 약간 우세, #28이 retreat/SR strict 약간 우세). 5e-6도 충분히 학습됨 → 도메인 shift 우려 기우.
- **time_near_5mm 0% 그대로** → 5mm 안 머무름은 BC + diffusion 구조적 한계 추정. 데모 자체에 "5mm 머무름" 신호 부족.
- **각도 미스 케이스 잔존** — 영상상 위치 완벽한데 angle 통과 실패로 SR strict 빠짐. SR을 판단지표로 안 보는 이유 재확인.

### 남은 한계 / 다음 트랙
1. **time_near_5mm 0% 미해결** → ① 5mm hold 데모 추가 수집, ② **sensor (1D contact) 인계 트랙으로 mm 영역 마무리** ← user 결정 방향
2. **각도 정밀도 부족** — vision-only 픽셀해상도 한계 + demo angle 분산 작음 가능성. Sensor 활용 시점에서 angle도 sensor pattern으로 검증 고려
3. **시간 남으면**: real_align 데이터 추가 continue → real eval 환경에서 sim→real gap 확인

### 산출물
- `#28 ckpt`: `/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_fine_align/checkpoint_5000.pt`
- `#29 ckpt`: `/data/public/NAS/VLANeXt/checkpoints/VLANeXt_SigLIP2_repro/b24_ft10mm_lr1e5_fine_align/checkpoint_5000.pt`
- 둘 다 sim→real transfer 출발점 candidate. 우열 미미하니 둘 중 아무거나 (retreat 우세인 #28 권장)

### ⚠️ GPU 인덱싱 함정 (자주 막힌 곳, 절대 잊지 말 것)
- **GPU 0 (PCI 0000:24:00.0) dead** → `nvidia-smi`는 인덱스 1, 2만 보여줌 (dead 0번은 enum에서 빠짐).
- 하지만 `CUDA_VISIBLE_DEVICES`의 인덱스는 **CUDA runtime이 NVML로 보는 enum 기준** — 따라서:
  - `CUDA_VISIBLE_DEVICES=0` → 실제 **nvidia-smi 1번** (한 칸 밀림! ft10mm 도는 GPU 잡혀 OOM)
  - `CUDA_VISIBLE_DEVICES=1` → 실제 **nvidia-smi 2번** (idle, 우리가 원하는 것)
  - `CUDA_VISIBLE_DEVICES=2` → 존재 안 함, `torch.cuda.is_available()=False`
- `Run_Eval_Parallel.sh`의 `GPUS=` 환경변수도 동일 규칙 적용 (CUDA enum index).
- NVML probe는 망가져서 `Can't initialize NVML` warning 늘 나오지만 CUDA runtime은 정상.
- **검증 명령**: `CUDA_VISIBLE_DEVICES=<n> python -c "import torch; print(torch.cuda.mem_get_info())"` 로 free memory 봐서 idle GPU 맞는지 확인 후 학습 띄울 것.

## 🌐 Sim → Real transfer 계획 (P2 결과 보고 분기)

### Real 데이터 검토 결과 (2026-05-13)
- 위치: `/data/public/NAS/VLANeXt/dataset/real_align/collected_data_real_{0430,0501,0502,0503,0503_2,0504}` 6 폴더 × 1000 ep = **6000 ep**
- 키 구조 sim과 동일 (`tool_camera`, `ee_pose`, `sensor_dist` ...). length 더 김 (~140 step)
- **`action` vs `action_sim` 두 키 존재** — rotation convention 다름
- **정규화 호환성** (sim hardcoded ±값 vs real p99):

| dim | sim ± | real `action` p99 | real `action_sim` p99 | 호환 |
|---|---|---|---|---|
| dx/dy/dz | 0.37 | ~0.30 | ~0.30 | ✅ |
| rx | 0.0025 | 0.0017 | 0.0014 | ✅ |
| **ry** | **0.0007** | **0.0014** | **0.0005** | `action` ❌ 2× over / `action_sim` ✅ |
| rz | 0.007 | 0.0021 | 0.0019 | ✅ |

→ **반드시 `action_sim` 키 사용**해야 모든 dim이 sim 정규화 안에 들어옴. clip 없이 학습 가능.

- 실제 로봇 eval은 별도 환경에서 (`Run_Real_Eval_Align.sh` / `Run_Real_Eval_Approach.sh` 류), 본 시스템에서는 학습 + sim eval만
- sensor 무관 (`use_sensor: false`)

### Phase 계획

| Phase | 시간 | 내용 |
|---|---|---|
| **P1** (지금~14:50) | ft10mm 학습 종료 대기 | |
| **P2** | ~15:00 | ft10mm sim eval 50ep + handoff metric. **검증 대상 = "continue training이 동작하는가?"** |
| **P3 결과 분기** | | |
| ┗ **시나리오 A**: ft10mm > repro_b24 (continue 효과적) | 15:30~16:30 | 같은 패턴으로 real 추가: ft10mm ckpt + sim + real (1-2 폴더). lr 5e-6 유지 |
| ┗ **시나리오 B**: ft10mm ≈ repro_b24 (continue 영향 없음, weight 거의 안 움직임) | 15:30 | lr 너무 보수적 → 1e-5 또는 2e-5로 재시도 |
| ┗ **시나리오 C**: ft10mm < repro_b24 (continue가 망친 것) | 15:30 | 처음부터 sim+real from-scratch 10k step. continue 패턴 폐기 |
| **P4** | 17:00~ | sim eval로 forgetting 확인 → ckpt 옮겨서 real eval (별도 환경) |

### 체크포인트 활용 설계 (현 상태 + 조정 가능 요소)

| 요소 | 현재 | 향후 조정 가능성 |
|---|---|---|
| `reset_optimizer_scheduler=true` | ✅ 적용 | fresh Adam + new cosine, 옛 스케줄 lr=0 문제 회피 |
| ckpt CPU load + del + empty_cache | ✅ 적용 | OOM 안전 (B=24 가능) |
| Peak LR | 5e-6 | P2 결과에 따라 1e-5~2e-5로 증가 가능 |
| Warmup | 200 step | fresh Adam moment 추정에 필요. 길게 가도 안전 |
| Multi-path data_root | ✅ 지원 | real 추가는 path 한 줄만 |
| **Real action_sim 키 분기** | ❌ 없음 | **real 추가 시 sim_act_align.py 분기 필수** |
| Sampling weight 제어 | ❌ 없음 | 현재 episode 수 비율 = 자연 sim-우세 (sim 9818 : real 1-2k) |
| EMA / weight averaging | ❌ 없음 | continue 안정성 ↑ 옵션이지만 우선순위 낮음 |

## 🧭 다음 방향성 (시간 1일 남음, ft10mm 검증 완료 후)

**ft10mm 결과 기반 우선순위**:

| 후보 | 본질 | 비용 | 우선순위 |
|---|---|---|---|
| **1D sensor 인계 트랙 통합** | close_once 48% / retreat 5mm 도달했으니 그 순간 sensor가 mm-precision으로 마무리 | sensor 모듈 + 인계 트리거 (~3-4h) | ★★★ user 명시 방향 — VLA stay 학습 불필요 |
| **Inference clamp / action gating**: 5mm 도달 후 action norm 작게 | 추론 시 강제 안정화 | 30분 코드 | ★★ 즉시 검증 가능, sensor 미통합 시 stop-gap |
| Real demo 추가 continue (`ft10mm + real_align`) | sim → real gap 해소 | 1.5h + real eval 환경 이동 | ★★ time_near_5mm은 어차피 0이라 real eval 정량 비교 어려움 |
| 5mm hold 데모 새로 수집 (5mm 근처에서 정지하는 trajectory) | demo 분포의 직접 fix | 데이터 수집 시간 큼 | ★ 시간 risk, sensor 트랙이 더 직접적 |

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

---

# 🔧 6. Sensor Handoff Controller (2026-05-13)

**한 줄**: VLA가 ≤5~10mm 까지 데려가면 거기서부터는 비전 모델 없이 **1D ray sensor + 트로카 축 정보**만으로 격자 탐색해 mm 정렬 + 축 방향 push. 추가 학습 0, 9 ep 중 첫 SUCCESS (Ep1 insertion depth 2.95mm).

**Architecture 위치**: Meca500(6-DoF)이 **트로카 정중앙 진입 + holding** 담당, intra-eye 각도는 향후 RCM end-tool이 담당. handoff는 임시 fallback이 아니라 **분업 architecture의 부품**.

## 알고리즘 (`scripts/sensor_handoff.py` v4)

```
Stage 0 trigger: min_dist ≤ 15mm AND not success
Stage 1 좌표계:  axis_dir = (depth - entry)/norm; u,v = axis_dir 수직 basis
Stage 2 coarse: u-v plane sweep ±3mm/1mm = 7×7 = 49 trial,
                각 trial = snapshot → apply_delta_ee(40 sim_step) → measure(sensor, lateral) → restore
                score = sensor − max(0, lateral−1.5) × 5
                best (du,dv) commit (실행 1번)
Stage 3 fine:   ±0.9mm/0.3mm 동일 방식
Stage 4 판정:   (sensor ≥ 25 or -1) AND lateral < 1.5 → ALIGNED
Stage 5 push:   axis_dir 방향으로 8mm / 3 step
* 회전은 모든 stage에서 0 고정 — RCM이 담당할 자유도라 분업 일치
```

## 탐색 dimension 요약

| stage | dim | trial | rotation |
|---|---|---|---|
| Coarse | u-v plane (트로카 단면), ±3mm/1mm | 49 | **0** |
| Fine   | u-v plane, ±0.9mm/0.3mm           | 49 | **0** |
| Insertion | axis_dir 1축, 8mm/3step          | 3  | **0** |

## v3→v4 핵심 버그 픽스

1. **Snapshot/restore** (qpos+qvel+ctrl+time 통째로) — sweep try & restore
2. **Trocar-axis aware insertion** — world-Z로 push했다가 depth=-15mm 역행 → entry→depth 축 사용
3. **sim_steps=40 통일** — sweep 40 / commit 60 mismatch 시 예측 32 → 실제 7 발산. 통일 후 commit이 예측과 < 0.5mm 일치

## 결과 (9-ep, ft10mm ckpt5000, 3×3 grid)

| Ep | VLA 종료 sensor/lat | Stage4 ALIGNED? | 결과 |
|---|---|---|---|
| **1** | 9.5 / 3.5 | ✓ (32.4/1.19) | **SUCCESS[handoff]** depth 2.95mm |
| 2 | 1.3/8.8 | ✗ (lat 8.8 outside ±3mm window) | FAIL |
| 4 | 19.3/4.4 | ✗ (false through-hole 정확히 reject) | FAIL |
| 5,8 | VLA success | skipped | SUCCESS[dist] (VLA) |
| 3,6,7,9 | min_dist>20mm | trigger 안 됨 | FAIL |

**SR 3/9 = 33.3%** (VLA 단독 2 + handoff 1). Sweep prediction ↔ commit 차이 0.00mm (v4 확정).

## 한계

1. VLA 사정거리 의존 — 9 ep 중 4 trigger 안 됨. VLA 자체 개선이 우선
2. ±3mm coarse window 부족 케이스 (Ep2 lat 8.8mm) — 확대 시 ep당 +25s
3. Insertion 중 lateral drift (1.19 → 4.6mm) — IK가 orientation 유지 못함. per_step 1mm + sensor abort 개선 후보
4. Real deploy에는 snapshot/restore 불가 — digital twin live sync 필요 (미구현)

## 사용

```bash
python -m scripts.sim_eval_align_only --config ... --checkpoint ... \
  --handoff --handoff-trigger-mm 15
```
영상: `handoff_videos_v4/episode_0001_success.mp4`

## 핵심 파일
- `scripts/sensor_handoff.py` (v4) — controller
- `scripts/probe_sensor.py` — sensor signature 검증
- `scripts/sim_eval_align_only.py` — `--handoff` CLI (VLA loop 끝나면 호출, `not success` 가드)

---

# 🧪 7. Real-data Fine-tune (2026-05-13 시작)

## 동기

Sim grid SR 33% 천장 — sim-to-real gap이 가장 큰 의심. Real 데이터(6000 ep, digital twin으로 같은 perturbation 정책) 가 모이는 상태라 sim ckpt에 real을 적응시켜 본다.

## 데이터 inventory + 검증한 사실

**Real**: `/data/public/NAS/VLANeXt/dataset/real_align/collected_data_real_0430~0504` = **6,000 ep × ~175 step**
**Sim (best)**: `dataset/fine_align/10mm_fine_align_00_tip2/collected_data_merged` = **~5,000 ep**

### 좌표/회전 분석

| 항목 | SIM | REAL | 처리 |
|---|---|---|---|
| ee_pose pos | MuJoCo world frame ~(200, 168, 130)mm | Mecademic base ~(115, 74, 257)mm | 100mm offset 있음. **proprio는 raw로 들어가서 OK** (model이 visual + delta로 학습; 정규화 안 됨) |
| ee_pose rot raw | mujoco extrinsic XYZ | mecademic intrinsic XYZ | `infer_convention(path)` 자동 감지 → `mujoco_to_mecademic_euler` 변환. **변환 후 잔차 ~2-5°** 남지만 action delta는 둘 다 작아서(<0.003 rad) 영향 작음 |
| action dpos max | ~0.20mm (p99 0.18) | **~0.56mm (p99 0.42, Y가 큼)** | normalize range ±0.37 → real Y 약 1~3% 클리핑. 우선 그대로 학습 (ckpt resume 효과 살림) |
| action drot max | ~0.003 rad | ~0.003 rad | range ±0.0025~0.007 충분 |
| sensor_dist | 0~28mm (valid 100%, <5mm 64%) | **0~239mm (valid 100%, ≥20mm 62%)** | real ray-cast가 phantom 너머도 잡음. **≥5mm는 신뢰 불가**. 첫 실험은 sensor proprio off (`use_sensor: false`) |
| aligned_qpos / target_entry_world | 동일 | 동일 | 같은 물리적 정렬 의도. 좌표 origin만 다름. |
| needle_tip_pos / trocar_entry_pos | MuJoCo body pos (정확) | sim ray-cast를 commanded pose에 적용 (digital twin 동시 가동) → **노이즈 있을 수 있음** | **aux_distance_loss OFF** (real supervision 부정확) |

### 핵심 관찰

1. **회전 변환 자동 (infer_convention)**: 경로에 `real` 포함되면 mecademic 가정. sim_act_align.py 가 이미 multi-path + per-path convention dispatch 지원 → **dataset loader 코드 수정 없음**.
2. **gripper auto-drop**: 7-dim action/ee_pose → `[:, :6]` 자동.
3. **proprio 정규화 안 됨**: model 입력에 raw mm/rad 그대로. offset이 100mm이지만 visual + delta 학습이므로 큰 영향 없을 것이라 판단.
4. **Sensor 신뢰도**: real은 5mm 이내만 신뢰. 학습에는 일단 빼고 (`use_sensor: false`), proprio sensor 추가는 baseline 본 뒤 결정.

## 2026-05-13 실험: real-only vs co-train (2 GPU 동시)

| 실험 | GPU | yaml | data | start ckpt | lr | aux |
|---|---|---|---|---|---|---|
| **real_only** | 0 | `sim_train_align_siglip2_b24_ft10mm_real_only_config.yaml` | real 6000 ep | ft10mm/ckpt5000 | 2e-5 | off |
| **cotrain**   | 1 | `sim_train_align_siglip2_b24_ft10mm_cotrain_config.yaml`   | sim 5000 + real 6000 (≈1:1.2) | ft10mm/ckpt5000 | 2e-5 | off |

**공통**: B=24, max_steps=5000, save_interval=1000, frozen SigLIP2-SO400m@512, vision-only, single-view(tool_camera), no aug, history=8 future=8, lr=2e-5 (memory `feedback_learning_rate_ceiling` 의 1e-5 ceiling 약간 위 — fine-tune에선 action head 적응 위해 약간 높이되 5e-5는 안전한 위)

**Resume rationale**: ft10mm/ckpt5000 가 현재 sim best (SR 33% on grid). 처음부터 학습하지 않고 sim 지식 위에 real 적응.

**비교 지점**:
- 1000/2000/3000/5000 step 마다 ckpt 저장 → sim grid eval (handoff metric)
- 가설: real_only는 sim 잊을 위험, cotrain은 더 안전. 결과 보고 결정.

## 운영
- 학습 PID 시작 후 약 1~2일. log: `logs/train_ft10mm_real_only.log`, `logs/train_ft10mm_cotrain.log`
- 모니터: gnorm 폭주(>500 지속) 시 즉시 kill. lr 절반.

## 결과 요약 (20-ep grid: xy=2 z=1 angle=5 — angle ±25° 포함 어려운 grid)

| ckpt | 학습 설정 | SR | close ≤5/8/10mm | min_dist med | 평가 |
|---|---|---|---|---|---|
| ft10mm baseline | sim only | (pending eval) | — | — | — |
| **cotrain/ckpt5000** | sim+real 1:1.2, lr 2e-5 × 5k | 5% | 0/0/10% | **21.86mm** | sim 보존, real 흡수 |
| **real_only/ckpt5000** | real only, lr 2e-5 × 5k | **0%** (0/19) | — / dist 40-88mm | — | **catastrophic forget** ❌ |
| **progressive_real/ckpt2000** | cotrain ckpt → real only, lr 5e-7 × 2k | 5% | 0/0/10% | **21.88mm** | cotrain과 **byte 단위 동일** (weight 변화 0) |
| **cotrain_lr_low/ckpt3000** | sim+real, lr 5e-6 × 10k (진행 중) | 5.9% (17ep) | **6/24/24%** | **20.4mm** | ⭐ cotrain보다 명백 개선 |

### 결론

1. **real_only 폐기**: lr 2e-5 × 5k이 real action 분포(0.5mm)로 collapse → sim에선 overshoot. action delta range 차이(sim 0.2mm vs real 0.5mm)가 직접 원인.
2. **cotrain baseline**: sim 22mm에서 stuck. close_once 5mm 0%. 어려운 grid 천장이라 추가 학습 여지 있음을 lr_low가 입증.
3. **progressive_real 무의미**: lr 5e-7 × 2k는 lr×step 곱이 너무 작아 weight 변화 0. wandb loss curve 낮아 보이지만 cotrain과 동일 행동. → memory `feedback_loss_eval_decoupling` 박힘.
4. **cotrain_lr_low (진행 중)가 첫 의미 있는 개선 신호**: min_dist 22mm → 20mm, close_once 8mm 0% → 24% — lr 5e-6 × longer training이 cotrain ckpt를 넘어섬. ckpt5000/10000도 평가해야.

### 진행 중 / TODO
- **cotrain_lr_low** 학습 step ~4.7k/10k (진행 중) — ckpt5000/10000 다 평가
- ft10mm baseline 같은 grid 측정 (sim 천장 확정)
- 진짜 효과 검증은 **real robot deploy** (`digital_twin/real_eval_align.py`) — sim 비교는 noise 만

### 운영 lesson (memory에도 박음)
- **SR 단일 비교 금지** — multi-metric 항상 (HANDOFF_OK / close_once / min_dist median). progressive_real이 SR 동일했지만 multi-metric에서 weight 변화 0 확인
- **lr × step 곱이 너무 작으면 학습 무의미**: progressive_real 5e-7 × 2k = 1e-3 — 의미 있는 weight 변화에 100× 부족
- **이 grid가 어려운 perturbation 모음** (angle ±25°) — baseline 자체가 5% 천장. 다른 grid에서 cotrain 22ep 슬라이스는 18.2% (앞쪽 쉬운 case 운). grid 일관 유지 필수

---

## 2026-05-14 sim-only 정밀도 푸시 (real mixing 폐기, BC-loss 변형 grid search)

**Why pivot**: real cotrain은 sim 보존만 가능, 정밀도 push는 sim 단독으로. 목표 **close_once 5mm ≥ 70% + time_near 5mm hold**.

### Baseline reset
- 출발 ckpt: `b24_ft10mm_fine_align/checkpoint_5000.pt` (sim only, aux w0.2/scale 5mm/boost 5x, lr 5e-6 × 5k)
- 위 cotrain baseline (close_5mm 0%, min_dist 21.86mm) 대비 sim-only는 미측정이지만 trace는 동일 영역으로 추정.

### 시도한 4가지 변형 (모두 20-ep 동일 grid xy2 z1 angle5)

| 가설 | config | resume | 핵심 변경 | step | SR | close 3/5/8/10mm | min_dist median | time_near 5mm |
|---|---|---|---|---|---|---|---|---|
| **aux_strong** ⭐ | `..._aux_strong_config.yaml` | ft10mm/5k | aux w0.5, scale 2mm, boost 10x, lr 1e-5 | **5000** | **5%** | **20/25/25/25%** | **17.99mm** | 0% |
| aux_strong | (same) | ft10mm/5k | (same) | 10000 | 0% | 15/25/25/25% | 18.89mm | 0% |
| aux_xstrong | `..._aux_xstrong_config.yaml` | ft10mm/5k | aux w1.0, scale 1mm, boost 20x, lr 1e-5 | 5000 | 5% | 10/20/25/25% | 18.36mm | 0% |
| aug_long | `..._aug_long_config.yaml` | ft10mm/5k | + photometric/crop aug, lr 5e-6 | 5000 | **0%** | 0/0/0/0% | 28.21mm | 0% |
| DDL | `..._ddl_config.yaml` | aux_strong/5k | + direction_decoupled (mag w1.0, dir w0.5), lr 5e-6 | 2500 | 0% | 10/20/25/25% | 17.94mm | 0% |

### 결론
1. **승자**: `aux_strong/ckpt5000` (close_5mm 25%, min_dist 17.99mm) — cotrain baseline 대비 +25pp / -3.87mm. **best sim fine-align 확정**.
2. **aux 더 강하게 (xstrong) ≠ 개선** — w0.5/scale 2mm/boost 10x가 sweet spot. 더 극단(w1.0/scale 1mm/boost 20x)은 over-supervision, close_5mm 25%→20%.
3. **aux_strong 5k→10k도 평탄** (25%→25%, min_dist 17.99→18.89) — aux 단독 BC fit은 5k에서 saturate.
4. **augmentation 5k는 오히려 망침** (close_5mm 0%) — fresh aug + warm-start resume이 ckpt 분포 무너뜨림. aug는 더 길게 가야 회복 가능했을 듯.
5. **DDL 효과 미미** (close_5mm 20%, min_dist 17.94 — aux_strong과 같은 수준). magnitude supervision 추가만으론 over-shoot 안 잡힘.
6. **공통 천장**: 모든 변형 close_5mm 20-25% / min_dist ~18mm / **time_near 5mm 0%**. 5mm 도달은 하나 hold 못 함 (over-shoot).

### 진짜 병목 (다음 axis 후보)
- **Vision spatial resolution (user 가설)**: patch16 단위 뭉개짐 → 5mm 이내 fine 정렬 정보 부재. 학습 loss로 안 풀림. 후보:
  - input resolution 키우기 (256 → 384/512)
  - tool-camera local crop (needle tip 주변 zoom)
- **Hold 데이터 부재**: BC demo가 5mm hold sequence를 거의 안 가짐. data augmentation: 정렬 후 sequence stall 추가, 또는 RL fine-tune.
- **Inference-time action damping**: 5mm 근접 시 action scale down (학습 없이 즉시 적용 가능, 별개 PR로 검토).

### 운영 lesson (추가)
- **BC loss 변형(aux/DDL/aug)으로 close_5mm 25% 천장 깨기 어려움** — 같은 데이터/같은 vision encoder에서 loss reshape는 도달 가능한 분포를 reshape할 뿐, encoder가 못 보는 mm-scale 정보를 못 만듦.
- **fresh aug + ckpt resume 위험** — 5k 정도로는 ckpt 분포 회복 못 함. aug 도입 시 from-scratch 또는 longer training 필수.

