# 세션 요약: Insertion 모델 분석 & 성능 향상 브레인스토밍 (2026-04-16)

## 1. 배경 및 핵심 문제

### 현재 상황
바늘-트로카 삽입(Needle-Trocar Insertion) 태스크를 VLA 모델로 학습 중.
Insertion 모델이 **미세 정렬(fine alignment)과 삽입(insertion)을 동시에 학습하는 데 실패**하고 있음.

### 핵심 원인 분석
- 모델이 바늘과 트로카의 **정렬 상태를 시각적으로 "인식"하지 못함**
- 정렬이 안 된 상태에서 삽입 동작을 시도하거나, 정렬이 된 상태에서도 불필요한 정렬 동작을 반복
- 하나의 모델이 "정렬해야 하는지 vs 삽입해야 하는지" 판단하는 것 자체가 어려움

### 결정된 방향
**태스크 분리 (Phase Decomposition)**
- **Align 모델**: 바늘을 트로카 위에 정확히 정렬하는 것에만 집중
- **Insertion 모델**: 이미 정렬된 상태에서 삽입만 수행
- 각 모델이 단일 목표에 집중하여 학습 난이도를 낮춤

---

## 2. 아키텍처 상세 분석

### 2.1 전체 파이프라인 흐름
```
[Wrist Camera Image] → [Qwen3.5-VL Vision Tower] → visual tokens (300개)
[Text Instruction]   → [Qwen3.5-VL Tokenizer]    → text tokens
[Proprio (ee_pose)]  → [Linear/Transformer Proj]  → proprio tokens

    ↓ (모두 concat하여 VLM에 입력)

[Qwen3.5-VL Transformer] → hidden_states (conditioning signal)

    ↓

[Diffusion Policy Head] → action prediction (연속 action 출력)
```

### 2.2 이미지 처리 경로 (Qwen3.5-VL)

| 항목 | 내용 |
|------|------|
| Vision encoder | Qwen3.5-VL **내장 vision tower** (ViT 기반) |
| Config의 SigLIP 설정 | `vision_encoder_path: "google/siglip2-base-patch16-256"` → **Qwen 사용 시 완전 무시됨** |
| 입력 해상도 | 480 × 640 (원본 유지, 256×256으로 리사이즈하지 않음) |
| 패치 크기 | 16 × 16 pixels |
| Merge 전략 | 2×2 인접 패치 병합 |
| 최종 토큰 수 | (480/16/2) × (640/16/2) = 15 × 20 = **300 visual tokens** |
| 토큰당 커버 영역 | 32 × 32 pixels |
| 트로카 구멍 커버리지 | 약 **9개 토큰** (3×3 영역) → **해상도는 병목이 아님** |

**결론:** 이미지 해상도나 vision encoder 문제가 아님. 트로카 구멍은 충분한 수의 토큰으로 표현되고 있음.

### 2.3 Proprioception 경로

| 항목 | 내용 |
|------|------|
| 기본 신호 | `ee_pose` (7D): position(3D) + orientation(4D, quaternion) |
| 추가 신호 (optional) | `sensor_dist` (1D): 센서 거리, 20mm 클리핑, 정규화 없이 raw 사용 |
| Proprio 차원 | 기본 7D, 센서 포함 시 8D |
| Projector | Linear (기본) 또는 ActionTransformerProjector (옵션) |
| 통합 방식 | Projector 출력을 VLM embedding sequence 앞에 **prepend** |
| VLM 내부 처리 | image + text + proprio가 self-attention으로 함께 처리됨 |

### 2.4 Diffusion Policy Head
- VLM의 마지막 hidden_states를 **conditioning signal**로 받음
- 이미지 픽셀을 직접 보지 않음 — VLM이 추출한 표현(representation)에 의존
- 즉, VLM이 "정렬 상태"를 표현에 담지 못하면 policy도 이를 알 수 없음

---

## 3. 검토 완료된 이미지 전처리 방향

### 3.1 Edge Detection (Canny, Sobel, Laplacian) — ❌ 비추천

**실험 내용:** 세 가지 edge detection 방법을 wrist camera 이미지에 적용하여 시각화

**문제점:**
- 배경 노이즈가 과도함 — 테이블 구멍, 프레임, 조명 반사 등이 모두 edge로 검출
- 바늘(needle) 자체의 edge가 불안정하게 감지됨 (가늘고 반사가 심함)
- Sobel이 가장 깔끔했으나, 추가 정보 가치가 의문
- **근본적 한계:** SigLIP/Qwen vision tower가 이미 내부적으로 edge-like feature를 학습하고 있음 → 중복 정보

**시각화 위치:** `/data/public/NAS/VLANeXt/viz_edge/`

### 3.2 Optical Flow (Farneback) — ❌ 비추천

**실험 내용:** 연속 프레임 간 optical flow를 계산하여 움직임 패턴 분석

**문제점:**
- Wrist camera 특성상 **ego-motion(카메라 자체의 움직임)이 flow를 지배**
- 바늘과 트로카의 상대적 움직임이 ego-motion에 묻혀서 보이지 않음
- `ee_pose` proprio history가 이미 동일한 정보(로봇 팔의 움직임)를 더 정확하게 제공
- **결론:** 정보 중복 + 추가 연산 비용만 발생

### 3.3 Hough Transform (Line + Circle Detection) — ❌ 비추천

**실험 내용:** 트로카 입구를 원(circle)으로 검출하고, 바늘을 직선(line)으로 검출하는 시도

**문제점:**
- 트로카 입구가 **깔끔한 원으로 검출되지 않음** — 회색 돔 위 작은 구멍, contrast가 약함
- Concentric circle 필터링 시도:
  - 느슨한 기준 → 거짓 양성(false positive) 과다
  - 엄격한 기준 → 아무것도 검출 못 함
- 파라미터 민감도가 매우 높아 **robust하지 않음**
- **sim → real 전이 불가능** — 시뮬레이터의 렌더링과 실제 환경의 시각 특성이 다름

### 3.4 전체 결론
> 이미지 전처리를 통한 명시적 특징 추출은 이 태스크에 적합하지 않음.
> VLM의 내부 표현을 개선하거나, 학습 전략을 변경하는 방향이 더 유망.

---

## 4. 아직 탐색하지 않은 방향 (우선순위 순)

### 4.1 학습 조건 개선 (config 변경만으로 가능 — 낮은 구현 비용)

| # | 방향 | 설명 | 관련 config |
|---|------|------|-------------|
| 1 | **Multi-view** | side camera 추가하여 Z축 깊이 정보 간접 제공. wrist cam만으로는 깊이 판단이 어려움 | `view_mode: "multi"` |
| 2 | **Video input** | 단일 프레임 대신 연속 프레임 입력으로 시간 정보 활용. 바늘이 들어가는 동적 변화 인식 가능 | `input_modality: "video"` |
| 3 | **더 긴 history** | action history를 16 스텝으로 늘려 삽입 중 미세한 변화 포착 | `history_len: 16` |
| 4 | **Inference steps 증가** | diffusion denoising step을 10으로 늘려 더 정밀한 action 생성 | `num_inference_timesteps: 10` |
| 5 | **Transformer proprio projector** | Linear 대신 Transformer projector로 proprio 표현력 향상 | `use_transformer_proprio_projector: true` |

### 4.2 모델 구조 변경 (중간 구현 비용)

| # | 방향 | 설명 | 구현 위치 |
|---|------|------|-----------|
| 6 | **Visual bypass** | VLM의 intermediate feature를 diffusion policy에 직접 cross-attention으로 전달. VLM bottleneck 우회 | `src/models/VLANeXt.py` |
| 7 | **ROI crop** | 트로카 주변 영역만 crop하여 입력. 전체 이미지 대비 타겟 영역 비율 향상 | 전처리 단계 |
| 8 | **Attention supervision** | SpatialCrossAttentionHead의 attention이 바늘/트로카 영역에 집중하도록 auxiliary loss 추가 | `src/models/VLANeXt.py` |

### 4.3 데이터/학습 전략 (다양한 구현 비용)

| # | 방향 | 설명 | 비고 |
|---|------|------|------|
| 9 | **데이터 양/다양성** | 현재 10k 에피소드가 충분한지 검증. 더 다양한 초기 조건 필요할 수 있음 | 데이터 수집 필요 |
| 10 | **Curriculum learning** | 쉬운(가까운 정렬) → 어려운(먼 정렬) 순서로 학습. 점진적 난이도 증가 | 학습 스크립트 수정 |
| 11 | **센서 데이터 활용** | `sensor_dist`가 insertion에서만 유의미한 신호. Align에서는 대부분 20mm(미감지) | 이미 구현됨 |
| 12 | **Spatial auxiliary loss** | 모델에 이미 구현되어 있음. config에서 활성화만 하면 됨 | config 변경만 |

### 4.4 시뮬레이터 특권 정보 (sim-only, real 전이 시 제거 필요)

| # | 방향 | 설명 | 주의사항 |
|---|------|------|----------|
| 13 | **Depth map** | Z축 거리 정보를 추가 채널로 직접 제공. 깊이 인식 문제 근본 해결 | sim→real 전이 시 depth sensor 필요 |
| 14 | **Segmentation mask** | 바늘/트로카를 명시적으로 구분하는 마스크 제공. 모델이 "어디를 봐야 하는지" 직접 학습 | sim에서만 가능 |

---

## 5. 현재 학습 상황 (2026-04-16 기준)

| 서버 | 태스크 | 설정 | 상태 |
|------|--------|------|------|
| **Nayohan** | Align 기본 학습 | 20k 에피소드, batch 100, 20k steps | 예정/진행중 |
| **Najo** | Insertion + 센서 | sensor_dist 포함, 체크포인트 없이 처음부터 학습 | 예정/진행중 |

---

## 6. 핵심 파일 참조

### 모델
| 파일 | 역할 |
|------|------|
| `src/models/VLANeXt.py` | 전체 모델 아키텍처. SpatialCrossAttentionHead, proprio integration, diffusion policy |
| `src/models/encoder.py` | ActionTransformerProjector — proprio를 VLM embedding으로 변환 |

### 데이터
| 파일 | 역할 |
|------|------|
| `src/datasets/sim_act_insertion.py` | Insertion 데이터 로더 (multi-path, h5 instruction 지원) |

### Config
| 파일 | 역할 |
|------|------|
| `config/sim_train_insertion_config.yaml` | Insertion 학습 설정 |
| `config/sim_train_align_config.yaml` | Align 학습 설정 |

### 스크립트 (`scripts/`)

#### 학습
| 파일 | 설명 |
|------|------|
| `scripts/train.py` | VLANeXt 모델 학습. DDP + DeepSpeed 분산 학습 지원. LIBERO/DROID/SimAct 등 다양한 데이터셋 선택 가능. DataCollatorForVLANeXt에서 multi-view, video, proprio, spatial target, augmentation을 처리. Config 파일(YAML)로 모든 설정 제어 |

#### 시뮬레이션 평가
| 파일 | 설명 |
|------|------|
| `scripts/sim_eval.py` | **전체 3단계 파이프라인** (align → approach → insertion) 평가. Closed-loop: render → predict_action → IK solver → mj_step. 성공률, 거리, 각도 등 메트릭을 CSV로 기록. 롤아웃 비디오 저장 가능 |
| `scripts/sim_eval_align_only.py` | **Align 단독 평가.** 트로카 근처 랜덤 위치에서 바늘 정렬까지 수행. 궤적(ee_pose)을 NPZ로 저장하고 3D + 2D 프로젝션 시각화 생성. 거리 임계값 기반 성공 판정 |
| `scripts/sim_eval_insertion_only.py` | **Insertion 단독 평가.** IK로 트로카에 미리 정렬한 후 approach offset에서 모델이 제어 시작. 삽입 깊이 + lateral tolerance 기반 성공 판정. Approach/Insertion 단계 별도 추적 |
| `scripts/sim_eval_approach_only.py` | **Approach 단독 평가.** 랜덤 홈 포즈에서 트로카 근처까지 접근. 거리 기반 성공 판정. 3D 궤적 시각화 (성공=초록, 실패=빨강) |

#### LIBERO 벤치마크
| 파일 | 설명 |
|------|------|
| `scripts/libero_bench_eval.py` | LIBERO 벤치마크 태스크 평가. Action scaling(min/max bounds) 지원. spatial/object/goal/10-action 변형 처리. 롤아웃 비디오 + CSV/JSON 결과 |
| `scripts/libero_plus_bench_eval.py` | LIBERO 확장 평가. libero_bench_eval과 거의 동일하나 추가 JSON 로깅 포함 |

#### 분석/후처리
| 파일 | 설명 |
|------|------|
| `scripts/analyze_eval.py` | 평가 결과 후처리. metrics_summary.csv를 입력받아 성공률, 실패 모드 분류, near-miss 분석, 단축 방향별(X/Y/Z/angle) perturbation 분석, 사분면 분석 수행. 약한 방향에 대한 데이터 수집 명령어 자동 생성 |
| `scripts/merge_eval_shards.py` | 병렬 평가 shard 병합. 여러 shard의 CSV/mp4/npz/png 파일을 하나로 합치고 merged 성공률 계산 후 analyze_eval 호출 |

#### 유틸리티
| 파일 | 설명 |
|------|------|
| `scripts/size_speed_eval.py` | 모델 크기/속도 벤치마크. 파라미터 수, 메모리(MB), 추론 latency 측정. PaliGemma/Llama/Qwen별 더미 입력 생성. multi-batch 지원 |
| `scripts/vis_grid_coverage.py` | 그리드 기반 데이터 수집 커버리지 시각화. 워커별 실패 로그(IK_FAIL/TIMEOUT) 집계 후 3D scatter + 2D 프로젝션(XY/XZ/YZ) 시각화. 대칭 그리드: XY ±30mm, Z ±20mm |

### 시각화 결과
| 경로 | 내용 |
|------|------|
| `/data/public/NAS/VLANeXt/viz_edge/` | Edge detection, Optical flow, Hough transform 시각화 결과 |

---

## 7. 다음 단계 (TODO)

1. **Align 학습 결과 확인** — Nayohan 서버 학습 완료 후 eval 수행
2. **Insertion + 센서 학습 결과 확인** — Najo 서버 학습 완료 후 eval 수행
3. **미탐색 방향 중 우선순위 결정** — 결과에 따라 multi-view, spatial aux loss 등 시도
4. **태스크 분리 효과 검증** — Align + Insertion 파이프라인 연결 테스트
