#!/bin/bash
# =============================================================
# Fine-Alignment Data Collection (병렬)
# =============================================================
#
# 사용법:
#   bash Run_Collect.sh [mode] [workers] [episodes_per_worker] [4th] [randomize_phantom]
#
# 인자:
#   mode      수집 모드 (기본: uniform)
#   workers   병렬 워커 수 (기본: 10)
#   episodes  워커당 에피소드 수 (기본: 1000)
#             총 수집량 = workers x episodes
#   5th arg   팬텀 랜덤화 (true/false, 기본: false)
#
# 모드 설명:
#   uniform     기본 수집. 모든 방향으로 균등한 random perturbation.
#               새 데이터 추가 수집 시 사용.
#
#   bias_x_neg  X 음수 방향 편향 수집.
#               needle이 X- 위치에서 시작 → X+ 방향으로 복귀하는 데이터.
#               eval에서 "왼쪽 출발 → 오른쪽 이동" SR이 낮을 때 보강용.
#
#   bias_y_neg  Y 음수 방향 편향 수집. (위와 동일 논리, Y축)
#
#   bias_all    약한 방향 전부 순차 수집 (bias_x_neg + bias_y_neg).
#               analyze_eval.py에서 특정 방향 SR이 낮게 나왔을 때
#               해당 방향들을 한번에 보강.
#
#   grid        그리드 기반 균등 수집 (stratified sampling).
#               XYZ 공간을 격자로 나눠 빈틈 없이 커버.
#               bins_xy x bins_xy x bins_z 개의 에피소드 생성.
#               2번째 인자: workers, 3번째: bins_xy, 4번째: bins_z
#               예: bash Run_Collect.sh grid 10 8 6  → 8x8x6=384 에피소드
#
#   full        approach+align+insert 전체 파이프라인 수집 (Save_dataset_approach_only.py).
#               align-only와는 다른 데이터셋.
#
#   insertion   Insertion-only 수집. 정렬 완료 상태에서 접근+삽입만 녹화.
#               align 모델 종료 위치(z 뒤 + xy 오프셋)에서 시작.
#
#   insertion_perturb  위와 동일 + 각도/위치 perturbation 포함.
#               불완전 정렬에서 보정하며 삽입하는 데이터.
#
# 저장 위치:
#   dataset/fine_align/
#   ├── collected_data_merged/          ← 기존 데이터 (건드리지 않음)
#   ├── uniform_new/collected_data_merged/
#   ├── bias_x_neg/collected_data_merged/
#   ├── bias_y_neg/collected_data_merged/
#   └── full_new/collected_data_merged/
#
#   학습 시 data_root을 dataset/fine_align/ 으로 설정하면
#   하위 폴더의 모든 .h5를 recursive로 로드함.
#
# 예시:
#   bash Run_Collect.sh uniform 20 1000           # 균등 10,000개
#   bash Run_Collect.sh uniform 10 1000 _ true    # 균등 + 팬텀 랜덤
#   bash Run_Collect.sh bias_x_neg 10 500         # X- 편향 5,000개
#   bash Run_Collect.sh bias_all 10 500            # X-,Y- 각 5,000개씩
#   bash Run_Collect.sh grid 10 8 6               # 8x8x6=384 그리드 에피소드
#   bash Run_Collect.sh grid 10 12 8              # 12x12x8=1152 촘촘한 그리드
#   bash Run_Collect.sh multi_phantom 10 2000       # Y축 5위치 × 10w × 2000ep
#   bash Run_Collect.sh approach 10 2000           # approach+align (no insertion), Y축 5위치
#   bash Run_Collect.sh full 5 500                 # full pipeline 2,500개
#   bash Run_Collect.sh insertion 10 500             # insertion only, 팬텀 고정
#   bash Run_Collect.sh insertion_perturb 10 500     # insertion + perturbation (불완전 정렬)
#
# =============================================================

# Phase 1: Micro-Alignment (가까이에서 정렬)

# 고정 팬텀 (현재 완료)
# bash Run_Collect.sh uniform 10 1000          # 10,000개, 팬텀 고정
# - 스크립트: Save_dataset_align_only.py
# - 가까운 perturbation(±30mm XY, ±20mm Z)에서 trocar 정렬

# Phase 2: Spatial Generalization (랜덤 팬텀 정렬)

# Y축 5위치에서 micro-alignment
# bash Run_Collect.sh multi_phantom 10 2000    # 5위치 × 20,000 = 100,000개
# - 스크립트: Save_dataset_align_only.py + --phantom-pos
# - 팬텀 위치별: Y=0.0, -0.1, -0.2, -0.3, -0.4 (X=0 고정)
# - 각 위치에서 가까운 perturbation → 정렬 녹화

# Phase 3: Out-of-view Approach (먼 거리 접근)

# Y축 5위치에서 approach+align (insertion 제외)
# bash Run_Collect.sh approach 10 2000         # 5위치 × 20,000 = 100,000개
# - 스크립트: Save_dataset_approach_only.py + --phantom-pos + --no-insertion
# - 먼 거리(home pose)에서 trocar까지 접근 + 정렬, 삽입 직전에 종료

# Phase 4: Insertion (삽입)

# Y축 5위치에서 full pipeline
# bash Run_Collect.sh full 10 2000             # 5위치 × 20,000 = 100,000개
# - 스크립트: Save_dataset_approach_only.py + --phantom-pos
# - approach + align + insert 전체 trajectory

# Phase 4-alt: Insertion Only (정렬 완료 후 삽입만)

# 팬텀 고정, 정렬 뒤 5mm에서 시작 → 접근+삽입 녹화
# bash Run_Collect.sh insertion 10 500         # 5,000개
# - 스크립트: Save_dataset_insertion_only.py
# - align 모델 종료 위치 시뮬레이션 (z offset + xy offset)
# - correct-while-inserting 전략 학습용

# 불완전 정렬 + perturbation 포함
# bash Run_Collect.sh insertion_perturb 10 500 # 5,000개
# - 위와 동일 + 각도/위치 perturbation 추가


MODE=${1:-uniform}
WORKERS=${2:-10}
EPISODES=${3:-1000}   # grid 모드에서는 3번째=bins_xy, 4번째=bins_z로 사용
RANDOMIZE_PHANTOM=${5:-false}  # true로 설정하면 팬텀 위치 랜덤화
BASE=/data/public/NAS/VLANeXt/dataset/fine_align_Random

# Phantom randomization flag
PHANTOM_FLAG=""
if [ "$RANDOMIZE_PHANTOM" = "true" ]; then
    PHANTOM_FLAG="--randomize-phantom-pos"
fi

# Side camera flag (set NO_SIDE_CAM=true to skip side camera)
SIDE_CAM_FLAG=""
if [ "${NO_SIDE_CAM:-false}" = "true" ]; then
    SIDE_CAM_FLAG="--no-side-camera"
fi

if [ "$MODE" = "grid" ]; then
    BINS_XY=${3:-8}
    BINS_Z=${4:-6}
    TOTAL=$((BINS_XY * BINS_XY * BINS_Z))
    echo "============================================================="
    echo "  Mode: ${MODE}"
    echo "  Workers: ${WORKERS}"
    echo "  Grid: ${BINS_XY} x ${BINS_XY} x ${BINS_Z} = ${TOTAL} cells"
    echo "============================================================="
else
    echo "============================================================="
    echo "  Mode: ${MODE}"
    echo "  Workers: ${WORKERS}"
    echo "  Episodes/worker: ${EPISODES}"
    echo "  Total: $((WORKERS * EPISODES))"
    echo "============================================================="
fi

case $MODE in
    uniform)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/uniform_new $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    bias_x_neg)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_x_neg \
            --bias x_neg $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    bias_y_neg)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_y_neg \
            --bias y_neg $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    bias_all)
        echo ""
        echo "=== [1/2] X negative bias ==="
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_x_neg \
            --bias x_neg $PHANTOM_FLAG $SIDE_CAM_FLAG

        echo ""
        echo "=== [2/2] Y negative bias ==="
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_y_neg \
            --bias y_neg $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    grid)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS \
            --base-dir ${BASE}/grid \
            --grid --grid-bins-xy $BINS_XY --grid-bins-z $BINS_Z $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    multi_phantom)
        # Y축 5개 고정 위치 (X=0)에서 순차 수집
        POSITIONS=("0.0 0.0" "0.0 -0.1" "0.0 -0.2" "0.0 -0.3" "0.0 -0.4")
        LABELS=("y000" "y010" "y020" "y030" "y040")
        for idx in "${!POSITIONS[@]}"; do
            POS=(${POSITIONS[$idx]})
            LABEL=${LABELS[$idx]}
            echo ""
            echo "=== [$((idx+1))/${#POSITIONS[@]}] Phantom pos=(${POS[0]}, ${POS[1]}) ==="
            python Sim/run_parallel.py \
                --script align --workers $WORKERS --episodes $EPISODES \
                --base-dir ${BASE}/phantom_${LABEL} \
                --phantom-pos ${POS[0]} ${POS[1]} $SIDE_CAM_FLAG
        done
        ;;

    approach)
        # Y축 5개 고정 위치에서 approach+align만 수집 (insertion 제외)
        POSITIONS=("0.0 0.0" "0.0 -0.1" "0.0 -0.2" "0.0 -0.3" "0.0 -0.4")
        LABELS=("y000" "y010" "y020" "y030" "y040")
        for idx in "${!POSITIONS[@]}"; do
            POS=(${POSITIONS[$idx]})
            LABEL=${LABELS[$idx]}
            echo ""
            echo "=== [$((idx+1))/${#POSITIONS[@]}] Approach: Phantom pos=(${POS[0]}, ${POS[1]}) ==="
            python Sim/run_parallel.py \
                --script approach --workers $WORKERS --episodes $EPISODES \
                --base-dir ${BASE}/approach_${LABEL} \
                --phantom-pos ${POS[0]} ${POS[1]} --no-insertion $SIDE_CAM_FLAG
        done
        ;;

    full)
        # Y축 5개 고정 위치에서 full pipeline (approach+align+insert)
        POSITIONS=("0.0 0.0" "0.0 -0.1" "0.0 -0.2" "0.0 -0.3" "0.0 -0.4")
        LABELS=("y000" "y010" "y020" "y030" "y040")
        for idx in "${!POSITIONS[@]}"; do
            POS=(${POSITIONS[$idx]})
            LABEL=${LABELS[$idx]}
            echo ""
            echo "=== [$((idx+1))/${#POSITIONS[@]}] Full: Phantom pos=(${POS[0]}, ${POS[1]}) ==="
            python Sim/run_parallel.py \
                --script approach --workers $WORKERS --episodes $EPISODES \
                --base-dir ${BASE}/full_${LABEL} \
                --phantom-pos ${POS[0]} ${POS[1]} $SIDE_CAM_FLAG
        done
        ;;

    insertion)
        # 팬텀 고정, 정렬 완료 상태에서 접근+삽입만 녹화
        python Sim/run_parallel.py \
            --script insertion --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/insertion \
            --approach-offset 5 --approach-xy-offset 2 $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    insertion_perturb)
        # 불완전 정렬 + perturbation 포함
        python Sim/run_parallel.py \
            --script insertion --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/insertion_perturb \
            --approach-offset 5 --approach-xy-offset 2 \
            --perturb --perturb-angle 5 $PHANTOM_FLAG $SIDE_CAM_FLAG
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  uniform          - 균등 random perturbation"
        echo "  bias_x_neg       - X 음수 방향 편향"
        echo "  bias_y_neg       - Y 음수 방향 편향"
        echo "  bias_all         - X-, Y- 순차 수집"
        echo "  grid             - 그리드 기반 균등 수집 (args: workers bins_xy bins_z)"
        echo "  multi_phantom    - Y축 5개 고정 위치 순차 수집 (X=0, Y=0~-0.4)"
        echo "  approach         - approach+align 수집, insertion 제외 (Y축 5위치)"
        echo "  full             - 전체 파이프라인 (삽입 포함)"
        echo "  insertion        - insertion only (정렬 후 접근+삽입)"
        echo "  insertion_perturb - insertion + perturbation (불완전 정렬)"
        exit 1
        ;;
esac
