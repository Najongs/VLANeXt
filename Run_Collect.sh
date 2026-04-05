#!/bin/bash
# =============================================================
# Fine-Alignment Data Collection (병렬)
# =============================================================
#
# 사용법:
#   bash Run_Collect.sh [mode] [workers] [episodes_per_worker]
#
# 인자:
#   mode      수집 모드 (기본: uniform)
#   workers   병렬 워커 수 (기본: 10)
#   episodes  워커당 에피소드 수 (기본: 1000)
#             총 수집량 = workers x episodes
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
#   full        전체 파이프라인 수집 (Save_dataset.py).
#               정렬뿐 아니라 삽입까지 포함된 full trajectory.
#               align-only와는 다른 데이터셋.
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
#   bash Run_Collect.sh uniform 10 1000      # 균등 10,000개
#   bash Run_Collect.sh bias_x_neg 10 500    # X- 편향 5,000개
#   bash Run_Collect.sh bias_all 10 500      # X-,Y- 각 5,000개씩
#   bash Run_Collect.sh full 5 500           # full pipeline 2,500개
#
# =============================================================

MODE=${1:-uniform}
WORKERS=${2:-10}
EPISODES=${3:-1000}
BASE=/data/public/NAS/VLANeXt/dataset/fine_align

echo "============================================================="
echo "  Mode: ${MODE}"
echo "  Workers: ${WORKERS}"
echo "  Episodes/worker: ${EPISODES}"
echo "  Total: $((WORKERS * EPISODES))"
echo "============================================================="

case $MODE in
    uniform)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/uniform_new
        ;;

    bias_x_neg)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_x_neg \
            --bias x_neg
        ;;

    bias_y_neg)
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_y_neg \
            --bias y_neg
        ;;

    bias_all)
        echo ""
        echo "=== [1/2] X negative bias ==="
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_x_neg \
            --bias x_neg

        echo ""
        echo "=== [2/2] Y negative bias ==="
        python Sim/run_parallel.py \
            --script align --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/bias_y_neg \
            --bias y_neg
        ;;

    full)
        python Sim/run_parallel.py \
            --script full --workers $WORKERS --episodes $EPISODES \
            --base-dir ${BASE}/full_new
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  uniform     - 균등 random perturbation"
        echo "  bias_x_neg  - X 음수 방향 편향"
        echo "  bias_y_neg  - Y 음수 방향 편향"
        echo "  bias_all    - X-, Y- 순차 수집"
        echo "  full        - 전체 파이프라인 (삽입 포함)"
        exit 1
        ;;
esac
