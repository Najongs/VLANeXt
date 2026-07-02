#!/usr/bin/env python3
"""
Grid 데이터 수집 결과 시각화.

워커별 grid_failed_cells.json을 통합하고,
성공/실패 셀을 3D scatter + 2D projection으로 시각화.

Usage:
    python scripts/vis_grid_coverage.py /home/najo/NAS/VLANeXt/dataset/fine_align/grid/collected_data_merged
    python scripts/vis_grid_coverage.py /path/to/grid_data --bins-xy 8 --bins-z 6
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_failed_cells(base_dir):
    """워커별 grid_failed_cells.json 통합. 새 포맷(dict with reason)과 구 포맷(list) 호환."""
    ik_fails = []
    timeout_fails = []
    legacy_fails = []
    for p in sorted(Path(base_dir).glob("worker_*/grid_failed_cells.json")):
        with open(p) as f:
            entries = json.load(f)
        for e in entries:
            if isinstance(e, dict):
                center = e["center"]
                if e.get("reason") == "IK_FAIL":
                    ik_fails.append(center)
                else:
                    timeout_fails.append(center)
            else:
                legacy_fails.append(e)
    return (
        np.array(ik_fails) if ik_fails else np.empty((0, 3)),
        np.array(timeout_fails) if timeout_fails else np.empty((0, 3)),
        np.array(legacy_fails) if legacy_fails else np.empty((0, 3)),
    )


def build_all_cells(bins_xy, bins_z, xy_mm=30.0, z_mm=20.0):
    """전체 그리드 셀 중심 좌표 생성"""
    x_centers = np.linspace(-xy_mm, xy_mm, bins_xy * 2 + 1)[1::2]
    y_centers = np.linspace(-xy_mm, xy_mm, bins_xy * 2 + 1)[1::2]
    z_centers = np.linspace(-z_mm, z_mm, bins_z * 2 + 1)[1::2]
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def main():
    parser = argparse.ArgumentParser(description="Visualize grid collection coverage")
    parser.add_argument("grid_dir", type=str, help="Grid data directory")
    parser.add_argument("--bins-xy", type=int, default=8)
    parser.add_argument("--bins-z", type=int, default=6)
    args = parser.parse_args()

    base = Path(args.grid_dir)
    ik_fail_pts, timeout_fail_pts, legacy_fail_pts = load_failed_cells(base)
    # legacy (구 포맷)는 통합
    if len(legacy_fail_pts) > 0:
        all_fail_combined = legacy_fail_pts
        ik_fail_pts = np.empty((0, 3))
        timeout_fail_pts = np.empty((0, 3))
    else:
        all_fail_combined = np.concatenate([ik_fail_pts, timeout_fail_pts]) if (len(ik_fail_pts) + len(timeout_fail_pts)) > 0 else np.empty((0, 3))

    fail_pts = all_fail_combined
    all_pts = build_all_cells(args.bins_xy, args.bins_z)
    total = len(all_pts)
    n_ik = len(ik_fail_pts)
    n_timeout = len(timeout_fail_pts)
    n_legacy = len(legacy_fail_pts)
    n_fail = len(fail_pts)
    n_success = total - n_fail

    # 성공 셀 = 전체 - 실패
    if n_fail > 0:
        fail_set = set(map(lambda r: (round(r[0], 2), round(r[1], 2), round(r[2], 2)), fail_pts))
        success_mask = np.array([
            (round(r[0], 2), round(r[1], 2), round(r[2], 2)) not in fail_set
            for r in all_pts
        ])
    else:
        success_mask = np.ones(total, dtype=bool)

    succ_pts = all_pts[success_mask]
    print(f"Total cells: {total}")
    print(f"Success:  {n_success} ({n_success/total*100:.1f}%)")
    if n_legacy > 0:
        print(f"Failed:   {n_fail} ({n_fail/total*100:.1f}%) [legacy format, no reason info]")
    else:
        print(f"Failed:   {n_fail} ({n_fail/total*100:.1f}%)")
        print(f"  IK_FAIL:  {n_ik} (singularity, can't reach perturbed pose)")
        print(f"  Timeout:  {n_timeout} (reached pose but alignment too slow)")

    # --- 통합 실패 셀 JSON 저장 ---
    merged_path = base / "grid_failed_cells_all.json"
    merged_data = []
    for pt in ik_fail_pts:
        merged_data.append({"center": pt.tolist(), "reason": "IK_FAIL"})
    for pt in timeout_fail_pts:
        merged_data.append({"center": pt.tolist(), "reason": "Timeout"})
    for pt in legacy_fail_pts:
        merged_data.append({"center": pt.tolist(), "reason": "unknown"})
    with open(merged_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"\nMerged failed cells → {merged_path}")

    # ================================================================
    # Figure 1: 3D scatter
    # ================================================================
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221, projection='3d')
    if len(succ_pts) > 0:
        ax1.scatter(succ_pts[:, 0], succ_pts[:, 1], succ_pts[:, 2],
                    c='dodgerblue', alpha=0.4, s=20, label=f'Success ({n_success})')
    if len(ik_fail_pts) > 0:
        ax1.scatter(ik_fail_pts[:, 0], ik_fail_pts[:, 1], ik_fail_pts[:, 2],
                    c='red', alpha=0.8, s=40, marker='x', label=f'IK_FAIL ({n_ik})')
    if len(timeout_fail_pts) > 0:
        ax1.scatter(timeout_fail_pts[:, 0], timeout_fail_pts[:, 1], timeout_fail_pts[:, 2],
                    c='orange', alpha=0.8, s=40, marker='^', label=f'Timeout ({n_timeout})')
    if len(legacy_fail_pts) > 0:
        ax1.scatter(legacy_fail_pts[:, 0], legacy_fail_pts[:, 1], legacy_fail_pts[:, 2],
                    c='red', alpha=0.8, s=40, marker='x', label=f'Failed ({n_legacy})')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Grid Coverage')
    ax1.legend(fontsize=8)

    # ================================================================
    # Figure 2-4: 2D projections (XY, XZ, YZ) with heatmap
    # ================================================================
    projections = [
        (222, 'XY Projection (top view)', 0, 1, 'X (mm)', 'Y (mm)'),
        (223, 'XZ Projection (front view)', 0, 2, 'X (mm)', 'Z (mm)'),
        (224, 'YZ Projection (side view)', 1, 2, 'Y (mm)', 'Z (mm)'),
    ]

    for subplot, title, ax_i, ax_j, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot)

        # 해당 축 조합의 unique 셀 중심
        xy_mm = 30.0
        z_mm = 20.0
        bins_i = args.bins_xy if ax_i < 2 else args.bins_z
        bins_j = args.bins_xy if ax_j < 2 else args.bins_z
        range_i = xy_mm if ax_i < 2 else z_mm
        range_j = xy_mm if ax_j < 2 else z_mm

        # 2D 실패 밀도 히트맵
        edges_i = np.linspace(-range_i, range_i, bins_i + 1)
        edges_j = np.linspace(-range_j, range_j, bins_j + 1)

        # 전체 셀 중 실패 비율 계산
        fail_count = np.zeros((bins_i, bins_j))
        total_count = np.zeros((bins_i, bins_j))

        for pt in all_pts:
            ii = min(int((pt[ax_i] + range_i) / (2 * range_i) * bins_i), bins_i - 1)
            jj = min(int((pt[ax_j] + range_j) / (2 * range_j) * bins_j), bins_j - 1)
            total_count[ii, jj] += 1

        if n_fail > 0:
            for pt in fail_pts:
                ii = min(int((pt[ax_i] + range_i) / (2 * range_i) * bins_i), bins_i - 1)
                jj = min(int((pt[ax_j] + range_j) / (2 * range_j) * bins_j), bins_j - 1)
                fail_count[ii, jj] += 1

        fail_ratio = np.divide(fail_count, total_count, where=total_count > 0,
                               out=np.zeros_like(fail_count))

        im = ax.imshow(fail_ratio.T, origin='lower', cmap='RdYlGn_r',
                       extent=[-range_i, range_i, -range_j, range_j],
                       aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Failure rate')

        # 셀별 실패율 텍스트
        for ii in range(bins_i):
            for jj in range(bins_j):
                ci = (edges_i[ii] + edges_i[ii + 1]) / 2
                cj = (edges_j[jj] + edges_j[jj + 1]) / 2
                rate = fail_ratio[ii, jj]
                if rate > 0:
                    color = 'white' if rate > 0.5 else 'black'
                    ax.text(ci, cj, f'{rate:.0%}', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(f'Grid Collection: {n_success}/{total} success ({n_success/total*100:.0f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = base / "grid_coverage.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
