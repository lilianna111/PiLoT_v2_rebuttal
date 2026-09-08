"""
在 DSM 和 DOM 上标出给定的世界坐标点，并检查是否超出边界。
单独运行：在 pixloc/crop 目录下执行 python vis_world_points_on_map.py
"""
import os
import sys
import numpy as np
import cv2

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_DSM_config
from proj2map_test import geo_coords_to_dsm_index


# ========== 配置：修改这里的路径和四点坐标 ==========
ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/dom_cgcs.tif"
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/dsm_cgcs.tif"
ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/domdsm_cgcs.npy"
# 输出目录（标注后的 DSM/DOM 图保存到这里）
out_dir = "/home/amax/Documents/datasets/crop/vis_points"

# 四个世界坐标点 [x, y, z]（与 world_coords 一致）
world_coords = [
    np.array([-9.23861120e+06,  1.25044486e+07,  9.71349106e+01]),
    np.array([-9.23861893e+06,  1.25045003e+07,  9.71349106e+01]),
    np.array([-9.23857556e+06,  1.25045104e+07,  9.71349106e+01]),
    np.array([-9.23856410e+06,  1.25044609e+07,  9.71349106e+01]),
]


def main():
    print("加载 DSM/DOM ...")
    geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(
        ref_DSM_path, ref_DOM_path, ref_npy_path
    )
    H, W = dsm_data.shape
    print(f"DSM/DOM 尺寸: H={H}, W={W}")

    # 世界坐标 -> 像素 (row, col)
    pixel_indices = []
    out_of_bounds = []
    for i, xyz in enumerate(world_coords):
        x, y = float(xyz[0]), float(xyz[1])
        row, col = geo_coords_to_dsm_index(x, y, dsm_trans)
        pixel_indices.append((row, col))
        if row < 0 or row >= H or col < 0 or col >= W:
            out_of_bounds.append((i, row, col, (x, y)))

    # 越界报告
    if out_of_bounds:
        print("\n*** 超出边界 ***")
        for i, row, col, (x, y) in out_of_bounds:
            print(f"  点 {i}: 像素 (row={row}, col={col}), 世界 (x={x:.2f}, y={y:.2f})")
            print(f"        有效范围: row in [0, {H}), col in [0, {W})")
        print()
    else:
        print("四个点均在边界内。")

    # 像素坐标汇总（cv2 用 (x,y) = (col, row)）
    print("像素坐标 (row, col)，对应 cv2 (x,y)=(col,row):")
    for i, (row, col) in enumerate(pixel_indices):
        print(f"  点 {i}: (row={row}, col={col})")

    # 准备画图：DSM 归一化到 0-255 并转 3 通道
    dsm_vis = dsm_data.astype(np.float32)
    valid = (dsm_vis > 0) & (dsm_vis < 10000)
    if np.any(valid):
        dsm_vis[~valid] = np.nan
        vmin, vmax = np.nanmin(dsm_vis), np.nanmax(dsm_vis)
        if vmax > vmin:
            dsm_vis = (dsm_vis - vmin) / (vmax - vmin) * 255
        else:
            dsm_vis = np.where(valid, 128, 0).astype(np.float32)
        dsm_vis = np.nan_to_num(dsm_vis, nan=0).astype(np.uint8)
    else:
        dsm_vis = np.zeros((H, W), dtype=np.uint8)
    dsm_vis = cv2.applyColorMap(dsm_vis, cv2.COLORMAP_VIRIDIS)
    dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()
    if dom_vis.shape[2] == 3:
        dom_vis = cv2.cvtColor(dom_vis, cv2.COLOR_RGB2BGR)

    # 在图上画点和连线（四点顺序：左上、右上、右下、左下）
    radius = max(8, min(H, W) // 200)
    thickness = max(2, min(H, W) // 500)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR

    for i, (row, col) in enumerate(pixel_indices):
        x, y = int(col), int(row)
        if 0 <= row < H and 0 <= col < W:
            cv2.circle(dsm_vis, (x, y), radius, colors[i], thickness)
            cv2.circle(dom_vis, (x, y), radius, colors[i], thickness)
            cv2.putText(
                dsm_vis, str(i), (x + radius + 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2
            )
            cv2.putText(
                dom_vis, str(i), (x + radius + 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2
            )
    pts = np.array([[int(col), int(row)] for row, col in pixel_indices], dtype=np.int32)
    if len(pts) == 4:
        cv2.polylines(dsm_vis, [pts], True, (255, 255, 255), 1)
        cv2.polylines(dom_vis, [pts], True, (255, 255, 255), 1)

    os.makedirs(out_dir, exist_ok=True)
    dsm_path = os.path.join(out_dir, "dsm_4points.png")
    dom_path = os.path.join(out_dir, "dom_4points.png")
    cv2.imwrite(dsm_path, dsm_vis)
    cv2.imwrite(dom_path, dom_vis)
    print(f"\n已保存: {dsm_path}")
    print(f"已保存: {dom_path}")


if __name__ == "__main__":
    main()
