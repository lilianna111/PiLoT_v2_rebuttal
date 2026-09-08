#!/usr/bin/env python3
"""
测试新的ENU射线投射实现
"""
import sys
sys.path.insert(0, '/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528')

from pixloc.crop.ray_casting_fix import caculate_predictXYZ_batch_enu
import numpy as np
from osgeo import gdal

# 测试数据
test_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
test_K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]

# 4个角点
test_pixel_coords = [
    [0, 0],           # 左上
    [1920, 0],        # 右上
    [1920, 1080],     # 右下
    [0, 1080]         # 左下
]

# 加载4326 DSM
dsm_path = '/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif'
dsm_map = gdal.Open(dsm_path)
geotransform = dsm_map.GetGeoTransform()
band = dsm_map.GetRasterBand(1)
area = band.ReadAsArray()
area_minZ = np.ma.masked_values(area, -9999).min()

print("=" * 70)
print("测试ENU射线投射实现")
print("=" * 70)
print(f"\n输入:")
print(f"  Pose: {test_pose}")
print(f"  K: {test_K}")
print(f"  Pixel coords: {test_pixel_coords}")
print(f"  DSM path: {dsm_path}")
print(f"  DSM shape: {area.shape}")
print(f"  area_minZ: {area_minZ:.2f}m")
print(f"  Geotransform: {geotransform}")

try:
    result_ecef, k_values, heights = caculate_predictXYZ_batch_enu(
        test_K, test_pose, test_pixel_coords,
        area, geotransform, area_minZ, num_sample=4000
    )

    print(f"\n输出:")
    print(f"  result_ecef shape: {result_ecef.shape}")
    print(f"  k_values: {k_values}")
    print(f"  heights: {heights}")

    print(f"\n4个角点的ECEF坐标:")
    for i, (coord, ecef) in enumerate(zip(test_pixel_coords, result_ecef)):
        print(f"  [{i}] Pixel {coord} → ECEF {ecef}")

    # 检查4个点是否不同
    unique_points = np.unique(result_ecef, axis=0)
    print(f"\n唯一点数量: {len(unique_points)} (应该是4)")

    if len(unique_points) == 4:
        print("\n✓ 测试通过: 4个角点的ECEF坐标各不相同!")

        # 计算点之间的距离
        print("\n点之间的距离 (米):")
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(result_ecef[i] - result_ecef[j])
                print(f"  点{i} ↔ 点{j}: {dist:.2f}m")

        sys.exit(0)
    else:
        print("\n✗ 测试失败: 点重合!")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ 测试出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
