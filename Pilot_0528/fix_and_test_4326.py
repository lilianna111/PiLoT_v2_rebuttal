#!/usr/bin/env python3
"""
完整的4326迁移测试和修复代码
添加缺失的 predict_points_alt 方法
"""
import sys
sys.path.insert(0, '/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528')

# 读取ray_casting.py，添加predict_points_alt方法
with open('pixloc/crop/ray_casting.py', 'r') as f:
    content = f.read()

# 检查是否已有predict_points_alt方法
if 'def predict_points_alt(' not in content:
    print("添加 predict_points_alt 方法...")

    # 在predict_center_alt方法后添加predict_points_alt
    insert_code = '''
    def predict_points_alt(self, DSM_path, pose, ref_npy_path, geotransform, K,
                          ray_area, ray_area_minZ, num_sample, object_pixel_coords_list):
        """
        批量预测多个像素点的3D坐标（ECEF）

        Args:
            object_pixel_coords_list: [[x1,y1], [x2,y2], ...] 像素坐标列表

        Returns:
            np.array: Nx3 ECEF坐标数组
        """
        result_points, _, _ = self.caculate_predictXYZ_batch(
            K, pose, object_pixel_coords_list,
            ray_area, geotransform, ray_area_minZ, num_sample
        )
        return result_points
'''

    # 找到predict_center_alt方法的结束位置
    import re
    # 查找最后一个方法定义的位置
    pattern = r'(    def predict_center_alt\(.*?\n(?:.*?\n)*?        return.*?\n)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        insert_pos = match.end()
        new_content = content[:insert_pos] + '\n' + insert_code + content[insert_pos:]

        with open('pixloc/crop/ray_casting.py', 'w') as f:
            f.write(new_content)
        print("✓ predict_points_alt 方法已添加")
    else:
        print("✗ 未找到插入位置")
        sys.exit(1)
else:
    print("✓ predict_points_alt 方法已存在")

print("\n测试crop功能...")

# 测试代码
from pixloc.crop.ray_casting import TargetLocation
from osgeo import gdal
import numpy as np

test_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
test_K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
test_pixel_coords = [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]

dsm_path = '/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif'
dsm_map = gdal.Open(dsm_path)
geotransform = dsm_map.GetGeoTransform()
band = dsm_map.GetRasterBand(1)
area = band.ReadAsArray()
area_minZ = np.ma.masked_values(area, -9999).min()

locator = TargetLocation({"ray_casting": {}}, use_dsm=False)

try:
    result_points = locator.predict_points_alt(
        dsm_path, test_pose, dsm_path.replace('.tif', '.npy'),
        geotransform, test_K, area, area_minZ, 4000, test_pixel_coords
    )

    print(f"\n结果:")
    print(f"  返回形状: {result_points.shape}")
    print(f"  4个角点ECEF坐标:")
    for i, pt in enumerate(result_points):
        print(f"    [{i}] {pt}")

    unique_points = np.unique(result_points, axis=0)
    if len(unique_points) == 4:
        print(f"\n✓ 测试通过: 4个角点各不相同!")
        print("✓ 4326迁移修复完成，可以运行主程序测试")
        sys.exit(0)
    else:
        print(f"\n✗ 测试失败: 只有 {len(unique_points)} 个唯一点")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ 测试出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
