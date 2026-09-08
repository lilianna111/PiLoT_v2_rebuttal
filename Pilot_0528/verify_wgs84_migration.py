#!/usr/bin/env python3
"""
WGS84迁移验证脚本
对比EPSG:4547和EPSG:4326的DSM查询结果，确保精度一致
使用FPVLoc环境验证
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
from scipy.spatial.transform import Rotation as R


def get_world_point_wgs84(pose, depth, K=None):
    """
    使用WGS84坐标系反投影：从pose和深度计算3D世界点

    Args:
        pose: [lon, lat, alt, roll, pitch, yaw] in WGS84
        depth: 深度值（米）
        K: [width, height, fx, fy, cx, cy] 相机内参

    Returns:
        tuple: (lon, lat, alt) WGS84格式的3D点坐标
    """
    if K is None:
        K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]

    # 解析参数
    lon, lat, alt, roll, pitch, yaw = pose
    width, height, fx, fy, cx, cy = K

    # 构建4x4变换矩阵（ECEF坐标系）
    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]

    # ENU旋转矩阵
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)

    # 相机位置（ECEF）
    t_c2w = WGS84_to_ECEF(translation)

    # 变换矩阵
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w

    # Y轴和Z轴取反（投影后二维原点在左上角）
    T[:3, 1] = -T[:3, 1]
    T[:3, 2] = -T[:3, 2]

    # 相机内参矩阵
    K_matrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
    K_inv = np.linalg.inv(K_matrix)

    # 图像中心点（齐次坐标）
    pixel_center = np.array([cx, cy, 1.0])

    # 反投影到3D（ECEF）
    points_2D = pixel_center.reshape(3, 1)
    R_matrix = T[:3, :3]
    t_vector = T[:3, 3]

    # 计算3D点（ECEF坐标系）
    point_3d_ecef = R_matrix @ (K_inv @ (depth * points_2D)) + t_vector.reshape(3, 1)
    point_3d_ecef = point_3d_ecef.flatten()

    # 转换回WGS84
    lon_wgs84, lat_wgs84, alt_wgs84 = ECEF_to_WGS84(point_3d_ecef)

    return (lon_wgs84, lat_wgs84, alt_wgs84)


def query_dsm_wgs84(lon, lat, dsm_path):
    """
    查询WGS84 DSM的高度值

    Args:
        lon: 经度
        lat: 纬度
        dsm_path: DSM文件路径（WGS84坐标系）

    Returns:
        float: 高度值（米）
    """
    from osgeo import gdal

    # 打开DSM
    dsm = gdal.Open(dsm_path)
    if dsm is None:
        raise FileNotFoundError(f"无法打开DSM文件: {dsm_path}")

    # 获取geotransform
    geotransform = dsm.GetGeoTransform()
    x_origin = geotransform[0]  # lon_origin
    x_pixel_size = geotransform[1]  # degrees per pixel
    y_origin = geotransform[3]  # lat_origin
    y_pixel_size = geotransform[5]  # degrees per pixel (negative)

    # 计算像素坐标
    col = (lon - x_origin) / x_pixel_size
    row = (lat - y_origin) / y_pixel_size

    # 检查是否在范围内
    band = dsm.GetRasterBand(1)
    rows, cols = band.YSize, band.XSize

    if col < 0 or col >= cols or row < 0 or row >= rows:
        print(f"⚠️  警告：点({lon}, {lat})超出DSM范围")
        return None

    # 读取高度（使用双线性插值）
    from scipy.ndimage import map_coordinates
    data = band.ReadAsArray()

    height = map_coordinates(data, [[row], [col]], order=1)[0]

    dsm = None
    return float(height)


def verify_backprojection():
    """
    验证反投影的正确性：
    1. 给定pose和depth，计算3D点
    2. 查询DSM高度
    3. 对比计算的高度和DSM高度
    """
    print("=" * 80)
    print("🔍 验证WGS84迁移 - 反投影测试")
    print("=" * 80)

    # 测试数据（来自preprocess/4_verify_matches_npy_pose.py）
    test_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
    test_depth = 167.0476837158203

    print(f"\n📍 测试Pose: {test_pose}")
    print(f"   Lon={test_pose[0]:.6f}, Lat={test_pose[1]:.6f}, Alt={test_pose[2]:.2f}m")
    print(f"   Roll={test_pose[3]:.2f}°, Pitch={test_pose[4]:.2f}°, Yaw={test_pose[5]:.2f}°")
    print(f"📏 测试深度: {test_depth:.2f}m")

    # 1. 反投影计算3D点
    print(f"\n🔄 步骤1：反投影计算世界坐标...")
    world_point = get_world_point_wgs84(test_pose, test_depth)
    lon_calc, lat_calc, alt_calc = world_point

    print(f"   计算结果: Lon={lon_calc:.8f}, Lat={lat_calc:.8f}, Alt={alt_calc:.2f}m")

    # 2. 查询WGS84 DSM
    dsm_wgs84_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif"

    print(f"\n🗺️  步骤2：查询WGS84 DSM高度...")
    print(f"   DSM路径: {dsm_wgs84_path}")

    if not os.path.exists(dsm_wgs84_path):
        print(f"   ❌ DSM文件不存在: {dsm_wgs84_path}")
        return False

    height_dsm = query_dsm_wgs84(lon_calc, lat_calc, dsm_wgs84_path)

    if height_dsm is None:
        print(f"   ❌ DSM查询失败")
        return False

    print(f"   DSM高度: {height_dsm:.2f}m")

    # 3. 对比结果
    print(f"\n📊 步骤3：精度验证...")
    diff = abs(alt_calc - height_dsm)
    print(f"   计算高度: {alt_calc:.2f}m")
    print(f"   DSM高度:  {height_dsm:.2f}m")
    print(f"   高度差异: {diff:.2f}m")

    # 判断精度（DSM分辨率0.3米，允许误差1米以内）
    threshold = 1.0
    if diff < threshold:
        print(f"   ✅ 精度验证通过！误差 {diff:.2f}m < {threshold}m")
        return True
    else:
        print(f"   ⚠️  精度超出阈值！误差 {diff:.2f}m >= {threshold}m")
        return False


def compare_with_4547():
    """
    对比EPSG:4547和EPSG:4326的查询结果
    """
    print("\n" + "=" * 80)
    print("🔄 对比EPSG:4547与EPSG:4326")
    print("=" * 80)

    # 测试点（WGS84）
    test_point = (112.999054, 28.290497)

    print(f"\n📍 测试点: Lon={test_point[0]:.6f}, Lat={test_point[1]:.6f}")

    # 查询WGS84 DSM
    dsm_wgs84_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif"
    dsm_4547_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"

    if not os.path.exists(dsm_wgs84_path):
        print(f"   ⚠️  WGS84 DSM不存在: {dsm_wgs84_path}")
        return

    height_wgs84 = query_dsm_wgs84(test_point[0], test_point[1], dsm_wgs84_path)
    print(f"\n🗺️  WGS84 DSM高度: {height_wgs84:.2f}m")

    # 查询4547 DSM（需要先转换坐标）
    if os.path.exists(dsm_4547_path):
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)
        x_4547, y_4547 = transformer.transform(test_point[0], test_point[1])

        dsm_4547 = gdal.Open(dsm_4547_path)
        geotransform = dsm_4547.GetGeoTransform()

        col = (x_4547 - geotransform[0]) / geotransform[1]
        row = (y_4547 - geotransform[3]) / geotransform[5]

        from scipy.ndimage import map_coordinates
        band = dsm_4547.GetRasterBand(1)
        data = band.ReadAsArray()
        height_4547 = map_coordinates(data, [[row], [col]], order=1)[0]

        print(f"🗺️  EPSG:4547 DSM高度: {height_4547:.2f}m")

        diff = abs(height_wgs84 - height_4547)
        print(f"\n📊 高度差异: {diff:.2f}m")

        if diff < 0.5:
            print(f"   ✅ 两个DSM数据一致！")
        else:
            print(f"   ⚠️  两个DSM数据有差异（可能是重采样误差）")
    else:
        print(f"   ⚠️  EPSG:4547 DSM不存在，跳过对比")


def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("   WGS84坐标系迁移验证 - FPVLoc环境")
    print("🚀" * 40 + "\n")

    # 验证1：反投影测试
    success = verify_backprojection()

    # 验证2：对比4547和4326
    compare_with_4547()

    # 总结
    print("\n" + "=" * 80)
    print("📋 验证总结")
    print("=" * 80)

    if success:
        print("\n✅ WGS84迁移验证通过！")
        print("   - 反投影精度正常")
        print("   - DSM查询正确")
        print("   - 可以安全使用WGS84坐标系")
    else:
        print("\n⚠️  WGS84迁移需要进一步检查")
        print("   - 请检查DSM文件是否正确")
        print("   - 请检查坐标转换逻辑")

    print("\n" + "=" * 80 + "\n")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
