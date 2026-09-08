#!/usr/bin/env python3
"""
验证4326迁移后的反投影逻辑是否正确
对比验证脚本的反投影结果和costs.py中的实现
"""
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528')

from preprocess.transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
from scipy.spatial.transform import Rotation as R

def verify_backprojection():
    """
    验证反投影逻辑：
    1. 使用验证脚本的方法计算3D点
    2. 模拟costs.py的反投影逻辑
    3. 对比结果
    """
    # 测试数据（来自验证脚本）
    my_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
    depth = 167.0476837158203
    K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]

    print("=" * 70)
    print("验证4326迁移后的反投影逻辑")
    print("=" * 70)

    # ========== 方法1: 验证脚本的反投影（参考标准） ==========
    lon, lat, alt, roll, pitch, yaw = my_pose
    width, height, fx, fy, cx, cy = K

    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation)
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    T[:3, 1] = -T[:3, 1]
    T[:3, 2] = -T[:3, 2]

    K_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K_matrix)
    pixel_center = np.array([cx, cy, 1.0])

    points_2D = pixel_center.reshape(3, 1)
    R_matrix = T[:3, :3]
    t_vector = T[:3, 3]

    point_3d_ecef = R_matrix @ (K_inv @ (depth * points_2D)) + t_vector.reshape(3, 1)
    point_3d_ecef = point_3d_ecef.flatten()
    lon_ref, lat_ref, alt_ref = ECEF_to_WGS84(point_3d_ecef)

    print("\n[方法1: 验证脚本反投影 - 参考标准]")
    print(f"  输入: pose={my_pose}")
    print(f"  输入: depth={depth:.4f}m")
    print(f"  ECEF: [{point_3d_ecef[0]:.4f}, {point_3d_ecef[1]:.4f}, {point_3d_ecef[2]:.4f}]")
    print(f"  WGS84: lon={lon_ref:.8f}, lat={lat_ref:.8f}, alt={alt_ref:.4f}m")

    # ========== 方法2: 模拟costs.py的反投影逻辑 ==========
    # 模拟 _center_lonlatalt_torch() 的实现
    # pose已经是ECEF的相机中心
    C_ecef = t_c2w

    # 中心像素反投影: ray_world = R_c2w[:, :, 2] * depth
    ray_world = R_c2w[:, 2] * depth

    # point_ecef = C_ecef + ray_world
    point_ecef_costs = C_ecef + ray_world

    lon_costs, lat_costs, alt_costs = ECEF_to_WGS84(point_ecef_costs)

    print("\n[方法2: costs.py反投影逻辑]")
    print(f"  相机中心(ECEF): [{C_ecef[0]:.4f}, {C_ecef[1]:.4f}, {C_ecef[2]:.4f}]")
    print(f"  射线(ECEF): [{ray_world[0]:.4f}, {ray_world[1]:.4f}, {ray_world[2]:.4f}]")
    print(f"  3D点(ECEF): [{point_ecef_costs[0]:.4f}, {point_ecef_costs[1]:.4f}, {point_ecef_costs[2]:.4f}]")
    print(f"  WGS84: lon={lon_costs:.8f}, lat={lat_costs:.8f}, alt={alt_costs:.4f}m")

    # ========== 对比结果 ==========
    diff_lon = abs(lon_ref - lon_costs)
    diff_lat = abs(lat_ref - lat_costs)
    diff_alt = abs(alt_ref - alt_costs)
    diff_ecef = np.linalg.norm(point_3d_ecef - point_ecef_costs)

    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    print(f"  经度差异: {diff_lon:.10f}°")
    print(f"  纬度差异: {diff_lat:.10f}°")
    print(f"  高程差异: {diff_alt:.6f}m")
    print(f"  ECEF距离差异: {diff_ecef:.6f}m")

    # 判断是否通过
    tolerance_deg = 1e-6  # 经纬度容差
    tolerance_alt = 0.01  # 高程容差(米)

    if diff_lon < tolerance_deg and diff_lat < tolerance_deg and diff_alt < tolerance_alt:
        print("\n✓ 验证通过: 反投影逻辑正确!")
        return True
    else:
        print("\n✗ 验证失败: 反投影逻辑存在误差!")
        return False

if __name__ == "__main__":
    success = verify_backprojection()
    sys.exit(0 if success else 1)
