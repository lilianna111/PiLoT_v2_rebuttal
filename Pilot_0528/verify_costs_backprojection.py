#!/usr/bin/env python3
"""
验证costs.py中的反投影逻辑是否与验证脚本一致
"""
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528')

from preprocess.transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
from scipy.spatial.transform import Rotation as R

def reference_backprojection(pose, depth, K):
    """
    参考标准: 验证脚本的反投影实现
    """
    lon, lat, alt, roll, pitch, yaw = pose
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

    lon_wgs84, lat_wgs84, alt_wgs84 = ECEF_to_WGS84(point_3d_ecef)

    return point_3d_ecef, (lon_wgs84, lat_wgs84, alt_wgs84)


def costs_backprojection(pose, depth, K):
    """
    模拟costs.py中 _center_lonlatalt_torch() 的反投影逻辑
    """
    lon, lat, alt, roll, pitch, yaw = pose
    width, height, fx, fy, cx, cy = K

    # 构建变换矩阵 (与验证脚本相同)
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

    # costs.py的实现: 中心像素反投影到光轴
    # K_inv @ [cx, cy, 1] = [0, 0, 1] (恒等)
    # 因此 cam_vec = [0, 0, depth]
    # ray_world = R_c2w @ [0, 0, depth] = depth * R_c2w[:, 2]

    C_ecef = T[:3, 3]  # 相机中心(ECEF)
    R_matrix = T[:3, :3]
    ray_world = R_matrix[:, 2] * depth  # 第584行的实现

    point_ecef = C_ecef + ray_world  # 第585行

    lon_wgs84, lat_wgs84, alt_wgs84 = ECEF_to_WGS84(point_ecef)

    return point_ecef, (lon_wgs84, lat_wgs84, alt_wgs84)


def verify():
    """
    验证两种方法的结果是否一致
    """
    # 测试数据
    my_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
    depth = 167.0476837158203
    K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]

    print("=" * 80)
    print("验证costs.py反投影逻辑")
    print("=" * 80)

    # 方法1: 参考标准
    ecef_ref, wgs84_ref = reference_backprojection(my_pose, depth, K)
    print("\n[方法1: 验证脚本反投影 - 参考标准]")
    print(f"  输入: pose={my_pose}")
    print(f"  输入: depth={depth:.4f}m")
    print(f"  ECEF: [{ecef_ref[0]:.6f}, {ecef_ref[1]:.6f}, {ecef_ref[2]:.6f}]")
    print(f"  WGS84: lon={wgs84_ref[0]:.10f}, lat={wgs84_ref[1]:.10f}, alt={wgs84_ref[2]:.6f}m")

    # 方法2: costs.py逻辑
    ecef_costs, wgs84_costs = costs_backprojection(my_pose, depth, K)
    print("\n[方法2: costs.py _center_lonlatalt_torch() 逻辑]")
    print(f"  ECEF: [{ecef_costs[0]:.6f}, {ecef_costs[1]:.6f}, {ecef_costs[2]:.6f}]")
    print(f"  WGS84: lon={wgs84_costs[0]:.10f}, lat={wgs84_costs[1]:.10f}, alt={wgs84_costs[2]:.6f}m")

    # 对比差异
    diff_ecef = np.linalg.norm(ecef_ref - ecef_costs)
    diff_lon = abs(wgs84_ref[0] - wgs84_costs[0])
    diff_lat = abs(wgs84_ref[1] - wgs84_costs[1])
    diff_alt = abs(wgs84_ref[2] - wgs84_costs[2])

    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    print(f"  ECEF距离差异: {diff_ecef:.10f}m")
    print(f"  经度差异: {diff_lon:.12f}°")
    print(f"  纬度差异: {diff_lat:.12f}°")
    print(f"  高程差异: {diff_alt:.10f}m")

    # 判断
    tolerance_ecef = 1e-6  # ECEF容差(米)
    tolerance_deg = 1e-10  # 经纬度容差(度)
    tolerance_alt = 1e-6   # 高程容差(米)

    if (diff_ecef < tolerance_ecef and
        diff_lon < tolerance_deg and
        diff_lat < tolerance_deg and
        diff_alt < tolerance_alt):
        print("\n✓ 验证通过: costs.py的反投影逻辑与验证脚本完全一致!")
        return True
    else:
        print("\n✗ 验证失败: 存在差异!")
        return False


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
