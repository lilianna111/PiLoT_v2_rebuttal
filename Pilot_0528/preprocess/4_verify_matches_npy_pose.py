import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R

# 1. 导入项目中的 transform 模块 (保证坐标系转换一致)
from transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
def get_world_point(pose, depth):
    """
    根据pose、相机内参K和深度值，恢复图像中心点的3D世界坐标
    
    Args:
        pose: [lon, lat, alt, roll, pitch, yaw] 姿态信息
        K: [width, height, fx, fy, cx, cy] 相机内参
        depth: 深度值（米）
    
    Returns:
        tuple: (lon, lat, alt) WGS84格式的3D点坐标
    """
    K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0] 
    # 1. 解析pose和K参数
    lon, lat, alt, roll, pitch, yaw = pose
    width, height, fx, fy, cx, cy = K
    
    # 2. 构建4x4变换矩阵T（从相机到世界，ECEF坐标系）
    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation)
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    # Y轴和Z轴取反（投影后二维原点在左上角）
    T[:3, 1] = -T[:3, 1]
    T[:3, 2] = -T[:3, 2]
    
    # 3. 构建相机内参矩阵K并求逆
    K_matrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
    K_inv = np.linalg.inv(K_matrix)
    
    # 4. 使用图像中心点作为像素坐标（齐次坐标）
    pixel_center = np.array([cx, cy, 1.0])
    
    # 5. 反投影到3D（参考get_3D_samples的实现）
    # Points_3D = R @ (K_inv @ (depth * points_2D)) + t
    points_2D = pixel_center.reshape(3, 1)
    R_matrix = T[:3, :3]
    t_vector = T[:3, 3]
    
    # 计算3D点（ECEF坐标系）
    point_3d_ecef = R_matrix @ (K_inv @ (depth * points_2D)) + t_vector.reshape(3, 1)
    point_3d_ecef = point_3d_ecef.flatten()
    
    # 6. 转换回WGS84格式
    lon_wgs84, lat_wgs84, alt_wgs84 = ECEF_to_WGS84(point_3d_ecef)
    
    return (lon_wgs84, lat_wgs84, alt_wgs84)

# =========================================================================
# 测试调用
# =========================================================================
if __name__ == "__main__":
    # 模拟输入数据
    # [Lon, Lat, Alt, Roll, Pitch, Yaw]
    my_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803] 

    
    depth = 167.0476837158203 # 假设中心点深度为50米
    
    result = get_world_point(my_pose, depth)
    print(f"计算出的世界坐标 (WGS84): {result}")

