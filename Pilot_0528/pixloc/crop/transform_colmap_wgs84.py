import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj


def convert_euler_to_matrix(euler_xyz):
    """
    euler_xyz = [yaw, pitch, roll] 的变体映射，保持你原来的姿态定义方式
    """
    ret = R.from_euler(
        'xyz',
        [euler_xyz[1] - 90, float(euler_xyz[2]), -euler_xyz[0]],
        degrees=True
    )
    return ret.as_matrix()


def wgs84_to_ecef(trans):
    """
    trans = [lon, lat, h]
    返回 ECEF XYZ（单位：米）
    """
    lon, lat, height = trans
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:4978", always_xy=True
    )
    x, y, z = transformer.transform(lon, lat, height)
    return np.array([x, y, z], dtype=np.float64)


def get_enu_to_ecef_rotation(lon_deg, lat_deg):
    """
    返回 ENU -> ECEF 的旋转矩阵
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    east = np.array([
        -np.sin(lon),
         np.cos(lon),
         0.0
    ])

    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat)
    ])

    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

    return np.column_stack([east, north, up])


def transform_colmap_pose_intrinsic(pose_data):
    """
    pose_data = [lon, lat, alt, roll, pitch, yaw]

    返回：
        poses_dict      -> 4x4 world-to-camera 外参（世界坐标现在是 ECEF）
        intrinsics_dict -> 3x3 K
        q_intrinsics_info
        osg_dict
    """
    lon, lat, alt, roll, pitch, yaw = pose_data

    # 保持你原来的角度预处理逻辑
    yaw = -yaw
    pitch = pitch - 90

    euler_xyz = [yaw, pitch, roll]

    # 相机姿态先在局部 ENU 下构建
    r_c2w_local = convert_euler_to_matrix(euler_xyz)

    # 平移：WGS84 -> ECEF
    t_c2w = wgs84_to_ecef([lon, lat, alt])

    # 把局部 ENU 姿态转到 ECEF 世界系
    R_enu_to_ecef = get_enu_to_ecef_rotation(lon, lat)
    r_c2w = R_enu_to_ecef @ r_c2w_local

    # 构造 w2c
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r_c2w.T
    T[:3, 3] = -r_c2w.T @ t_c2w

    poses_dict = T

    # 这里先保持你原来的内参不动
    intrinsics_dict = np.array([
        [1350.0, 0.0, 957.85],
        [0.0, 1350.0, 537.55],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    osg_dict = euler_xyz
    q_intrinsics_info = [1920, 1080, 1920, 1080, 24.00]

    return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict