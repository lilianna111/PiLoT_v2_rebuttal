"""
新的射线投射实现：在局部ENU坐标系进行
"""
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

def caculate_predictXYZ_batch_enu(K, pose, obj_pixel_coords, area, geotransform, area_minZ, num_sample):
    """
    在局部ENU坐标系进行射线投射，避免ECEF数值精度问题

    流程:
    1. 将相机pose转换到局部ENU坐标系（原点在相机位置）
    2. 在ENU空间进行射线投射（类似4547平面坐标）
    3. ENU采样点 → ECEF → WGS84 → 查4326 DSM
    4. 找到最佳匹配点，转换为ECEF输出
    """
    coords = np.asarray(obj_pixel_coords, dtype=np.float64).reshape(-1, 2)

    # 解析pose
    raw_lon, raw_lat, raw_alt = pose[0], pose[1], pose[2]
    raw_roll, raw_pitch, raw_yaw = pose[3], pose[4], pose[5]

    # 构建ENU到ECEF的转换
    wgs84_to_ecef = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    ecef_to_wgs84 = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    # 相机中心的ECEF坐标
    cam_ecef = np.array(wgs84_to_ecef.transform(raw_lon, raw_lat, raw_alt))

    # 计算ENU到ECEF的旋转矩阵
    lat_rad = np.radians(raw_lat)
    lon_rad = np.radians(raw_lon)

    # ENU基向量在ECEF中的表示
    up = np.array([
        np.cos(lon_rad) * np.cos(lat_rad),
        np.sin(lon_rad) * np.cos(lat_rad),
        np.sin(lat_rad)
    ])
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    north = np.cross(up, east)
    R_enu_to_ecef = np.column_stack([east, north, up])

    # 相机姿态在ENU中的旋转
    R_cam_in_enu = R.from_euler('xyz', [raw_pitch, raw_roll, raw_yaw], degrees=True).as_matrix()

    # Y轴和Z轴取反（OpenGL风格）
    coord_transform = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float64)
    R_cam_in_enu = R_cam_in_enu @ coord_transform

    # 在ENU空间中，相机位于原点
    cam_center_enu = np.array([0.0, 0.0, 0.0])

    # 反投影到相机坐标系
    K_matrix = np.array([
        [K[2], 0, K[4]],
        [0, K[3], K[5]],
        [0, 0, 1]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K_matrix)

    pixels = np.concatenate(
        [coords, np.ones((coords.shape[0], 1), dtype=np.float64)], axis=1
    )
    p_camera = (K_inv @ pixels.T).T

    # 转换到ENU空间
    targets_enu = (R_cam_in_enu @ p_camera.T).T + cam_center_enu[None, :]

    # 射线方向
    ray_dirs_enu = targets_enu - cam_center_enu[None, :]

    # 计算与参考高度平面的交点（在ENU空间中，相机在原点，高度=0）
    # area_minZ是WGS84高度，需要转换为相对于相机的ENU高度
    z_ref_enu = area_minZ - raw_alt  # 参考平面相对于相机的高度

    z_denom = ray_dirs_enu[:, 2]
    z_denom = np.where(np.abs(z_denom) < 1e-12, 1e-12, z_denom)
    scale_to_ground = (z_ref_enu - cam_center_enu[2]) / z_denom
    intersections_enu = cam_center_enu[None, :] + ray_dirs_enu * scale_to_ground[:, None]

    # 在ENU空间沿射线采样
    alpha = np.linspace(0.0, 1.0, num_sample, dtype=np.float64)[None, :]

    x_enu = cam_center_enu[0] + (intersections_enu[:, 0:1] - cam_center_enu[0]) * alpha
    y_enu = cam_center_enu[1] + (intersections_enu[:, 1:2] - cam_center_enu[1]) * alpha
    z_enu = cam_center_enu[2] + (intersections_enu[:, 2:3] - cam_center_enu[2]) * alpha

    # ENU → ECEF
    enu_samples = np.stack([x_enu.ravel(), y_enu.ravel(), z_enu.ravel()], axis=1)
    ecef_samples = (R_enu_to_ecef @ enu_samples.T).T + cam_ecef[None, :]

    # ECEF → WGS84
    lon_samples, lat_samples, alt_samples = ecef_to_wgs84.transform(
        ecef_samples[:, 0], ecef_samples[:, 1], ecef_samples[:, 2]
    )
    lon_samples = lon_samples.reshape(x_enu.shape)
    lat_samples = lat_samples.reshape(x_enu.shape)
    alt_samples = alt_samples.reshape(x_enu.shape)

    # 查询4326 DSM
    lon_origin, lon_pixel_size, _, lat_origin, _, lat_pixel_size = geotransform
    col = ((lon_samples - lon_origin) / lon_pixel_size).astype(int)
    row = ((lat_samples - lat_origin) / lat_pixel_size).astype(int)
    sample_height = map_coordinates(area, [row, col], order=1)

    # 找到最接近DSM高度的点
    abs_diff = np.abs(alt_samples - sample_height)
    min_idx = np.argmin(abs_diff, axis=1)
    batch_idx = np.arange(coords.shape[0])

    # 获取最佳匹配点的ECEF坐标
    ecef_samples_reshaped = ecef_samples.reshape(x_enu.shape[0], x_enu.shape[1], 3)
    result_ecef = ecef_samples_reshaped[batch_idx, min_idx]
    result_sample_height = sample_height[batch_idx, min_idx]

    # 计算k_value
    origin_x_enu = cam_center_enu[0]
    origin_y_enu = cam_center_enu[1]
    target_x_enu = targets_enu[:, 0]
    target_y_enu = targets_enu[:, 1]

    result_enu = enu_samples.reshape(x_enu.shape[0], x_enu.shape[1], 3)[batch_idx, min_idx]
    result_x_enu = result_enu[:, 0]
    result_y_enu = result_enu[:, 1]

    use_x = np.abs(target_x_enu - origin_x_enu) > 1e-6
    k_value = np.empty(coords.shape[0], dtype=np.float64)
    k_value[use_x] = (result_x_enu[use_x] - origin_x_enu) / (target_x_enu[use_x] - origin_x_enu)
    k_value[~use_x] = (result_y_enu[~use_x] - origin_y_enu) / (target_y_enu[~use_x] - origin_y_enu)

    return result_ecef, k_value, result_sample_height
