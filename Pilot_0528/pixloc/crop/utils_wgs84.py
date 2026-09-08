import math
import yaml
import glob
import os
import numpy as np
from osgeo import gdal
from pathlib import Path
import time
import rasterio
import json
from scipy.spatial.transform import Rotation 
import pyproj
import itertools
# from immatch.utils.mountain_detection import detect_mountain_terrain, print_mountain_detection_result, is_mountain_terrain
def w2c_to_c2w(rot_matrix_w2c, trans_w2c):
    """
    Convert world-to-camera pose to camera-to-world pose
    Args:
        quat_w2c: quaternion in w,x,y,z format for w2c rotation
        trans_w2c: translation vector for w2c
    Returns:
        quat_c2w: quaternion in w,x,y,z format for c2w rotation
        trans_c2w: translation vector for c2w
    """
    
    # Inverse rotation is transpose
    rot_matrix_c2w = rot_matrix_w2c.T
    
    # Inverse translation: -R.T @ t
    trans_c2w = -rot_matrix_c2w @ trans_w2c
    
    # Convert back to quaternion
    rot_c2w = Rotation.from_matrix(rot_matrix_c2w)
    quat_c2w_scipy = rot_c2w.as_quat(canonical=True)  # Returns x,y,z,w
    
    # Convert back to w,x,y,z format
    quat_c2w = np.array([quat_c2w_scipy[3], quat_c2w_scipy[0], quat_c2w_scipy[1], quat_c2w_scipy[2]])
    
    return quat_c2w, trans_c2w

def generate_dji_seed(q_intrinsic, q_prior_pose, q_name, osg_angle,
    x_interval,  # interval in meters for X axis
    y_interval,  # interval in meters for Y axis
    z_interval,  # interval in meters for Z axis
    yaw_interval,  # interval in degrees for yaw angle
    num_x_samples,  # number of samples in X direction (should be odd number)
    num_y_samples,  # number of samples in Y direction (should be odd number)
    num_z_samples,  # number of samples in Z direction (should be odd number)
    num_yaw_samples):

    for name, num in [("num_x_samples", num_x_samples), 
                     ("num_y_samples", num_y_samples),
                     ("num_z_samples", num_z_samples),
                     ("num_yaw_samples", num_yaw_samples)]:
        if num % 2 == 0:
            raise ValueError(f"{name} should be an odd number")
    
    # Generate sampling points
    x_samples = np.linspace(-x_interval * (num_x_samples // 2), 
                          x_interval * (num_x_samples // 2), 
                          num_x_samples)
    y_samples = np.linspace(-y_interval * (num_y_samples // 2), 
                          y_interval * (num_y_samples // 2), 
                          num_y_samples)
    z_samples = np.linspace(-z_interval * (num_z_samples // 2),
                          z_interval * (num_z_samples // 2),
                          num_z_samples)
    yaw_samples = np.linspace(-yaw_interval * (num_yaw_samples // 2), 
                            yaw_interval * (num_yaw_samples // 2), 
                            num_yaw_samples)
    
    rot_matrix_w2c, trans_w2c = q_prior_pose[:3, :3], q_prior_pose[:3, 3]
    print('rot_matrix_w2c', rot_matrix_w2c)
    print('trans_w2c', trans_w2c)
    quat_c2w, trans_c2w = w2c_to_c2w(rot_matrix_w2c, trans_w2c)

    reference_pose = {}
    reference_intrinsic = {}
    reference_yaw = {}
    
    # Generate all combinations of x, y, z and yaw
    for i, (x, y, z, yaw) in enumerate(itertools.product(x_samples, y_samples, z_samples, yaw_samples)):
        # Apply translation offset
        new_trans_c2w = trans_c2w.copy()
        new_euler = osg_angle.copy()
        new_trans_c2w[0] += x  # X offset
        new_trans_c2w[1] += y  # Y offset
        new_trans_c2w[2] += z  # Z offset
        t_c2w = new_trans_c2w

        new_euler[0] += yaw 
        # print(new_euler[0], new_euler[1], new_euler[2])
        # ret = Rotation.from_euler('zxy', [new_euler[0], float(90-new_euler[1]), new_euler[2]], degrees=True)
        ret = Rotation.from_euler('xyz', [new_euler[1]-90, float(new_euler[2]), -new_euler[0]], degrees=True)
        new_rot_c2w = ret.as_matrix()
        # qw, qx, qy, qz = rotmat2qvec(new_rot_c2w)
        # q = [qx,qy,qz,qw]
        # new_rot_c2w = np.asmatrix(qvec2rotmat(q))
        # new_left2right = np.diag([1, -1,-1])
        # new_rot_c2w = new_rot_c2w @ new_left2right

       
        # new_rot_c2w = Rotation.from_euler('xyz', new_euler, degrees=True).as_matrix()

        T = np.identity(4)
        T[:3, :3] = new_rot_c2w.T
        T[:3, 3] = -new_rot_c2w.T @ t_c2w
        # T[:3, :3] = rot_matrix_w2c
        # T[:3, 3] = -rot_matrix_w2c @ t_c2w


        name = f"{q_name}_{i}"
        reference_pose[name] = T
        reference_intrinsic[name] = q_intrinsic
        reference_yaw[name] = new_euler[0]

    return reference_pose, reference_intrinsic, reference_yaw


def log_mountain_detection_result(mountain_result, dsm_path):
    """
    将山地检测结果保存到日志文件
    
    Args:
        mountain_result: 山地检测结果字典
        dsm_path: DSM文件路径
    """
    import os
    from datetime import datetime
    
    # 固定日志文件路径
    log_file = "logs/mountain_detection_results.log"
    
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 准备日志内容
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dsm_filename = os.path.basename(dsm_path)
    
    # 转换bool类型为字符串，确保JSON序列化
    mountain_result_json = {}
    for key, value in mountain_result.items():
        if isinstance(value, bool):
            mountain_result_json[key] = "True" if value else "False"
        else:
            mountain_result_json[key] = value
    
    log_entry = {
        "timestamp": timestamp,
        "dsm_file": dsm_filename,
        "dsm_path": dsm_path,
        "mountain_detection": mountain_result_json
    }
    
    # 写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n")
        f.write("-" * 80 + "\n")
    
    print(f"📝 山地检测结果已保存到: {log_file}")


def load_pixel_point(json_path):
    """
    读取 pixel.json 并转为 {图片名: [点列表]} 的字典
    """
    with open(json_path, 'r') as f:
        pixel_data = json.load(f)
    pixel_dict = {}
    for item in pixel_data:
        img_name = item['image'].split('.')[0]  # 去掉扩展名
        pixel_dict[img_name] = item['points']
    return pixel_dict


def load_gt_points(gt_path):
    """
    加载GT.json，返回 {id: [x, y, z]} 的字典
    """
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {}
    for pt in gt_data["points"]:
        gt_dict[str(pt["idx"])] = [pt["x"], pt["y"], pt["z"]]
    return gt_dict

def resize_value(value):

    if 128 <= value <= 256:
        return 256
    elif value < 128:
        return 128
    else:
        return 640

def read_DSM_config(ref_dsm, ref_dom, npy_save_path):

    with rasterio.open(ref_dsm) as dsm_ds:
        dsm_data = dsm_ds.read(1)
        dsm_transform = dsm_ds.transform
    
    with rasterio.open(ref_dom) as dom_ds:
        dom_data = dom_ds.read([1, 2, 3])
    
    start = time.time()
    dataset = gdal.Open(ref_dsm)
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()
    
    if Path(npy_save_path).exists():
        area = np.load(npy_save_path)
    else:
        area = band.ReadAsArray()
        np.save(npy_save_path, area)

    # 屏蔽无效值（-9999），仅保留有效高程进行 min 统计
    valid_mask = (area > 0) & (area < 10000)
    
    # 山地检测
    # mountain_result = detect_mountain_terrain(area, valid_mask)
    # print_mountain_detection_result("DSM分析", mountain_result)
        
    ## 根据地形类型设置不同的area_minZ值
    # if is_mountain_terrain():
    #     # 山地地形：使用75%分位数，避免低洼区域的影响
    #     area_minZ = np.percentile(area[valid_mask], 75)
    #     print(f"山地地形：使用75%分位数作为最低高程 = {area_minZ:.1f}m")
    # else:
        # 平原地形：使用中位数
    # area_minZ = np.median(area[valid_mask]) + 3.58
    area_minZ = np.median(area[valid_mask])
     
    # print(f"平原地形：使用中位数作为最低高程 = {area_minZ:.1f}m")

    # print("data size:", band.XSize, "x", band.YSize)
    print("area_minZ (valid minimum elevation):", area_minZ)
    # print("Loading time cost:", time.time() - start, "s")

   
    return geotransform, area, area_minZ, dsm_data, dsm_transform,dom_data

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_colmap_intrinsic(sensor_w, sensor_h, f_mm, image_w, image_h, rgb_path, intrinsic_txt):

    """
    output: colmap_styles.txt
    name PINHOLE  width height fx fy cx cy
    """
    fx = image_w * f_mm / sensor_w
    fy = image_h * f_mm / sensor_h
    cx = image_w / 2
    cy = image_h / 2

    images_files = glob.glob(os.path.join(rgb_path, "*.[jp][pn]g"))+glob.glob(os.path.join(rgb_path, '*.tif'))
    image_names = [os.path.basename(f) for f in images_files]
    
    with open(intrinsic_txt, 'w') as f:
        for name in image_names:
            f.write(f"{name} PINHOLE {image_w} {image_h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

def compute_focal_length_pitch_correct(H_m, pitch_deg, sensor_width_mm, image_width_px, gsd_m_per_pixel):
    """
    正确计算倾斜成像情况下，为满足目标 GSD 所需的焦距（单位：mm）

    Args:
        H_m (float): 无人机高度（垂直方向，单位：米）
        pitch_deg (float): 俯视角度（度）
        sensor_width_mm (float): 传感器宽度（毫米）
        image_width_px (int): 图像宽度（像素）
        gsd_m_per_pixel (float): 目标 GSD（米/像素）

    Returns:
        float: 所需焦距（毫米）
    """
    pitch_rad = math.radians(pitch_deg)
    cos_theta = math.cos(pitch_rad)

    f_mm = (H_m * sensor_width_mm) / (gsd_m_per_pixel * image_width_px * cos_theta)
    return f_mm


# 示例参数
# H_m = 1000
# pitch_deg = 60
# sensor_width_mm = 7.6
# image_width_px = 1920
# gsd_m_per_pixel = 0.35

# focal_mm = compute_focal_length_pitch_correct(H_m, pitch_deg, sensor_width_mm, image_width_px, gsd_m_per_pixel)
# print(f"✅ 正确计算：在 pitch = {pitch_deg}° 时，所需焦距为：{focal_mm:.2f} mm")

