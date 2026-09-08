import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj
import re
import os
import glob
# from immatch.utils.read_exif import read_exif_data


def convert_euler_to_matrix(euler_xyz):
    """
    Convert Euler angles in 'xyz' order to a rotation matrix.
    
    Parameters:
    euler_xyz (list or numpy.ndarray): The Euler angles in 'xyz' order in degrees.seed_num_yaw
    yaw, pitch roll
    Returns:
    numpy.ndarray: The rotation matrix as a 3x3 numpy array.
    """
    # Convert the Euler angles from degrees to radians
    # float(euler_xyz[1]-90)
    # ret = R.from_euler('zxy', [float(euler_xyz[0]), 180, float(euler_xyz[2])], degrees=True)
    # ret = R.from_euler('zxy', [float(euler_xyz[0]), float(euler_xyz[1]-90), float(euler_xyz[2])], degrees=True)
    ret = R.from_euler('xyz', [euler_xyz[1]-90, float(euler_xyz[2]), -euler_xyz[0]], degrees=True) #pitch roll yaw
    # print('---------', float(euler_xyz[1]-90),'---------')
    rotation_matrix = ret.as_matrix()
    
    return rotation_matrix

def wgs84tocgcs2000(trans):
    """Convert coordinates from WGS84 to CGCS2000.
    
    Args:
        trans (list): [lon, lat, height] in WGS84 format
    """
    lon, lat, height = trans  # Unpack the WGS84 coordinates
    
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 CRS definition
    # cgcs2000 = pyproj.CRS('EPSG:4544')  # 需要修改，现在是阿拉善盟
    cgcs2000 = pyproj.CRS('EPSG:4547')  # 需要修改，现在是长沙
    
    # Create a transformer from WGS84 to CGCS2000
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    
    # Perform the transformation
    x, y = transformer.transform(lon, lat)
    
    # Return the transformed coordinates as a list
    return [x, y, height]  # Keep the original height from WGS84  

# def transform_colmap_pose_intrinsic(img_path):
#     """
#     Transform WGS84 pose to CGCS2000 pose.
#     """
#     exif_data = read_exif_data(img_path)
#     poses_dict = {}
#     osg_dict = {}
#     intrinsics_dict = {}
#     q_intrinsics_info = {}
#     for name in exif_data.keys():
#         # pitch,roll,yaw
#         # euler_xyz = [exif_data[name]['pitch'], exif_data[name]['roll'], exif_data[name]['yaw']]
#         euler_xyz = [exif_data[name]['yaw'], exif_data[name]['pitch'], exif_data[name]['roll']]
#         r_c2w = convert_euler_to_matrix(euler_xyz)
#         trans = [exif_data[name]['longitude'], exif_data[name]['latitude'], exif_data[name]['altitude']]
#         t_c2w = wgs84tocgcs2000(trans)
#         print('投影坐标',t_c2w,'------------------------')
#         T = np.identity(4)
#         T[:3, :3] = r_c2w.T
#         T[:3, 3] = -r_c2w.T @ t_c2w
#         poses_dict[name] = T
#         f_mm = exif_data[name]['focal_value']
#         osg_dict[name] = euler_xyz
#         img_w, img_h = exif_data[name]['img_w'], exif_data[name]['img_h']
#         if f_mm == 6.73:
#             # width_lens
#             sensor_w, sensor_h = 9.69, 7.27
#         elif img_w == 8064:
#             # medium_lens
#             sensor_w, sensor_h = 9.69, 7.27
#         elif img_w == 8192:
#             # long_lens
#             sensor_w, sensor_h = 8.29, 6.23

#         ###try
#         elif f_mm == 40:
#             sensor_w, sensor_h = 8.29, 6.23
#             print('------------------40mm lens------------------')
#         elif f_mm == 19.35:
#             sensor_w, sensor_h = 9.69, 7.27
#             print('------------------19.35mm lens------------------')
#         ###
        
#         elif (f_mm == 12) and (img_w == 640):
#             # thermal_lens
#             sensor_w, sensor_h = 7.68, 6.14
#         else:
#             raise ValueError(f"Unknown focal length: {f_mm}")
#         fx = f_mm * img_w / sensor_w
#         fy = f_mm * img_h / sensor_h
#         cx = img_w / 2
#         cy = img_h / 2
#         intrinsics_dict[name] = np.array([
#                 [fx, 0.0, cx],
#                 [0.0, fy, cy],
#                 [0.0, 0.0, 1.0],
#             ])
#         q_intrinsics_info[name] = [img_w, img_h, sensor_w, sensor_h, f_mm]
#     return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict
def calculate_intrinsic_matrix(equivalent_focal_length, ir_gain_mode):
    """
    计算三摄系统的内参矩阵（先高度裁切16:9，再数字变焦，最后缩放到1920x1080）
    
    参数:
        equivalent_focal_length (float): 等效35mm焦距(mm)
    
    返回:
        dict: 包含相机类型和内参矩阵的信息
    """
    # 定义相机参数
    cameras = {
        'wide': {
            'sensor_size': (9.69, 7.27),  # mm (宽, 高)
            'pixel_size': 1.197e-3,       # mm/像素
            'resolution': (8064, 6048),    # 像素 (宽, 高)
            'real_focal_length': 6.73,     # mm
            'min_equivalent_fl': 24.0,
            'max_equivalent_fl': 81.1,
            'full_frame_equivalent': 6.73 * (43.266/np.sqrt(9.69**2 + 7.27**2))  # 24mm
        },
        'medium': {
            'sensor_size': (9.69, 7.27),
            'pixel_size': 1.197e-3,
            'resolution': (8064, 6048),
            'real_focal_length': 19.35,
            'min_equivalent_fl': 81.2,
            'max_equivalent_fl': 172.1,
            'full_frame_equivalent': 19.35 * (43.266/np.sqrt(9.69**2 + 7.27**2))  # ~69mm
        },
        'tele': {
            'sensor_size': (8.29, 6.23),
            'pixel_size': 1.008e-3,
            'resolution': (8192, 6144),
            'real_focal_length': 40.0,
            'min_equivalent_fl': 172.2,
            'max_equivalent_fl': float('inf'),
            'full_frame_equivalent': 40.0 * (43.266/np.sqrt(8.29**2 + 6.23**2))  # ~172mm
        },
        ##thermal相机
        'thermal': {
            'sensor_size': (7.68, 6.14),   # mm
            'pixel_size': 0.012 ,           # mm/像素 (常见红外探测器是 17µm)
            'resolution': (640, 512),      # 红外分辨率
            'real_focal_length': 12.0,     # mm
            'min_equivalent_fl': 52.7,        # 
            'max_equivalent_fl': 52.7,      # 你自己定义
            'full_frame_equivalent': 12.0 * (43.266/np.sqrt(7.68**2 + 6.14**2))
        }

    }
    
    # import pdb;pdb.set_trace()
    # 确定使用的相机
    camera_type = None
    if ir_gain_mode is not None and len(ir_gain_mode) > 0 and ir_gain_mode[0] is not None:
        camera_type = "thermal"
        # print(camera_type)
        K = np.array([
            [1125, 0, 320],
            [0, 1125, 256],
            [0, 0, 1]
        ])
        return K
        
    else:
        for name, cam in cameras.items():
            if cam['min_equivalent_fl'] <= equivalent_focal_length <= cam['max_equivalent_fl']:
                camera_type = name
                break
    
    if camera_type is None:
        raise ValueError(f"无效的等效焦距: {equivalent_focal_length}mm")
    # print(f"使用相机类型: {camera_type}")
    cam = cameras[camera_type]
    
    # 1. 在全分辨率下先在高度上裁切为16:9
    original_width, original_height = cam['resolution']
    target_aspect = 16.0 / 9.0
    
    # 计算需要保留的高度
    new_height = original_width / target_aspect
    height_crop_ratio = new_height / original_height
    
    # 裁切后的分辨率
    cropped_resolution = (original_width, int(original_height * height_crop_ratio))
    
    # 2. 计算数字变焦因子
    if equivalent_focal_length > cam['min_equivalent_fl']:
        # 计算相对于该相机最小等效焦距的变焦倍数
        zoom_factor = cam['full_frame_equivalent'] / equivalent_focal_length
    else:
        zoom_factor = 1.0
    
    # 3. 计算最终缩放到1920x1080的总缩放比例
    # 数字变焦后的分辨率
    zoomed_width = cropped_resolution[0] * zoom_factor
    zoomed_height = cropped_resolution[1] * zoom_factor
    
    # 计算到目标分辨率的缩放比例
    scale_x = 1920 / zoomed_width
    scale_y = 1080 / zoomed_height
    
    # 4. 计算内参矩阵
    # 原始焦距(像素单位)
    fx = cam['real_focal_length'] / cam['pixel_size']
    fy = fx
    
    # 应用高度裁切
    # fy *= height_crop_ratio
    
    # 应用数字变焦
    fx *= zoom_factor
    fy *= zoom_factor
    
    # 应用最终缩放
    fx *= scale_x
    fy *= scale_y
    
    # 主点固定在(960,540)
    cx = 960
    cy = 540
    
    # 构建内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # return {
    #     'camera_type': camera_type,
    #     'intrinsic_matrix': K,
    #     'equivalent_focal_length': equivalent_focal_length,
    #     'real_focal_length': cam['real_focal_length'],
    #     'zoom_factor': zoom_factor,
    #     'height_crop_ratio': height_crop_ratio,
    #     'processing_steps': {
    #         'original_resolution': cam['resolution'],
    #         'cropped_resolution': cropped_resolution,
    #         'zoomed_resolution': (zoomed_width, zoomed_height),
    #         'final_resolution': (1920, 1080),
    #         'scale_factors': (scale_x, scale_y)
    #     }
    # }
    return K
def extract_parameters_video(line):
    # 定义正则表达式模式，包括可能缺失的字段
    patterns = {
        'focal_len': r'focal_len:\s*([\d.]+)',
        'latitude': r'latitude:\s*([\d.-]+)',
        'longitude': r'longitude:\s*([\d.-]+)',
        'abs_alt': r'abs_alt:\s*([\d.]+)',
        'gb_yaw': r'gb_yaw:\s*([\d.-]+)',
        'gb_pitch': r'gb_pitch:\s*([\d.-]+)',
        'gb_roll': r'gb_roll:\s*([\d.-]+)',
        'ir_gain_mode': r'ir_gain_mode:\s*([\d.-]+)'   # 有可能缺失
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                result[key] = None
        else:
            result[key] = None   # 没匹配到就赋值 None
    
    return result



def transform_colmap_pose_intrinsic(img_path, txt_path):
    """
    将 WGS84 坐标的相机位姿和相机内参转换为 CGCS2000 坐标系，并生成对应字典。
    按图片名匹配 txt 中的参数，并且字典 key 不带扩展名。
    
    参数：
    - img_path: 图片所在目录
    - txt_path: 包含相机参数的 txt 文件路径，每行开头是图片名，如：
        20250814_112155953.jpg: FrameCnt: 0 2025-08-14 ... [focal_len: 24.00] [latitude: ...] ...
    
    返回：
    - poses_dict: {图片名(不带扩展名): 4x4 外参矩阵 T}（相机在 CGCS2000 坐标系下的位姿）
    - intrinsics_dict: {图片名: 3x3 相机内参矩阵 K}
    - q_intrinsics_info: {图片名: [宽, 高, 焦距]}
    - osg_dict: {图片名: [yaw, pitch, roll]} 欧拉角
    """

    # ----------------------------
    # 1. 获取图片列表（不带扩展名）
    # ----------------------------
    all_img_path = glob.glob(os.path.join(img_path, "*.[jp][pn]g"))
    # 去掉路径和扩展名
    all_img_names = [os.path.splitext(os.path.basename(p))[0] for p in all_img_path]

    # ----------------------------
    # 2. 读取 txt 文件，建立图片名到参数字典（去掉扩展名）
    # ----------------------------
    txt_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割出图片名和参数字符串
            img_name_full, param_str = line.split(":", 1)
            img_name = os.path.splitext(img_name_full)[0]  # 去掉扩展名
            result = extract_parameters_video(param_str)
            txt_dict[img_name] = result

    # ----------------------------
    # 3. 初始化输出字典
    # ----------------------------
    poses_dict = {}         # {图片名: 外参矩阵 T}
    intrinsics_dict = {}    # {图片名: 相机内参矩阵 K}
    q_intrinsics_info = {}  # {图片名: [宽, 高, 焦距]}
    osg_dict = {}           # {图片名: [yaw, pitch, roll]}

    # ----------------------------
    # 4. 遍历图片，根据名字查参数
    # ----------------------------
    for img_name in all_img_names:
        if img_name not in txt_dict:
            print(f"警告：{img_name} 在 txt 中找不到参数")
            continue

        result = txt_dict[img_name]

        # 4.1 提取欧拉角姿态
        euler_xyz = [result['gb_yaw'], result['gb_pitch'], result['gb_roll']]
        r_c2w = convert_euler_to_matrix(euler_xyz)  # 欧拉角转旋转矩阵

        # 4.2 提取 WGS84 坐标并转换为 CGCS2000
        trans = [result['longitude'], result['latitude'], result['abs_alt']]
        t_c2w = wgs84tocgcs2000(trans)

        # 4.3 构造 4x4 外参矩阵 T
        T = np.identity(4)
        T[:3, :3] = r_c2w.T
        T[:3, 3] = -r_c2w.T @ t_c2w
        poses_dict[img_name] = T

        # 4.4 计算相机内参矩阵 K
        f_mm = result['focal_len']
        ir_gain_mode = [result.get('ir_gain_mode', None)]
        intrinsics_dict[img_name] = calculate_intrinsic_matrix(f_mm, ir_gain_mode)

        # 4.5 记录欧拉角和基础内参信息
        osg_dict[img_name] = euler_xyz
        q_intrinsics_info[img_name] = [1920, 1080, 1920, 1080, f_mm]  # 假设图片固定分辨率 1920x1080
        # [img_w, img_h, sensor_w, sensor_h, f_mm]

    return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict


