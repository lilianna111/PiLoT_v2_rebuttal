import numpy as np
import rasterio
import matplotlib.pyplot as plt
import time
from pathlib import Path
from osgeo import gdal
# from immatch.utils.transform import parse_pose_list, parse_intrinsic_list
# from immatch.utils.mountain_detection import is_mountain_terrain, detect_mountain_terrain, print_mountain_detection_result
import os
from PIL import Image
import cv2
import numpy as np


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
        
    # # 根据地形类型设置不同的area_minZ值
    # if is_mountain_terrain():
    #     # 山地地形：使用75%分位数，避免低洼区域的影响
    #     area_minZ = np.percentile(area[valid_mask], 75)
    #     print(f"山地地形：使用75%分位数作为最低高程 = {area_minZ:.1f}m")
    # else:
    # 平原地形：使用中位数
    area_minZ = np.median(area[valid_mask])
    print(f"平原地形：使用中位数作为最低高程 = {area_minZ:.1f}m")

    print("data size:", band.XSize, "x", band.YSize)
    print("area_minZ (valid minimum elevation):", area_minZ)
    print("Loading time cost:", time.time() - start, "s")

   
    return geotransform, area, area_minZ, dsm_data, dsm_transform,dom_data


def pixel_to_world(K, pose_w2c, uv, target_z):
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    cam_center = c2w[:3, 3]
    scale = (target_z - cam_center[2]) / ray_dir[2]
    world_pt = cam_center + ray_dir * scale
    return world_pt

def pixel_to_ray_dir(K, pose_w2c, uv):
    """
    计算像素对应的射线方向（世界坐标系）
    """
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    return ray_dir

def ray_intersect_dsm(ray_origin, ray_dir, dsm_data, dsm_transform, max_distance=10000):
    """
    计算射线与DSM的相交点
    
    Args:
        ray_origin: 射线起点（世界坐标系）
        ray_dir: 射线方向（世界坐标系）
        dsm_data: DSM数据
        dsm_transform: DSM地理变换
        max_distance: 最大搜索距离（米）
    
    Returns:
        交点的世界坐标 (x, y, z)，如果没有找到交点则返回 None
    """
    # 沿着射线采样，找到与DSM的最高交点
    num_samples = int(max_distance / 0.5)  # 每0.5米采样一次
    best_point = None
    best_z = -np.inf
    
    for i in range(num_samples):
        distance = i * 0.5
        point = ray_origin + ray_dir * distance
        
        # 检查点是否在DSM范围内
        row, col = geo_coords_to_dsm_index(point[0], point[1], dsm_transform)
        
        if 0 <= row < dsm_data.shape[0] and 0 <= col < dsm_data.shape[1]:
            dsm_z = dsm_data[row, col]
            
            if not np.isnan(dsm_z) and dsm_z > 0:
                # 如果射线的高度高于DSM高度，说明相交了
                point_z = point[2]
                if point_z <= dsm_z + 0.1:  # 允许0.1米的误差
                    if dsm_z > best_z:
                        best_z = dsm_z
                        best_point = np.array([point[0], point[1], dsm_z])
    
    return best_point

def get_height_from_center_pixel(K, pose_w2c, center_point, dsm_data, dsm_transform):
    """
    通过图像中心点和DSM相交来计算高度和相交点坐标
    
    Args:
        K: 相机内参
        pose_w2c: 相机位姿
        center_point: 图像中心点
        dsm_data: DSM数据
        dsm_transform: DSM地理变换
    
    Returns:
        相交点坐标 (x, y, z) 的numpy数组，如果没有找到交点则返回 None
    """
    # 计算相机中心和射线方向
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    cam_center = c2w[:3, 3]
    
    # 计算中心像素对应的射线方向
    ray_dir = pixel_to_ray_dir(K, pose_w2c, center_point)
    
    # 射线与DSM相交
    intersect_point = ray_intersect_dsm(cam_center, ray_dir, dsm_data, dsm_transform)
    
    if intersect_point is not None:
        height = intersect_point[2]
        print(f"通过中心点计算的DSM高度: {height:.2f}m")
        return intersect_point  # 返回完整的相交点坐标
    else:
        print("警告：无法通过中心点计算DSM高度，使用默认值")
        return None

def geo_coords_to_dsm_index(x, y, transform):
    col, row = ~transform * (x, y)
    return int(row), int(col)


def crop_dsm_dom(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point, area_minZ,
                 crop_padding=10):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        print('----------',xyz,'--------------')
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    # # 确保裁剪区域的高度和宽度不超过200像素
    # row_max = min(row_min + 100, dsm_data.shape[0])  
    # col_max = min(col_min + 100, dsm_data.shape[1])  #

    # 新增：限制裁剪区域大小
    max_crop_size = 10000  # 设置最大裁剪尺寸，留一些余量给OpenCV
    
    # 检查当前裁剪区域是否过大
    current_height = row_max - row_min
    current_width = col_max - col_min
    
    if current_height > max_crop_size or current_width > max_crop_size:
        print(f"警告：裁剪区域过大 {current_width}x{current_height}，正在缩小...")
        
        # 计算中心点
        center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
        center_row, center_col = geo_coords_to_dsm_index(center_xyz[0], center_xyz[1], dsm_transform)
                
        # 计算新的裁剪区域，保持中心不变
        half_size = max_crop_size // 2
        
        row_min = max(center_row - half_size, 0)
        row_max = min(center_row + half_size, dsm_height)
        col_min = max(center_col - half_size, 0)
        col_max = min(center_col + half_size, dsm_width)
        
        print(f"裁剪区域已缩小至 {col_max-col_min}x{row_max-row_min}")


    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min)



def crop_dsm_dom_geo(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point, area_minZ,
                 crop_padding=10):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        ## 打印坐标 change!!
        # print('----------',xyz,'--------------')
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    # 新增：限制裁剪区域大小 - 基于地理坐标范围而不是像素
    max_crop_geo_size = 200.0  # 设置最大裁剪地理尺寸（米），可根据需要调整
    
    # 计算当前裁剪区域的地理范围
    # 使用地理变换参数计算边界坐标
    min_x = col_min * dsm_transform.a + dsm_transform.c
    max_x = col_max * dsm_transform.a + dsm_transform.c
    min_y = row_max * dsm_transform.e + dsm_transform.f  # 注意：地理坐标中y轴方向与像素行方向相反
    max_y = row_min * dsm_transform.e + dsm_transform.f
    
    current_geo_width = abs(max_x - min_x)
    current_geo_height = abs(max_y - min_y)
    
    if current_geo_width > max_crop_geo_size or current_geo_height > max_crop_geo_size:
        print(f"警告：裁剪区域过大 {current_geo_width:.1f}m x {current_geo_height:.1f}m，正在缩小...")
        
        # 计算中心点
        center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
        center_row, center_col = geo_coords_to_dsm_index(center_xyz[0], center_xyz[1], dsm_transform)
        
        # 计算新的裁剪区域，保持中心不变
        half_geo_size = max_crop_geo_size / 2
        
        # 将地理范围转换为像素范围
        center_x = center_col * dsm_transform.a + dsm_transform.c
        center_y = center_row * dsm_transform.e + dsm_transform.f
        
        # 计算新的边界
        new_min_x = center_x - half_geo_size
        new_max_x = center_x + half_geo_size
        new_min_y = center_y - half_geo_size
        new_max_y = center_y + half_geo_size
        
        # 将地理坐标转换回像素索引
        col_min = max(int((new_min_x - dsm_transform.c) / dsm_transform.a), 0)
        col_max = min(int((new_max_x - dsm_transform.c) / dsm_transform.a), dsm_width)
        row_min = max(int((new_max_y - dsm_transform.f) / dsm_transform.e), 0)  # 注意y轴方向
        row_max = min(int((new_min_y - dsm_transform.f) / dsm_transform.e), dsm_height)
        
        # 重新计算地理范围用于显示
        final_min_x = col_min * dsm_transform.a + dsm_transform.c
        final_max_x = col_max * dsm_transform.a + dsm_transform.c
        final_min_y = row_max * dsm_transform.e + dsm_transform.f
        final_max_y = row_min * dsm_transform.e + dsm_transform.f
        
        print(f"裁剪区域已缩小至 {abs(final_max_x-final_min_x):.1f}m x {abs(final_max_y-final_min_y):.1f}m")
        print(f"对应像素范围：{col_max-col_min} x {row_max-row_min}")


    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min)



def crop_dsm_dom_mount(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point, area_minZ,
                 crop_size=200):
    """
    从DOM/DSM裁剪出一个固定的地理范围（默认为200x200米），
    以无人机视锥在地面形成的梯形的短边中点作为中心。
    """
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    # ---------- 计算梯形短边中点 ----------
    # 假设image_points是四个角点，按顺序排列：[左上, 右上, 右下, 左下]
    # 对于无人机视锥，通常形成梯形，短边是靠近相机的边
    if len(world_points) >= 4:
        def find_parallel_edges(world_points, parallel_threshold=0.15):
            """找到最接近平行的两条边，返回较短边的信息"""
            edges = []
            for i in range(4):
                p1 = np.array(world_points[i][:2])
                p2 = np.array(world_points[(i+1)%4][:2])
                edge_vector = p2 - p1
                edge_length = np.linalg.norm(edge_vector)
                edges.append((edge_vector, edge_length, i, (i+1)%4))
            
            # 计算所有边对之间的平行度
            parallel_pairs = []
            for i in range(4):
                for j in range(i+1, 4):
                    v1, len1, idx1, _ = edges[i]
                    v2, len2, idx2, _ = edges[j]
                    
                    # 计算平行度：|v1·v2| / (|v1|*|v2|) 越接近1越平行
                    parallel_degree = abs(np.dot(v1, v2)) / (len1 * len2)
                    
                    if parallel_degree > (1 - parallel_threshold):  # 平行度足够高
                        parallel_pairs.append((parallel_degree, i, j, len1, len2))
            
            if parallel_pairs:
                # 选择平行度最高的边对
                best_pair = max(parallel_pairs, key=lambda x: x[0])
                _, edge1_idx, edge2_idx, len1, len2 = best_pair
                
                # 返回较短边的信息
                if len1 < len2:
                    return edges[edge1_idx]
                else:
                    return edges[edge2_idx]
            
            return None  # 没找到足够平行的边
        
        parallel_edge = find_parallel_edges(world_points, parallel_threshold=0.15)
        if parallel_edge:
            # 使用平行边中的短边中点
            _, _, start_idx, end_idx = parallel_edge
            p1 = np.array(world_points[start_idx][:2])
            p2 = np.array(world_points[end_idx][:2])
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            print(f"找到平行边，短边中点: ({center_x:.2f}, {center_y:.2f})")
        else:
            # 回退到原来的方法：找最短的两条边
            edges = []
            for i in range(4):
                p1 = np.array(world_points[i][:2])
                p2 = np.array(world_points[(i+1)%4][:2])
                edge_length = np.linalg.norm(p2 - p1)
                edges.append((edge_length, i, (i+1)%4))
            
            edges.sort(key=lambda x: x[0])
            short_edge1 = edges[0]  # 最短边
            short_edge2 = edges[1]  # 第二短边
            
            # 计算两条短边的中点
            p1_1 = np.array(world_points[short_edge1[1]][:2])
            p1_2 = np.array(world_points[short_edge1[2]][:2])
            mid1 = (p1_1 + p1_2) / 2
            
            p2_1 = np.array(world_points[short_edge2[1]][:2])
            p2_2 = np.array(world_points[short_edge2[2]][:2])
            mid2 = (p2_1 + p2_2) / 2
            
            center_x = (mid1[0] + mid2[0]) / 2
            center_y = (mid1[1] + mid2[1]) / 2
            
            print(f"未找到平行边，使用最短边: {short_edge1[0]:.2f}m, {short_edge2[0]:.2f}m")
            print(f"短边中点: ({mid1[0]:.2f}, {mid1[1]:.2f}), ({mid2[0]:.2f}, {mid2[1]:.2f})")
            print(f"最终中心: ({center_x:.2f}, {center_y:.2f})")
    else:
        # 如果点数不足4个，回退到原来的方法
        print("警告：角点数量不足4个，使用默认中心点")
        center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
        center_x = center_xyz[0]
        center_y = center_xyz[1]

    # ---------- 将中心地理坐标转为像素索引 ----------
    center_row, center_col = geo_coords_to_dsm_index(center_x, center_y, dsm_transform)

    # 裁剪范围（固定200x200米）
    half_size = crop_size / 2
    min_x = center_x - half_size
    max_x = center_x + half_size
    min_y = center_y - half_size
    max_y = center_y + half_size

    # 转回像素坐标
    col_min = max(int((min_x - dsm_transform.c) / dsm_transform.a), 0)
    col_max = min(int((max_x - dsm_transform.c) / dsm_transform.a), dsm_data.shape[1])
    row_min = max(int((max_y - dsm_transform.f) / dsm_transform.e), 0)  # 注意y轴方向
    row_max = min(int((min_y - dsm_transform.f) / dsm_transform.e), dsm_data.shape[0])

    print(f"固定裁剪：{crop_size}m x {crop_size}m")
    print(f"对应像素范围：{col_max - col_min} x {row_max - row_min}")

    # 执行裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop

    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])
        zz[:row_end, :col_end] = dsm_crop[:row_end, :col_end]
        zz = np.nan_to_num(zz, nan=0.01)
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    # 返回裁剪的offset，用于坐标转换
    return dom_crop, points_3d, np.array(dsm_indices), world_points, crop_size, (row_min, col_min)


def calculate_polygon_area(vertices):
    """
    使用Shoelace公式计算多边形面积
    参数: vertices - 顶点坐标列表 [(x1, y1), (x2, y2), ...]
    返回: 面积（像素²）
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def clip_line_to_rect(p1, p2, rect):
    """
    将线段裁剪到矩形内，返回在矩形内的线段部分
    
    参数:
        p1, p2: 线段的两个端点 (x, y)
        rect: (x_min, y_min, x_max, y_max) 矩形的边界
    
    返回:
        [(x1, y1), (x2, y2)]: 裁剪后的线段端点，如果线段不在矩形内则返回空列表
    """
    x1, y1 = p1
    x2, y2 = p2
    x_min, y_min, x_max, y_max = rect
    
    # 如果线段完全在矩形外
    if (x1 < x_min and x2 < x_min) or (x1 > x_max and x2 > x_max) or \
       (y1 < y_min and y2 < y_min) or (y1 > y_max and y2 > y_max):
        return []
    
    # 如果线段完全在矩形内
    if x_min <= x1 <= x_max and x_min <= x2 <= x_max and \
       y_min <= y1 <= y_max and y_min <= y2 <= y_max:
        return [(x1, y1), (x2, y2)]
    
    # 计算线段与矩形边界的交点
    points = []
    
    # 计算方向向量
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        # 线段退化为点
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            return [(x1, y1)]
        return []
    
    # 计算与四条边的交点
    if abs(dx) > 1e-9:
        # 与左右边的交点
        for x_bound in [x_min, x_max]:
            t = (x_bound - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if y_min <= y <= y_max:
                    points.append((x_bound, y))
    
    if abs(dy) > 1e-9:
        # 与上下边的交点
        for y_bound in [y_min, y_max]:
            t = (y_bound - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if x_min <= x <= x_max:
                    points.append((x, y_bound))
    
    # 添加在矩形内的端点
    if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
        points.append((x1, y1))
    if x_min <= x2 <= x_max and y_min <= y2 <= y_max:
        points.append((x2, y2))
    
    # 去重并排序
    if len(points) == 0:
        return []
    
    # 按角度排序，找到线段的两个端点
    if len(points) == 1:
        return [points[0]]
    
    # 计算点到起点的距离
    def dist_to_p1(p):
        return ((p[0] - x1)**2 + (p[1] - y1)**2)**0.5
    
    sorted_points = sorted(set(points), key=dist_to_p1)
    
    if len(sorted_points) >= 2:
        return [sorted_points[0], sorted_points[-1]]
    return sorted_points

def clip_polygon_to_rect(polygon, rect):
    """
    将多边形裁剪到矩形内，返回在矩形内的多边形顶点
    
    参数:
        polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
        rect: (x_min, y_min, x_max, y_max) 矩形的边界
    
    返回:
        裁剪后的多边形顶点列表
    """
    if len(polygon) == 0:
        return []
    
    clipped_points = []
    n = len(polygon)
    x_min, y_min, x_max, y_max = rect
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        # 裁剪这条边
        clipped_edge = clip_line_to_rect(p1, p2, rect)
        
        # 添加裁剪后的点（避免重复添加）
        for p in clipped_edge:
            if len(clipped_points) == 0 or clipped_points[-1] != p:
                clipped_points.append(p)
    
    # 移除重复的相邻点
    if len(clipped_points) > 1 and clipped_points[0] == clipped_points[-1]:
        clipped_points = clipped_points[:-1]
    
    return clipped_points


def create_polygon_mask(image_shape_hw, polygon_vertices):
    """
    根据裁剪图中的多边形顶点创建二值 mask。

    参数:
        image_shape_hw: (height, width)
        polygon_vertices: [(x1, y1), ...]

    返回:
        mask: uint8, 有效区域为255，其余为0
    """
    height, width = image_shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon_vertices is None or len(polygon_vertices) < 3:
        mask[:, :] = 255
        return mask

    pts = np.array(polygon_vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_white_mask_to_dom_crop(dom_crop, mask):
    """
    将裁剪后的 DOM 图像中 mask 外区域填成白色。
    dom_crop shape: (3, H, W) 或 (H, W, 3)
    """
    if dom_crop.ndim == 3 and dom_crop.shape[0] == 3:
        img = np.transpose(dom_crop, (1, 2, 0)).copy()
        transposed = True
    else:
        img = dom_crop.copy()
        transposed = False

    img = np.clip(img, 0, 255).astype(np.uint8)
    img[mask == 0] = 255

    if transposed:
        return np.transpose(img, (2, 0, 1))
    return img

def calculate_polygon_rect_intersection_area(polygon, rect):
    """
    计算多边形和矩形的交集面积
    
    参数:
        polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
        rect: (x_min, y_min, x_max, y_max) 矩形的边界
    
    返回:
        交集面积（像素²）
    """
    # 裁剪多边形到矩形内
    clipped_polygon = clip_polygon_to_rect(polygon, rect)
    
    # 如果裁剪后的多边形为空或点数少于3，交集面积为0
    if len(clipped_polygon) < 3:
        return 0.0
    
    # 计算裁剪后多边形的面积
    return calculate_polygon_area(clipped_polygon)

def find_short_parallel_edge(world_points, parallel_threshold=0.15):
    """
    找到梯形两条平行边中较短的那条边
    
    参数:
        world_points: 四个角点的世界坐标列表 [(x1, y1, z1), (x2, y2, z2), ...]
        parallel_threshold: 平行度阈值（默认0.15，即约81度以上认为平行）
    
    返回:
        (short_edge_length, short_edge_direction, start_idx, end_idx)
        - short_edge_length: 短边的长度（米）
        - short_edge_direction: 短边的方向向量（归一化，2D，只取xy平面）
        - start_idx, end_idx: 短边的起点和终点索引
        如果找不到平行边，返回None
    """
    if len(world_points) < 4:
        return None
    
    # 计算四条边的信息
    edges = []
    for i in range(4):
        p1 = np.array(world_points[i][:2])  # 只取xy坐标
        p2 = np.array(world_points[(i+1)%4][:2])
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        if edge_length > 0:
            edge_direction = edge_vector / edge_length  # 归一化
        else:
            edge_direction = edge_vector
        edges.append((edge_vector, edge_direction, edge_length, i, (i+1)%4))
    
    # 找到最接近平行的两条边
    parallel_pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            _, dir1, len1, idx1, _ = edges[i]
            _, dir2, len2, idx2, _ = edges[j]
            
            # 计算平行度：|v1·v2| / (|v1|*|v2|) 越接近1越平行
            # 由于已经归一化，直接计算点积的绝对值
            parallel_degree = abs(np.dot(dir1, dir2))
            
            if parallel_degree > (1 - parallel_threshold):  # 平行度足够高
                parallel_pairs.append((parallel_degree, i, j, len1, len2))
    
    if not parallel_pairs:
        return None
    
    # 选择平行度最高的边对
    best_pair = max(parallel_pairs, key=lambda x: x[0])
    _, edge1_idx, edge2_idx, len1, len2 = best_pair
    
    # 返回较短边的信息
    if len1 < len2:
        _, direction, length, start_idx, end_idx = edges[edge1_idx]
        return (length, direction, start_idx, end_idx)
    else:
        _, direction, length, start_idx, end_idx = edges[edge2_idx]
        return (length, direction, start_idx, end_idx)

def build_rotated_rectangle_from_center_and_edge(center_world, edge_length_along_short_direction,
                                                 short_edge_direction, image_aspect_ratio):
    """
    基于中心点投影和短边方向构建斜向长方形。

    设计约定：
    1. 长方形中心 = query 图像中心点投影到地面的世界坐标
    2. 沿 short_edge_direction 的那条边长度 = 视域梯形短边长度
    3. 另一条边长度按 query 图像宽高比推得，保证 reference 与 query 形状一致

    参数:
        center_world: 中心点的世界坐标 (x, y, z) 或 (x, y)
        edge_length_along_short_direction: 沿 short_edge_direction 的边长（米）
        short_edge_direction: 短边方向向量（归一化，2D）
        image_aspect_ratio: query 图像宽高比 width / height

    返回:
        rect_vertices: 长方形四个顶点的世界坐标列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    center_xy = np.array(center_world[:2], dtype=np.float64)

    # 沿短边方向的边，使用视域梯形短边长度
    width_along_u = float(edge_length_along_short_direction)

    # 另一条边按 query 图像宽高比确定，保证 crop 形状与 query 一致
    if image_aspect_ratio <= 0:
        image_aspect_ratio = 1.0
    height_along_v = width_along_u / image_aspect_ratio

    u = np.array(short_edge_direction, dtype=np.float64)
    if np.linalg.norm(u) == 0:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = u / np.linalg.norm(u)

    # 垂直方向
    v = np.array([-u[1], u[0]], dtype=np.float64)

    half_u = width_along_u / 2.0
    half_v = height_along_v / 2.0

    p1 = center_xy + half_u * u + half_v * v
    p2 = center_xy + half_u * u - half_v * v
    p3 = center_xy - half_u * u - half_v * v
    p4 = center_xy - half_u * u + half_v * v

    return [p1, p2, p3, p4]

def crop_dsm_dom_400(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point, area_minZ,
                 crop_padding=0):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        ## 打印坐标 change!!
        # print('----------',xyz,'--------------')
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape
    
    max_crop_geo_size = 200.0  # 设置最大裁剪地理尺寸（米），可根据需要调整

    # 计算中心点
    center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
    center_row, center_col = geo_coords_to_dsm_index(center_xyz[0], center_xyz[1], dsm_transform)
    
    # 计算新的裁剪区域，保持中心不变
    half_geo_size = max_crop_geo_size / 2
    
    # 将地理范围转换为像素范围
    center_x = center_col * dsm_transform.a + dsm_transform.c
    center_y = center_row * dsm_transform.e + dsm_transform.f
    
    # 计算新的边界
    new_min_x = center_x - half_geo_size
    new_max_x = center_x + half_geo_size
    new_min_y = center_y - half_geo_size
    new_max_y = center_y + half_geo_size
    
    # 将地理坐标转换回像素索引
    col_min = max(int((new_min_x - dsm_transform.c) / dsm_transform.a), 0)
    col_max = min(int((new_max_x - dsm_transform.c) / dsm_transform.a), dsm_width)
    row_min = max(int((new_max_y - dsm_transform.f) / dsm_transform.e), 0)  # 注意y轴方向
    row_max = min(int((new_min_y - dsm_transform.f) / dsm_transform.e), dsm_height)
    
    # 重新计算地理范围用于显示
    final_min_x = col_min * dsm_transform.a + dsm_transform.c
    final_max_x = col_max * dsm_transform.a + dsm_transform.c
    final_min_y = row_max * dsm_transform.e + dsm_transform.f
    final_max_y = row_min * dsm_transform.e + dsm_transform.f
    
    print(f"裁剪区域已缩小至 {abs(final_max_x-final_min_x):.1f}m x {abs(final_max_y-final_min_y):.1f}m")
    print(f"对应像素范围：{col_max-col_min} x {row_max-row_min}")
    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min)

def generate_ref_map_multi_ratios(query_intrinsics, query_poses, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path, debug, crop_vis_save_path=None, test_ratio=None):
    """
    为多个crop比例生成独立的reference文件
    每个比例会生成独立的DOM图片和点云文件
    
    参数:
        test_ratio: 如果传入列表，只使用第一个值；如果传入单个值，直接使用
    """
    data = {}
    
    # 处理 test_ratio 的默认值
    if test_ratio is None:
        test_ratio = 0.4
    
    for name in query_intrinsics.keys():
        K_w2c = query_intrinsics[name]
        pose_w2c = query_poses[name]
        width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
        image_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
        center_point = (width // 2, height // 2)
        
        print('--------------',name,'--------------')
        
        # 使用单个比例值（不再循环）
        print(f"  生成比例: {test_ratio*100:.0f}%")
        
        # if is_mountain_terrain():
        #     # 山区：使用crop_dsm_dom_mount_angle，使用射线计算的高度
        #     dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, crop_offset, _ = crop_dsm_dom_mount_angle(
        #         dom_data, dsm_data, dsm_transform,
        #         K_w2c, pose_w2c, image_points, center_point,
        #         crop_padding=0, target_ratio=test_ratio
        #     )
        # else:
        # 平原：使用crop_dsm_dom_geo，使用射线计算的高度
        dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, crop_offset, _ = crop_dsm_dom_angle(
            dom_data, dsm_data, dsm_transform,
            K_w2c, pose_w2c, image_points, center_point, area_minZ, crop_padding=0, target_ratio=test_ratio
        )

        ratio_suffix = f'_ratio_{int(test_ratio*100)}'
        
        ratio_img_path = os.path.join(ref_rgb_path, f'{name}{ratio_suffix}_dom.jpg')
        ratio_npy_path = os.path.join(ref_depth_path, f'{name}{ratio_suffix}_dom.npy')
        
        # 保存图片
        ratio_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)
        if ratio_img_uint8.shape[0] == 3:
            ratio_img_uint8 = np.transpose(ratio_img_uint8, (1, 2, 0))
        Image.fromarray(ratio_img_uint8).save(ratio_img_path)
        
        # 保存npy
        np.save(ratio_npy_path, point_cloud_crop)
        
        print(f"    💾 已保存: {ratio_img_path}")

        # 计算实际比例（与可视化标签保持一致）：梯形在矩形范围内的交集面积 / 图像总面积
        ref_h = dom_crop.shape[1]
        ref_w = dom_crop.shape[2] if len(dom_crop.shape) == 3 else dom_crop.shape[1]
        ref_total_area = ref_w * ref_h
        
        # 获取裁剪区域的偏移量
        row_min, col_min = crop_offset
        
        # 构建矩形边界（在原始DSM坐标系中）
        actual_rect = (col_min, row_min, col_min + ref_w, row_min + ref_h)
        
        # 构建梯形的顶点列表（用于交集计算）
        # dsm_indices 格式：[(col, row), ...] 或 numpy array
        if isinstance(dsm_indices, np.ndarray):
            # 如果是numpy数组，转换为列表
            trapezoid_polygon = [(col, row) for col, row in dsm_indices]
        else:
            # 如果是列表，直接使用
            trapezoid_polygon = [(col, row) for col, row in dsm_indices]
        
        # 计算交集面积（梯形在裁剪矩形范围内的部分）
        intersection_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, actual_rect)
        
        # 计算实际比例：交集面积 / 矩形面积
        actual_ratio_percent = (intersection_area / ref_total_area) * 100 if ref_total_area > 0 else 0.0
        
        # Debug模式下，生成可视化图
        if debug and crop_vis_save_path:
   
            # 确保debug目录存在
            os.makedirs(crop_vis_save_path, exist_ok=True)
            
            ref_vis = np.transpose(dom_crop, (1, 2, 0)).copy()
            ref_height, ref_width = ref_vis.shape[0], ref_vis.shape[1]
            
            # 转换坐标系：从原始DOM坐标转为裁剪图像内的相对坐标
            row_min, col_min = crop_offset
            ref_indices = dsm_indices.copy()
            ref_indices[:, 0] = ref_indices[:, 0] - col_min  # col (x)
            ref_indices[:, 1] = ref_indices[:, 1] - row_min  # row (y)
            
            # 根据图片大小自适应参数 - 使用更小的字号
            # 使用对数缩放，适应不同尺寸的图像
            min_dimension = min(ref_width, ref_height)
            # 新的字体缩放策略：更小的字号
            if min_dimension < 500:
                adaptive_font_scale = 0.5
            elif min_dimension < 1000:
                adaptive_font_scale = 0.6
            elif min_dimension < 2000:
                adaptive_font_scale = 0.7
            else:
                adaptive_font_scale = 0.7
            
            # 使用更细的线条和更小的圆形
            adaptive_thickness = 1
            line_thickness = 2
            circle_radius = 5
            
            # 绘制梯形的每条边（仅绘制在图片内的部分）
            # 这样可以避免当角点在图片外时产生歧义的新形状
            polygon_vertices = [(float(x), float(y)) for x, y in ref_indices]
            rect = (0.0, 0.0, float(ref_width - 1), float(ref_height - 1))
            n = len(polygon_vertices)
            
            for i in range(n):
                p1 = polygon_vertices[i]
                p2 = polygon_vertices[(i + 1) % n]
                
                # 裁剪这条边到图片范围内
                clipped_edge = clip_line_to_rect(p1, p2, rect)
                
                # 如果这条边有部分在图片内，就绘制它
                if len(clipped_edge) >= 2:
                    edge_points = np.array(clipped_edge, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(ref_vis, [edge_points],
                                isClosed=False, color=(255, 0, 0), thickness=line_thickness)
            
            # 绘制梯形角点（黄色圆点 + 数字标签）- 只绘制在图像内的原始角点
            for idx, (x, y) in enumerate(ref_indices):
                x, y = float(x), float(y)
                if 0 <= x < ref_width and 0 <= y < ref_height:
                    cv2.circle(ref_vis, (int(x), int(y)), radius=circle_radius, color=(0, 255, 255), thickness=-1)
                    cv2.putText(ref_vis, str(idx), (int(x) + 5, int(y) - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                    color=(0, 0, 0), thickness=adaptive_thickness, lineType=cv2.LINE_AA)
                else:
                    # 对于在图片外的角点，在边界上标记其位置方向
                    # 找到最近的边界点
                    near_x = np.clip(x, 0, ref_width - 1)
                    near_y = np.clip(y, 0, ref_height - 1)
                    # 绘制一个小十字标记，表示角点在图片外
                    cv2.drawMarker(ref_vis, (int(near_x), int(near_y)), 
                                 color=(255, 255, 0), markerType=cv2.MARKER_CROSS, 
                                 markerSize=10, thickness=1)
                    cv2.putText(ref_vis, str(idx) + '*', (int(near_x) + 5, int(near_y) - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                    color=(255, 255, 0), thickness=adaptive_thickness, lineType=cv2.LINE_AA)

            
            # ── 绘制水平中线（红色）并标注地面长度 ──
            try:
                mid_y_img = height / 2.0
                left_mid_uv  = (0, mid_y_img)
                right_mid_uv = (width - 1, mid_y_img)

                world_left_mid  = pixel_to_world(K_w2c, pose_w2c, left_mid_uv,  target_z=area_minZ)
                world_right_mid = pixel_to_world(K_w2c, pose_w2c, right_mid_uv, target_z=area_minZ)

                midline_length_m = float(np.linalg.norm(world_right_mid - world_left_mid))

                row_ml, col_ml = geo_coords_to_dsm_index(world_left_mid[0],  world_left_mid[1],  dsm_transform)
                row_mr, col_mr = geo_coords_to_dsm_index(world_right_mid[0], world_right_mid[1], dsm_transform)

                ml_x1 = col_ml - col_min
                ml_y1 = row_ml - row_min
                ml_x2 = col_mr - col_min
                ml_y2 = row_mr - row_min

                pt_ml1 = (int(ml_x1), int(ml_y1))
                pt_ml2 = (int(ml_x2), int(ml_y2))

                cv2.line(ref_vis, pt_ml1, pt_ml2, color=(255, 0, 0), thickness=line_thickness + 1, lineType=cv2.LINE_AA)
                cv2.circle(ref_vis, pt_ml1, radius=circle_radius, color=(255, 0, 0), thickness=-1)
                cv2.circle(ref_vis, pt_ml2, radius=circle_radius, color=(255, 0, 0), thickness=-1)

                mid_label = f"midline: {midline_length_m:.1f}m"
                label_x = (pt_ml1[0] + pt_ml2[0]) // 2
                label_y = (pt_ml1[1] + pt_ml2[1]) // 2 - 15
                ts, bl = cv2.getTextSize(mid_label, cv2.FONT_HERSHEY_SIMPLEX,
                                         adaptive_font_scale, adaptive_thickness + 1)
                tw_ml, th_ml = ts
                overlay_ml = ref_vis.copy()
                cv2.rectangle(overlay_ml,
                              (label_x - tw_ml // 2 - 4, label_y - th_ml - 4),
                              (label_x + tw_ml // 2 + 4, label_y + bl + 4),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay_ml, 0.6, ref_vis, 0.4, 0, ref_vis)
                cv2.putText(ref_vis, mid_label,
                            (label_x - tw_ml // 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale,
                            (255, 255, 255), adaptive_thickness + 1, cv2.LINE_AA)
            except Exception as _ml_err:
                print(f"    ⚠️ 绘制中线失败: {_ml_err}")

            # 准备文字标签（显示目标和实际比例）
            ratio_text = f"T:{test_ratio*100:.0f}% A:{actual_ratio_percent:.1f}%"
            
            # 添加半透明背景（便于文字清晰显示）
            text_size, baseline = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, adaptive_thickness+1)
            text_width, text_height = text_size
            overlay = ref_vis.copy()
            cv2.rectangle(overlay, (5, 5), (text_width + 15, text_height + baseline + 20), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, ref_vis, 0.4, 0, ref_vis)
            
            # 绘制文字标签
            cv2.putText(ref_vis, ratio_text, (10, 25),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                      color=(255, 255, 255), thickness=adaptive_thickness+1, lineType=cv2.LINE_AA)
            
            # 保存图像
            vis_save_path = os.path.join(crop_vis_save_path, f'{name}{ratio_suffix}.jpg')
            # 确保目录存在
            os.makedirs(crop_vis_save_path, exist_ok=True)
            cv2.imwrite(vis_save_path, cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))
            print(f"    💾 保存可视化: {vis_save_path} ({ref_width}x{ref_height}, 实际比例:{actual_ratio_percent:.2f}%)")
            # （移除分散日志写入；统一在保存ref后记录到全局 output/crop_ratio_log.txt）
        
        # 注册数据
        data[f'{name}{ratio_suffix}'] = {
            'imgr_name': f'{name}{ratio_suffix}_dom',
            'exr_name': f'{name}{ratio_suffix}_dom',
            'min_shape': min_shape
        }
    
    return data

def generate_ref_map(name, query_intrinsics, query_poses, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path, debug, crop_vis_save_path=None, test_ratio=None, use_bbox=True):
    # ref_DSM_path, pose_data, ref_npy_path, geotransform, query_intrinsics_dict, query_prior_poses_dict, name, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path
    """
    为多个crop比例生成独立的reference文件
    每个比例会生成独立的DOM图片和点云文件
    
    参数:
        test_ratio: 如果传入列表，只使用第一个值；如果传入单个值，直接使用（仅在use_bbox=False时使用）
        use_bbox: 如果为True，使用中心点投影和短边构建斜向长方形裁剪，不使用比例扩展（默认True）
    """
    data = {}
    
    # for name in query_intrinsics.keys():
    K_w2c = query_intrinsics
    pose_w2c = query_poses
    output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.png')
    output_npy_path = os.path.join(ref_depth_path, f'{name}_dsm.npy')
    width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
    image_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
    center_point = (width // 2, height // 2)
    
    print('--------------',name,'--------------')
    
    if use_bbox:
        # 使用中心点投影和短边构建斜向长方形裁剪（不进行比例扩展）
        print(f"  使用中心点+短边构建斜向长方形裁剪")
        target_ratio = None
    
    # if is_mountain_terrain():
    #     # 山区：使用crop_dsm_dom_mount_angle，使用射线计算的高度
    #     dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, crop_offset, rotated_rect_vertices_relative = crop_dsm_dom_mount_angle(
    #         dom_data, dsm_data, dsm_transform,
    #         K_w2c, pose_w2c, image_points, center_point,
    #         crop_padding=0, target_ratio=target_ratio
    #     )
    # else:
    # 平原：使用crop_dsm_dom_geo，使用射线计算的高度
    dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, crop_offset, rotated_rect_vertices_relative = crop_dsm_dom_angle(
        dom_data, dsm_data, dsm_transform,
        K_w2c, pose_w2c, image_points, center_point, area_minZ, crop_padding=0, target_ratio=target_ratio
    )

    # 根据裁剪方式设置后缀
    if use_bbox:
        ratio_suffix = '_bbox'
    else:
        ratio_suffix = f'_ratio_{int(test_ratio*100)}'
    
    ratio_img_path = os.path.join(ref_rgb_path, f'{name}{ratio_suffix}_dom.jpg')
    ratio_npy_path = os.path.join(ref_depth_path, f'{name}{ratio_suffix}_dom.npy')
    
    # 保存图片：斜向长方形模式下，将长方形外区域填白，并保存 mask 供旋转后裁边
    ratio_mask_path = os.path.join(ref_rgb_path, f'{name}{ratio_suffix}_dom_mask.png')
    if use_bbox and rotated_rect_vertices_relative is not None:
        ref_h = dom_crop.shape[1]
        ref_w = dom_crop.shape[2] if len(dom_crop.shape) == 3 else dom_crop.shape[1]
        crop_mask = create_polygon_mask((ref_h, ref_w), rotated_rect_vertices_relative)
        dom_crop_masked = apply_white_mask_to_dom_crop(dom_crop, crop_mask)
        ratio_img_uint8 = np.clip(dom_crop_masked, 0, 255).astype(np.uint8)
        if ratio_img_uint8.shape[0] == 3:
            ratio_img_uint8 = np.transpose(ratio_img_uint8, (1, 2, 0))
        Image.fromarray(ratio_img_uint8).save(ratio_img_path)
        cv2.imwrite(ratio_mask_path, crop_mask)
    else:
        ratio_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)
        if ratio_img_uint8.shape[0] == 3:
            ratio_img_uint8 = np.transpose(ratio_img_uint8, (1, 2, 0))
        Image.fromarray(ratio_img_uint8).save(ratio_img_path)
    
    # 保存npy
    np.save(ratio_npy_path, point_cloud_crop)
    
    print(f"    💾 已保存: {ratio_img_path}")

    # 计算实际比例（与可视化标签保持一致）：梯形在矩形范围内的交集面积 / 图像总面积
    ref_h = dom_crop.shape[1]
    ref_w = dom_crop.shape[2] if len(dom_crop.shape) == 3 else dom_crop.shape[1]
    ref_total_area = ref_w * ref_h
    
    # 获取裁剪区域的偏移量
    row_min, col_min = crop_offset
    
    # 构建矩形边界（在原始DSM坐标系中）
    actual_rect = (col_min, row_min, col_min + ref_w, row_min + ref_h)
    
    # 构建梯形的顶点列表（用于交集计算）
    # dsm_indices 格式：[(col, row), ...] 或 numpy array
    if isinstance(dsm_indices, np.ndarray):
        # 如果是numpy数组，转换为列表
        trapezoid_polygon = [(col, row) for col, row in dsm_indices]
    else:
        # 如果是列表，直接使用
        trapezoid_polygon = [(col, row) for col, row in dsm_indices]
    
    # 计算交集面积（梯形在裁剪矩形范围内的部分）
    intersection_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, actual_rect)
    
    # 计算实际比例：交集面积 / 矩形面积
    actual_ratio_percent = (intersection_area / ref_total_area) * 100 if ref_total_area > 0 else 0.0
    
    # Debug模式下，生成可视化图
    if debug and crop_vis_save_path:
        
        # 确保debug目录存在
        os.makedirs(crop_vis_save_path, exist_ok=True)
        
        ref_vis = np.transpose(dom_crop, (1, 2, 0)).copy()
        ref_height, ref_width = ref_vis.shape[0], ref_vis.shape[1]
        
        # 转换坐标系：从原始DOM坐标转为裁剪图像内的相对坐标
        row_min, col_min = crop_offset
        ref_indices = dsm_indices.copy()
        ref_indices[:, 0] = ref_indices[:, 0] - col_min  # col (x)
        ref_indices[:, 1] = ref_indices[:, 1] - row_min  # row (y)
        
        # 根据图片大小自适应参数 - 使用更小的字号
        # 使用对数缩放，适应不同尺寸的图像
        min_dimension = min(ref_width, ref_height)
        # 新的字体缩放策略：更小的字号
        if min_dimension < 500:
            adaptive_font_scale = 0.5
        elif min_dimension < 1000:
            adaptive_font_scale = 0.6
        elif min_dimension < 2000:
            adaptive_font_scale = 0.7
        else:
            adaptive_font_scale = 0.7
        
        # 使用更细的线条和更小的圆形
        adaptive_thickness = 1
        line_thickness = 2
        circle_radius = 5
        
        # 绘制梯形的每条边（仅绘制在图片内的部分）
        # 这样可以避免当角点在图片外时产生歧义的新形状
        polygon_vertices = [(float(x), float(y)) for x, y in ref_indices]
        rect = (0.0, 0.0, float(ref_width - 1), float(ref_height - 1))
        n = len(polygon_vertices)
        
        for i in range(n):
            p1 = polygon_vertices[i]
            p2 = polygon_vertices[(i + 1) % n]
            
            # 裁剪这条边到图片范围内
            clipped_edge = clip_line_to_rect(p1, p2, rect)
            
            # 如果这条边有部分在图片内，就绘制它
            if len(clipped_edge) >= 2:
                edge_points = np.array(clipped_edge, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(ref_vis, [edge_points],
                            isClosed=False, color=(255, 0, 0), thickness=line_thickness)
        
        # 绘制梯形角点（黄色圆点 + 数字标签）- 只绘制在图像内的原始角点
        for idx, (x, y) in enumerate(ref_indices):
            x, y = float(x), float(y)
            if 0 <= x < ref_width and 0 <= y < ref_height:
                cv2.circle(ref_vis, (int(x), int(y)), radius=circle_radius, color=(0, 255, 255), thickness=-1)
                cv2.putText(ref_vis, str(idx), (int(x) + 5, int(y) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                color=(0, 0, 0), thickness=adaptive_thickness, lineType=cv2.LINE_AA)
            else:
                # 对于在图片外的角点，在边界上标记其位置方向
                # 找到最近的边界点
                near_x = np.clip(x, 0, ref_width - 1)
                near_y = np.clip(y, 0, ref_height - 1)
                # 绘制一个小十字标记，表示角点在图片外
                cv2.drawMarker(ref_vis, (int(near_x), int(near_y)), 
                                color=(255, 255, 0), markerType=cv2.MARKER_CROSS, 
                                markerSize=10, thickness=1)
                cv2.putText(ref_vis, str(idx) + '*', (int(near_x) + 5, int(near_y) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                color=(255, 255, 0), thickness=adaptive_thickness, lineType=cv2.LINE_AA)

        # ── 绘制斜向长方形（中心点投影 + 短边） ──
        print(f"  Debug: rotated_rect_vertices_relative = {rotated_rect_vertices_relative}")
        if rotated_rect_vertices_relative is not None:
            rect_points = np.array(rotated_rect_vertices_relative, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(ref_vis, [rect_points], isClosed=True, color=(0, 255, 0), thickness=line_thickness + 2, lineType=cv2.LINE_AA)

            for rect_idx, (x, y) in enumerate(rotated_rect_vertices_relative):
                x, y = float(x), float(y)
                if 0 <= x < ref_width and 0 <= y < ref_height:
                    cv2.circle(ref_vis, (int(x), int(y)), radius=circle_radius + 2, color=(0, 255, 0), thickness=-1)
                    cv2.putText(ref_vis, f'R{rect_idx}', (int(x) + 5, int(y) - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                                color=(0, 0, 0), thickness=adaptive_thickness + 1, lineType=cv2.LINE_AA)
        else:
            print(f"  Debug: rotated_rect_vertices_relative 为 None，不绘制斜向长方形")
        
        # ── 绘制水平中线（红色）并标注地面长度 ──
        try:
            mid_y_img = height / 2.0
            left_mid_uv  = (0, mid_y_img)
            right_mid_uv = (width - 1, mid_y_img)

            world_left_mid  = pixel_to_world(K_w2c, pose_w2c, left_mid_uv,  target_z=area_minZ)
            world_right_mid = pixel_to_world(K_w2c, pose_w2c, right_mid_uv, target_z=area_minZ)

            # 中线地面长度（米）
            midline_length_m = float(np.linalg.norm(world_right_mid - world_left_mid))

            # 投影到 DSM 坐标再转裁剪图内坐标
            row_ml, col_ml = geo_coords_to_dsm_index(world_left_mid[0],  world_left_mid[1],  dsm_transform)
            row_mr, col_mr = geo_coords_to_dsm_index(world_right_mid[0], world_right_mid[1], dsm_transform)

            ml_x1 = col_ml - col_min
            ml_y1 = row_ml - row_min
            ml_x2 = col_mr - col_min
            ml_y2 = row_mr - row_min

            pt_ml1 = (int(ml_x1), int(ml_y1))
            pt_ml2 = (int(ml_x2), int(ml_y2))

            # 红色中线
            cv2.line(ref_vis, pt_ml1, pt_ml2, color=(255, 0, 0), thickness=line_thickness + 1, lineType=cv2.LINE_AA)
            cv2.circle(ref_vis, pt_ml1, radius=circle_radius, color=(255, 0, 0), thickness=-1)
            cv2.circle(ref_vis, pt_ml2, radius=circle_radius, color=(255, 0, 0), thickness=-1)

            # 标注长度文字
            mid_label = f"midline: {midline_length_m:.1f}m"
            label_x = (pt_ml1[0] + pt_ml2[0]) // 2
            label_y = (pt_ml1[1] + pt_ml2[1]) // 2 - 15
            ts, bl = cv2.getTextSize(mid_label, cv2.FONT_HERSHEY_SIMPLEX,
                                        adaptive_font_scale, adaptive_thickness + 1)
            tw_ml, th_ml = ts
            overlay_ml = ref_vis.copy()
            cv2.rectangle(overlay_ml,
                            (label_x - tw_ml // 2 - 4, label_y - th_ml - 4),
                            (label_x + tw_ml // 2 + 4, label_y + bl + 4),
                            (0, 0, 0), -1)
            cv2.addWeighted(overlay_ml, 0.6, ref_vis, 0.4, 0, ref_vis)
            cv2.putText(ref_vis, mid_label,
                        (label_x - tw_ml // 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale,
                        (255, 255, 255), adaptive_thickness + 1, cv2.LINE_AA)
        except Exception as _ml_err:
            print(f"    ⚠️ 绘制中线失败: {_ml_err}")

        # 准备文字标签（显示目标和实际比例）
        if use_bbox or test_ratio is None:
            ratio_text = f"RotRect A:{actual_ratio_percent:.1f}%"
        else:
            ratio_text = f"T:{test_ratio*100:.0f}% A:{actual_ratio_percent:.1f}%"
        
        # 添加半透明背景（便于文字清晰显示）
        text_size, baseline = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, adaptive_thickness+1)
        text_width, text_height = text_size
        overlay = ref_vis.copy()
        cv2.rectangle(overlay, (5, 5), (text_width + 15, text_height + baseline + 20), 
                        (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, ref_vis, 0.4, 0, ref_vis)
        
        # 绘制文字标签
        cv2.putText(ref_vis, ratio_text, (10, 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                    color=(255, 255, 255), thickness=adaptive_thickness+1, lineType=cv2.LINE_AA)
        
        # 保存图像
        vis_save_path = os.path.join(crop_vis_save_path, f'{name}{ratio_suffix}.jpg')
        # 确保目录存在
        os.makedirs(crop_vis_save_path, exist_ok=True)
        cv2.imwrite(vis_save_path, cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))
        print(f"    💾 保存可视化: {vis_save_path} ({ref_width}x{ref_height}, 实际比例:{actual_ratio_percent:.2f}%)")
        # （移除分散日志写入；统一在保存ref后记录到全局 output/crop_ratio_log.txt）
        
    data = {
            'imgr_name':name+'_dom',
            'exr_name':name+'_dsm',
            'min_shape': min_shape
        }
    return data

def crop_dsm_dom_angle(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point, area_minZ,
                 crop_padding=10, target_ratio=0.55):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        ## 打印坐标 change!!
        # print('----------',xyz,'--------------')
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    # ori模式：使用图片中心点投影和短边构建斜向长方形
    if target_ratio is None:
        # 步骤1: 找到梯形的短边（两条平行边中较短的那条）
        short_edge_info = find_short_parallel_edge(world_points, parallel_threshold=0.15)
        
        if short_edge_info is None:
            # 如果找不到平行边，回退到外接矩形方法
            print("警告：无法找到平行边，回退到外接矩形方法")
            row_min_raw = max(min(rows), 0)
            row_max_raw = min(max(rows), dsm_height)
            col_min_raw = max(min(cols), 0)
            col_max_raw = min(max(cols), dsm_width)
            row_min = row_min_raw
            row_max = row_max_raw
            col_min = col_min_raw
            col_max = col_max_raw
            bbox_width = col_max_raw - col_min_raw
            bbox_height = row_max_raw - row_min_raw
            bbox_area = bbox_width * bbox_height
            if bbox_width <= 0 or bbox_height <= 0:
                bbox_width = max(bbox_width, 1)
                bbox_height = max(bbox_height, 1)
                bbox_area = bbox_width * bbox_height
            trapezoid_polygon = [(col, row) for row, col in dsm_indices]
            dsm_map_rect = (0, 0, dsm_width, dsm_height)
            trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
            rotated_rect_vertices_relative = None
        else:
            short_edge_length, short_edge_direction, start_idx, end_idx = short_edge_info
            print(f"找到短边：长度={short_edge_length:.2f}m, 方向=({short_edge_direction[0]:.3f}, {short_edge_direction[1]:.3f})")

            # 步骤2: 使用 query 中心点投影作为长方形中心
            center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
            print(f"中心点投影（世界坐标）: ({center_xyz[0]:.2f}, {center_xyz[1]:.2f})")

            # 步骤3: 保持 query 图像宽高比，沿短边方向构建斜向长方形
            img_width = max([pt[0] for pt in image_points]) - min([pt[0] for pt in image_points]) + 1
            img_height = max([pt[1] for pt in image_points]) - min([pt[1] for pt in image_points]) + 1
            image_aspect_ratio = img_width / img_height if img_height > 0 else 1.0
            rect_vertices = build_rotated_rectangle_from_center_and_edge(
                center_xyz, short_edge_length, short_edge_direction, image_aspect_ratio
            )
            print(f"长方形顶点（世界坐标）:")
            for i, v in enumerate(rect_vertices):
                print(f"  顶点{i+1}: ({v[0]:.2f}, {v[1]:.2f})")

            # 步骤4: 将长方形顶点转换为DSM像素坐标
            rotated_rect_pixel_coords = []
            for vertex in rect_vertices:
                row, col = geo_coords_to_dsm_index(vertex[0], vertex[1], dsm_transform)
                rotated_rect_pixel_coords.append((row, col))

            # 步骤5: 计算长方形在DSM中的轴对齐边界框（用于裁剪）
            rect_rows, rect_cols = zip(*rotated_rect_pixel_coords)
            row_min_raw = max(min(rect_rows), 0)
            row_max_raw = min(max(rect_rows), dsm_height)
            col_min_raw = max(min(rect_cols), 0)
            col_max_raw = min(max(rect_cols), dsm_width)
            
            # 使用边界框作为裁剪区域（参考现有逻辑的边界处理）
            row_min = row_min_raw
            row_max = row_max_raw
            col_min = col_min_raw
            col_max = col_max_raw
            
            # 计算边界框的尺寸和面积（用于后续计算）
            bbox_width = col_max_raw - col_min_raw
            bbox_height = row_max_raw - row_min_raw
            bbox_area = bbox_width * bbox_height
            
            # 确保宽高有效
            if bbox_width <= 0 or bbox_height <= 0:
                bbox_width = max(bbox_width, 1)
                bbox_height = max(bbox_height, 1)
                bbox_area = bbox_width * bbox_height
            
            # 计算梯形与DSM地图的交集面积（用于后续比例计算）
            trapezoid_polygon = [(col, row) for row, col in dsm_indices]
            dsm_map_rect = (0, 0, dsm_width, dsm_height)
            trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
            
            use_rotated_rect = True
    else:
        # target_ratio 不为 None 的情况
        # 如果计算失败，回退到轴对齐边界框
        row_min_raw = max(min(rows), 0)
        row_max_raw = min(max(rows), dsm_height)
        col_min_raw = max(min(cols), 0)
        col_max_raw = min(max(cols), dsm_width)
        
        bbox_width = col_max_raw - col_min_raw
        bbox_height = row_max_raw - row_min_raw
        bbox_area = bbox_width * bbox_height

        # 确保宽高有效
        if bbox_width <= 0 or bbox_height <= 0:
            bbox_width = max(bbox_width, 1)
            bbox_height = max(bbox_height, 1)
            bbox_area = bbox_width * bbox_height
        
        # 计算梯形与DSM地图的交集面积（用于后续比例计算）
        # 构建梯形的顶点列表
        trapezoid_polygon = [(col, row) for row, col in dsm_indices]
        # 构建DSM地图的边界矩形（整个地图范围）
        dsm_map_rect = (0, 0, dsm_width, dsm_height)
        # 计算梯形在地图范围内的交集面积（只计算在地图内的部分）
        trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
        
        rotated_rect_vertices_relative = None
        
        # 步骤2: 根据target_ratio计算scale_factor，使得梯形面积占最终crop区域的比例为target_ratio
        # 目标：(trapezoid_area) / (最终crop区域面积) = target_ratio
        # 最终crop区域面积 = bbox_area * scale_factor^2
        # 所以：trapezoid_area / (bbox_area * scale_factor^2) = target_ratio
        # 解得：scale_factor = sqrt(trapezoid_area / (bbox_area * target_ratio))
        scale_factor = np.sqrt(trapezoid_area / (bbox_area * target_ratio)) if bbox_area > 0 and trapezoid_area > 0 else 1.0
        
        # 如果 scale_factor < 1，说明需要缩小，但这里只扩展不缩减，直接使用原始外接矩形
        if scale_factor < 1.0:
            # 直接使用原始外接矩形，不进行任何扩展或缩减操作
            col_min = col_min_raw
            col_max = col_max_raw
            row_min = row_min_raw
            row_max = row_max_raw
        else:
            # 步骤3: 按照最小外接矩形的宽高比，计算最终的crop尺寸
            final_width = int(round(bbox_width * scale_factor))
            final_height = int(round(bbox_height * scale_factor))
            
            # 步骤4: 在梯形外围按照优先级顺序扩展像素：左、右、上、下
            # 计算需要扩展的像素数
            width_expand = final_width - bbox_width
            height_expand = final_height - bbox_height
            
            # 按优先级顺序分配扩展像素：左、右、上、下
            # 宽度扩展：优先左边，然后右边
            left_expand = int(width_expand/2)  # 优先全部从左边扩展
            right_expand = int(width_expand/2)
            # 如果左边扩展后无法满足（会越界），则从右边补足
            # 这里暂时全部从左边扩展，后续越界检查会调整
            
            # 高度扩展：优先上边，然后下边
            top_expand = int(height_expand/2)  # 优先全部从上边扩展
            bottom_expand = int(height_expand/2)
            # 如果上边扩展后无法满足（会越界），则从下边补足
            # 这里暂时全部从上边扩展，后续越界检查会调整
            
            # 从梯形的最小外接矩形开始扩展
            col_min = int(col_min_raw - left_expand)
            col_max = int(col_max_raw + right_expand)
            row_min = int(row_min_raw - top_expand)
            row_max = int(row_max_raw + bottom_expand)

            # 若越界，按照优先级顺序调整：左、右、上、下
            # 如果左边越界，从右边补足（保持总宽度不变）
            if col_min < 0:
                overflow_left = -col_min
                col_min = 0
                # 从右边补足
                col_max = min(col_max + overflow_left, dsm_width)
            
            # 如果右边越界，从左边补足（优先保证左边，但需要调整）
            if col_max > dsm_width:
                overflow_right = col_max - dsm_width
                col_max = dsm_width
                # 从左边补足（但不超过已确定的col_min）
                col_min = max(0, col_min - overflow_right)
            
            # 如果上边越界，从下边补足（保持总高度不变）
            if row_min < 0:
                overflow_top = -row_min
                row_min = 0
                # 从下边补足
                row_max = min(row_max + overflow_top, dsm_height)
            
            # 如果下边越界，从上边补足（优先保证上边，但需要调整）
            if row_max > dsm_height:
                overflow_bottom = row_max - dsm_height
                row_max = dsm_height
                # 从上边补足（但不超过已确定的row_min）
                row_min = max(0, row_min - overflow_bottom)

    # 再次确保范围有效并转换为整数（数组切片必须使用整数）
    row_min = int(max(0, row_min))
    col_min = int(max(0, col_min))
    row_max = int(min(dsm_height, row_max))
    col_max = int(min(dsm_width, col_max))
    
    # 如果使用斜向长方形，在最终确定 row_min/col_min 后计算顶点的相对坐标
    if target_ratio is None and 'use_rotated_rect' in locals() and use_rotated_rect and 'rotated_rect_pixel_coords' in locals() and rotated_rect_pixel_coords is not None:
        rotated_rect_vertices_relative = [(col - col_min, row - row_min) for row, col in rotated_rect_pixel_coords]
        print(f"  长方形顶点相对坐标（最终）: {rotated_rect_vertices_relative}")
    elif 'rotated_rect_vertices_relative' not in locals():
        rotated_rect_vertices_relative = None
    
    # 打印调试信息
    final_bbox_width = col_max - col_min
    final_bbox_height = row_max - row_min
    final_bbox_area = final_bbox_width * final_bbox_height

    
    # 计算实际比例（不含padding的外接矩形）
    actual_bbox_width = col_max - col_min - 2 * crop_padding
    actual_bbox_height = row_max - row_min - 2 * crop_padding
    actual_bbox_area = actual_bbox_width * actual_bbox_height
    actual_ratio = trapezoid_area / actual_bbox_area if actual_bbox_area > 0 else 0
    
    if target_ratio is None:
        print(f"原始梯形 (Ori): 梯形面积={trapezoid_area:.0f}px², 外接矩形={actual_bbox_width}x{actual_bbox_height}px, 自然比例={actual_ratio*100:.2f}%")
    else:
        print(f"目标比例 {target_ratio*100:.0f}%: 梯形面积={trapezoid_area:.0f}px², ref图={actual_bbox_width}x{actual_bbox_height}px, 实际比例={actual_ratio*100:.2f}%")


    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    # 返回裁剪的 offset，用于坐标转换
    # rotated_rect_vertices_relative 为斜向长方形顶点（在裁剪图像内的相对坐标）
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min), (row_min, col_min), rotated_rect_vertices_relative

def crop_dsm_dom_mount_angle(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, center_point,
                 crop_padding=0, target_ratio=0.55):
    """
    从DOM/DSM裁剪：通过图像中心点计算高度，然后按比例裁剪。
    高度通过图像中心点和DSM相交计算。
    """
    # 首先通过图像中心点和DSM相交来计算高度和相交点坐标
    center_intersect_point = get_height_from_center_pixel(K, pose_w2c, center_point, dsm_data, dsm_transform)

    center_height = center_intersect_point[2]

    # 计算四个角点的世界坐标
    world_points = []
    dsm_indices = []
    for uv in image_points:
        # 使用通过DSM计算的高度
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=center_height)
        world_points.append(xyz)
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    # ori模式：使用中心点投影和短边构建斜向长方形
    if target_ratio is None:
        # 步骤1: 找到梯形的短边（两条平行边中较短的那条）
        short_edge_info = find_short_parallel_edge(world_points, parallel_threshold=0.15)
        
        if short_edge_info is None:
            # 如果找不到平行边，回退到外接矩形方法
            print("警告：无法找到平行边，回退到外接矩形方法")
            row_min_raw = max(min(rows), 0)
            row_max_raw = min(max(rows), dsm_height)
            col_min_raw = max(min(cols), 0)
            col_max_raw = min(max(cols), dsm_width)
            row_min = row_min_raw
            row_max = row_max_raw
            col_min = col_min_raw
            col_max = col_max_raw
            bbox_width = col_max_raw - col_min_raw
            bbox_height = row_max_raw - row_min_raw
            bbox_area = bbox_width * bbox_height
            if bbox_width <= 0 or bbox_height <= 0:
                bbox_width = max(bbox_width, 1)
                bbox_height = max(bbox_height, 1)
                bbox_area = bbox_width * bbox_height
            trapezoid_polygon = [(col, row) for row, col in dsm_indices]
            dsm_map_rect = (0, 0, dsm_width, dsm_height)
            trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
            use_rotated_rect = False
            rotated_rect_pixel_coords = None
        else:
            short_edge_length, short_edge_direction, start_idx, end_idx = short_edge_info
            print(f"找到短边：长度={short_edge_length:.2f}m, 方向=({short_edge_direction[0]:.3f}, {short_edge_direction[1]:.3f})")

            # 山区直接使用中心点与 DSM 相交得到的中心投影点
            center_xyz = center_intersect_point
            print(f"中心点投影（世界坐标）: ({center_xyz[0]:.2f}, {center_xyz[1]:.2f})")

            img_width = max([pt[0] for pt in image_points]) - min([pt[0] for pt in image_points]) + 1
            img_height = max([pt[1] for pt in image_points]) - min([pt[1] for pt in image_points]) + 1
            image_aspect_ratio = img_width / img_height if img_height > 0 else 1.0
            rect_vertices = build_rotated_rectangle_from_center_and_edge(
                center_xyz, short_edge_length, short_edge_direction, image_aspect_ratio
            )
            print(f"长方形顶点（世界坐标）:")
            for i, v in enumerate(rect_vertices):
                print(f"  顶点{i+1}: ({v[0]:.2f}, {v[1]:.2f})")

            rotated_rect_pixel_coords = []
            for vertex in rect_vertices:
                row, col = geo_coords_to_dsm_index(vertex[0], vertex[1], dsm_transform)
                rotated_rect_pixel_coords.append((row, col))

            rect_rows, rect_cols = zip(*rotated_rect_pixel_coords)
            row_min_raw = max(min(rect_rows), 0)
            row_max_raw = min(max(rect_rows), dsm_height)
            col_min_raw = max(min(rect_cols), 0)
            col_max_raw = min(max(rect_cols), dsm_width)
            
            # 使用边界框作为裁剪区域（参考现有逻辑的边界处理）
            row_min = row_min_raw
            row_max = row_max_raw
            col_min = col_min_raw
            col_max = col_max_raw
            
            # 计算边界框的尺寸和面积（用于后续计算）
            bbox_width = col_max_raw - col_min_raw
            bbox_height = row_max_raw - row_min_raw
            bbox_area = bbox_width * bbox_height
            
            # 确保宽高有效
            if bbox_width <= 0 or bbox_height <= 0:
                bbox_width = max(bbox_width, 1)
                bbox_height = max(bbox_height, 1)
                bbox_area = bbox_width * bbox_height
            
            # 计算梯形与DSM地图的交集面积（用于后续比例计算）
            trapezoid_polygon = [(col, row) for row, col in dsm_indices]
            dsm_map_rect = (0, 0, dsm_width, dsm_height)
            trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
            
            use_rotated_rect = True
    else:
        # 非ori模式：使用轴对齐边界框（AABB），与 crop_dsm_dom_angle 保持一致
        use_rotated_rect = False
        rotated_rect_pixel_coords = None
        row_min_raw = max(min(rows), 0)
        row_max_raw = min(max(rows), dsm_height)
        col_min_raw = max(min(cols), 0)
        col_max_raw = min(max(cols), dsm_width)
        
        bbox_width = col_max_raw - col_min_raw
        bbox_height = row_max_raw - row_min_raw
        bbox_area = bbox_width * bbox_height

        
        # 确保宽高有效
        if bbox_width <= 0 or bbox_height <= 0:
            bbox_width = max(bbox_width, 1)
            bbox_height = max(bbox_height, 1)
            bbox_area = bbox_width * bbox_height
        
        # 计算梯形与DSM地图的交集面积（用于后续比例计算）
        # 构建梯形的顶点列表
        trapezoid_polygon = [(col, row) for row, col in dsm_indices]
        # 构建DSM地图的边界矩形（整个地图范围）
        dsm_map_rect = (0, 0, dsm_width, dsm_height)
        # 计算梯形在地图范围内的交集面积（只计算在地图内的部分）
        trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, dsm_map_rect)
        
        # 步骤2: 根据target_ratio计算scale_factor，使得梯形面积占最终crop区域的比例为target_ratio
        # 目标：(trapezoid_area) / (最终crop区域面积) = target_ratio
        # 最终crop区域面积 = bbox_area * scale_factor^2
        # 所以：trapezoid_area / (bbox_area * scale_factor^2) = target_ratio
        # 解得：scale_factor = sqrt(trapezoid_area / (bbox_area * target_ratio))
        scale_factor = np.sqrt(trapezoid_area / (bbox_area * target_ratio)) if bbox_area > 0 and trapezoid_area > 0 else 1.0
        
        # 如果 scale_factor < 1，说明需要缩小，但这里只扩展不缩减，直接使用原始外接矩形
        if scale_factor < 1.0:
            # 直接使用原始外接矩形，不进行任何扩展或缩减操作
            col_min = col_min_raw
            col_max = col_max_raw
            row_min = row_min_raw
            row_max = row_max_raw
        else:
            # 步骤3: 按照最小外接矩形的宽高比，计算最终的crop尺寸
            final_width = int(round(bbox_width * scale_factor))
            final_height = int(round(bbox_height * scale_factor))
            
            # 步骤4: 在梯形外围按照优先级顺序扩展像素：左、右、上、下
            # 计算需要扩展的像素数
            width_expand = final_width - bbox_width
            height_expand = final_height - bbox_height
            
            # 按优先级顺序分配扩展像素：左、右、上、下
            # 宽度扩展：优先左边，然后右边
            left_expand = int(width_expand/2)  # 优先全部从左边扩展
            right_expand = int(width_expand/2)
            # 如果左边扩展后无法满足（会越界），则从右边补足
            # 这里暂时全部从左边扩展，后续越界检查会调整
            
            # 高度扩展：优先上边，然后下边
            top_expand = int(height_expand/2)  # 优先全部从上边扩展
            bottom_expand = int(height_expand/2)
            # 如果上边扩展后无法满足（会越界），则从下边补足
            # 这里暂时全部从上边扩展，后续越界检查会调整
            
            # 从梯形的最小外接矩形开始扩展
            col_min = int(col_min_raw - left_expand)
            col_max = int(col_max_raw + right_expand)
            row_min = int(row_min_raw - top_expand)
            row_max = int(row_max_raw + bottom_expand)

            # 若越界，按照优先级顺序调整：左、右、上、下
            # 如果左边越界，从右边补足（保持总宽度不变）
            if col_min < 0:
                overflow_left = -col_min
                col_min = 0
                # 从右边补足
                col_max = min(col_max + overflow_left, dsm_width)
            
            # 如果右边越界，从左边补足（优先保证左边，但需要调整）
            if col_max > dsm_width:
                overflow_right = col_max - dsm_width
                col_max = dsm_width
                # 从左边补足（但不超过已确定的col_min）
                col_min = max(0, col_min - overflow_right)
            
            # 如果上边越界，从下边补足（保持总高度不变）
            if row_min < 0:
                overflow_top = -row_min
                row_min = 0
                # 从下边补足
                row_max = min(row_max + overflow_top, dsm_height)
            
            # 如果下边越界，从上边补足（优先保证上边，但需要调整）
            if row_max > dsm_height:
                overflow_bottom = row_max - dsm_height
                row_max = dsm_height
                # 从上边补足（但不超过已确定的row_min）
                row_min = max(0, row_min - overflow_bottom)

    # 再次确保范围有效并转换为整数（数组切片必须使用整数）
    row_min = int(max(0, row_min))
    col_min = int(max(0, col_min))
    row_max = int(min(dsm_height, row_max))
    col_max = int(min(dsm_width, col_max))
    
    # 如果使用斜向长方形，在最终确定 row_min/col_min 后计算顶点的相对坐标
    if target_ratio is None and 'use_rotated_rect' in locals() and use_rotated_rect and 'rotated_rect_pixel_coords' in locals() and rotated_rect_pixel_coords is not None:
        rotated_rect_vertices_relative = [(col - col_min, row - row_min) for row, col in rotated_rect_pixel_coords]
        print(f"  长方形顶点相对坐标（最终）: {rotated_rect_vertices_relative}")
    elif 'rotated_rect_vertices_relative' not in locals():
        rotated_rect_vertices_relative = None
    
    # 打印调试信息
    final_bbox_width = col_max - col_min
    final_bbox_height = row_max - row_min
    final_bbox_area = final_bbox_width * final_bbox_height
    
    # 计算实际比例（不含padding的外接矩形）
    actual_bbox_width = col_max - col_min - 2 * crop_padding
    actual_bbox_height = row_max - row_min - 2 * crop_padding
    actual_bbox_area = actual_bbox_width * actual_bbox_height
    actual_ratio = trapezoid_area / actual_bbox_area if actual_bbox_area > 0 else 0
    
    if target_ratio is None:
        print(f"原始梯形 (Ori): 梯形面积={trapezoid_area:.0f}px², 外接矩形={actual_bbox_width}x{actual_bbox_height}px, 自然比例={actual_ratio*100:.2f}%")
    else:
        print(f"目标比例 {target_ratio*100:.0f}%: 梯形面积={trapezoid_area:.0f}px², ref图={actual_bbox_width}x{actual_bbox_height}px, 实际比例={actual_ratio*100:.2f}%")

    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    # 返回裁剪的 offset，用于坐标转换
    # rotated_rect_vertices_relative 为斜向长方形顶点（在裁剪图像内的相对坐标）
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min), (row_min, col_min), rotated_rect_vertices_relative

