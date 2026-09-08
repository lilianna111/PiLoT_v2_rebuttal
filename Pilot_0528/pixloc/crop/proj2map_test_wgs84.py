import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyproj
import rasterio
from osgeo import gdal
from PIL import Image


# =========================
# 坐标变换：WGS84 <-> ECEF
# =========================
_WGS84_TO_ECEF = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:4978", always_xy=True
)
_ECEF_TO_WGS84 = pyproj.Transformer.from_crs(
    "EPSG:4978", "EPSG:4326", always_xy=True
)


def lonlatalt_to_ecef(lon, lat, alt):
    x, y, z = _WGS84_TO_ECEF.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_lonlatalt(x, y, z):
    lon, lat, alt = _ECEF_TO_WGS84.transform(x, y, z)
    return lon, lat, alt


def get_enu_to_ecef_rotation(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    east = np.array([
        -np.sin(lon),
         np.cos(lon),
         0.0
    ], dtype=np.float64)

    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat)
    ], dtype=np.float64)

    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ], dtype=np.float64)

    return np.column_stack([east, north, up])


def get_ecef_to_enu_rotation(lon_deg, lat_deg):
    return get_enu_to_ecef_rotation(lon_deg, lat_deg).T


def ecef_points_to_local_enu_xy(points_ecef, origin_ecef=None):
    """
    points_ecef: (N,3) 或 (3,)
    返回:
        pts_xy: (N,2)
        origin_ecef
    """
    pts = np.asarray(points_ecef, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]

    if origin_ecef is None:
        origin_ecef = pts.mean(axis=0)
    origin_ecef = np.asarray(origin_ecef, dtype=np.float64)

    lon0, lat0, _ = ecef_to_lonlatalt(*origin_ecef)
    R_ecef_to_enu = get_ecef_to_enu_rotation(lon0, lat0)

    local = (pts - origin_ecef[None, :]) @ R_ecef_to_enu.T
    return local[:, :2], origin_ecef


def build_ecef_grid_from_geo(xs, ys, zz):
    """
    xs: lon 数组
    ys: lat 数组
    zz: 高程矩阵 shape=(len(ys), len(xs))
    返回: (H,W,3) 的 ECEF XYZ
    """
    xx, yy = np.meshgrid(xs, ys)
    x, y, z = _WGS84_TO_ECEF.transform(xx, yy, zz)
    points_3d = np.stack([x, y, z], axis=-1).astype(np.float64)
    return points_3d


# =========================
# DSM / DOM 读取
# =========================
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

    valid_mask = np.isfinite(area) & (area > -1000) & (area < 10000)
    if not np.any(valid_mask):
        raise RuntimeError("DSM 中没有可用高程值，无法计算 area_minZ")

    area_minZ = np.median(area[valid_mask])
    print(f"平原地形：使用中位数作为最低高程 = {area_minZ:.3f}m")
    print("data size:", band.XSize, "x", band.YSize)
    print("area_minZ (valid minimum elevation):", area_minZ)
    print("Loading time cost:", time.time() - start, "s")

    return geotransform, area, area_minZ, dsm_data, dsm_transform, dom_data


# =========================
# 像素 -> 射线 / 地面点
# =========================
def ray_intersect_altitude_plane(ray_origin_ecef, ray_dir_ecef, target_alt, max_iter=60):
    """
    与固定大地高程面 alt = target_alt 求交
    在 ECEF 射线上用二分搜索
    """
    ray_origin_ecef = np.asarray(ray_origin_ecef, dtype=np.float64)
    ray_dir_ecef = np.asarray(ray_dir_ecef, dtype=np.float64)
    ray_dir_ecef = ray_dir_ecef / np.linalg.norm(ray_dir_ecef)

    def f(t):
        p = ray_origin_ecef + t * ray_dir_ecef
        _, _, alt = ecef_to_lonlatalt(*p)
        return alt - target_alt

    t0 = 0.0
    t1 = 100.0
    f0 = f(t0)
    f1 = f(t1)

    expand_count = 0
    while f0 * f1 > 0 and expand_count < 40:
        t1 *= 2.0
        f1 = f(t1)
        expand_count += 1

    if f0 * f1 > 0:
        return None

    for _ in range(max_iter):
        tm = 0.5 * (t0 + t1)
        fm = f(tm)

        if abs(fm) < 1e-3:
            return ray_origin_ecef + tm * ray_dir_ecef

        if f0 * fm <= 0:
            t1, f1 = tm, fm
        else:
            t0, f0 = tm, fm

    return ray_origin_ecef + 0.5 * (t0 + t1) * ray_dir_ecef


def pixel_to_world(K, pose_w2c, uv, target_z):
    """
    像素投影到固定高程面 target_z
    返回 ECEF XYZ
    """
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    cam_center = c2w[:3, 3]

    world_pt = ray_intersect_altitude_plane(cam_center, ray_dir, target_z)
    if world_pt is None:
        raise RuntimeError(f"无法将像素 {uv} 投影到 altitude={target_z}m 的参考面上")

    return world_pt


def pixel_to_ray_dir(K, pose_w2c, uv):
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    return ray_dir / np.linalg.norm(ray_dir)


def geo_coords_to_dsm_index(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def ray_intersect_dsm(ray_origin, ray_dir, dsm_data, dsm_transform, max_distance=10000.0):
    """
    在 ECEF 中沿射线采样，查询 EPSG:4326 DSM
    返回 ECEF 交点
    """
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    step = 1.0
    num_samples = int(max_distance / step)

    best_point = None
    best_alt = -np.inf

    for i in range(num_samples):
        distance = i * step
        point_ecef = ray_origin + ray_dir * distance

        lon, lat, point_alt = ecef_to_lonlatalt(*point_ecef)
        row, col = geo_coords_to_dsm_index(lon, lat, dsm_transform)

        if 0 <= row < dsm_data.shape[0] and 0 <= col < dsm_data.shape[1]:
            dsm_z = dsm_data[row, col]
            if np.isfinite(dsm_z):
                if point_alt <= dsm_z + 0.1:
                    point_on_dsm = lonlatalt_to_ecef(lon, lat, dsm_z)
                    if dsm_z > best_alt:
                        best_alt = dsm_z
                        best_point = point_on_dsm

    return best_point


def get_height_from_center_pixel(K, pose_w2c, center_point, dsm_data, dsm_transform):
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    cam_center = c2w[:3, 3]

    ray_dir = pixel_to_ray_dir(K, pose_w2c, center_point)
    intersect_point = ray_intersect_dsm(cam_center, ray_dir, dsm_data, dsm_transform)

    if intersect_point is not None:
        lon, lat, height = ecef_to_lonlatalt(*intersect_point)
        print(f"通过中心点计算的DSM高度: {height:.2f}m @ lon={lon:.8f}, lat={lat:.8f}")
        return intersect_point
    else:
        print("警告：无法通过中心点计算DSM高度，使用默认值")
        return None


# =========================
# 多边形 / 裁剪辅助
# =========================
def calculate_polygon_area(vertices):
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
    x1, y1 = p1
    x2, y2 = p2
    x_min, y_min, x_max, y_max = rect

    if (x1 < x_min and x2 < x_min) or (x1 > x_max and x2 > x_max) or \
       (y1 < y_min and y2 < y_min) or (y1 > y_max and y2 > y_max):
        return []

    if x_min <= x1 <= x_max and x_min <= x2 <= x_max and \
       y_min <= y1 <= y_max and y_min <= y2 <= y_max:
        return [(x1, y1), (x2, y2)]

    points = []
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            return [(x1, y1)]
        return []

    if abs(dx) > 1e-9:
        for x_bound in [x_min, x_max]:
            t = (x_bound - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if y_min <= y <= y_max:
                    points.append((x_bound, y))

    if abs(dy) > 1e-9:
        for y_bound in [y_min, y_max]:
            t = (y_bound - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if x_min <= x <= x_max:
                    points.append((x, y_bound))

    if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
        points.append((x1, y1))
    if x_min <= x2 <= x_max and y_min <= y2 <= y_max:
        points.append((x2, y2))

    if len(points) == 0:
        return []

    if len(points) == 1:
        return [points[0]]

    def dist_to_p1(p):
        return ((p[0] - x1) ** 2 + (p[1] - y1) ** 2) ** 0.5

    sorted_points = sorted(set(points), key=dist_to_p1)

    if len(sorted_points) >= 2:
        return [sorted_points[0], sorted_points[-1]]
    return sorted_points


def clip_polygon_to_rect(polygon, rect):
    if len(polygon) == 0:
        return []

    clipped_points = []
    n = len(polygon)

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        clipped_edge = clip_line_to_rect(p1, p2, rect)

        for p in clipped_edge:
            if len(clipped_points) == 0 or clipped_points[-1] != p:
                clipped_points.append(p)

    if len(clipped_points) > 1 and clipped_points[0] == clipped_points[-1]:
        clipped_points = clipped_points[:-1]

    return clipped_points


def calculate_polygon_rect_intersection_area(polygon, rect):
    clipped_polygon = clip_polygon_to_rect(polygon, rect)
    if len(clipped_polygon) < 3:
        return 0.0
    return calculate_polygon_area(clipped_polygon)


def create_polygon_mask(image_shape_hw, polygon_vertices):
    height, width = image_shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon_vertices is None or len(polygon_vertices) < 3:
        mask[:, :] = 255
        return mask

    pts = np.array(polygon_vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_white_mask_to_dom_crop(dom_crop, mask):
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


# =========================
# ECEF 下的几何 crop 辅助
# =========================
def find_short_parallel_edge(world_points, parallel_threshold=0.15):
    """
    在局部 ENU 平面里找四边形中两条近似平行边里较短的那条
    返回:
        (length, direction_xy, start_idx, end_idx, origin_ecef)
    """
    if len(world_points) < 4:
        return None

    pts_xy, origin_ecef = ecef_points_to_local_enu_xy(
        np.asarray(world_points, dtype=np.float64)
    )

    edges = []
    for i in range(4):
        p1 = pts_xy[i]
        p2 = pts_xy[(i + 1) % 4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        direction = vec / length if length > 1e-12 else vec
        edges.append((vec, direction, length, i, (i + 1) % 4))

    parallel_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            _, dir1, len1, _, _ = edges[i]
            _, dir2, len2, _, _ = edges[j]
            parallel_degree = abs(np.dot(dir1, dir2))
            if parallel_degree > (1 - parallel_threshold):
                parallel_pairs.append((parallel_degree, i, j, len1, len2))

    if not parallel_pairs:
        return None

    _, e1, e2, len1, len2 = max(parallel_pairs, key=lambda x: x[0])
    chosen = e1 if len1 < len2 else e2
    _, direction, length, start_idx, end_idx = edges[chosen]

    return length, direction, start_idx, end_idx, origin_ecef


def build_rotated_rectangle_from_center_and_edge(center_world, edge_length_along_short_direction,
                                                 short_edge_direction, image_aspect_ratio):
    """
    在局部 ENU 平面构造长方形，再转回 ECEF
    返回 4 个 ECEF 顶点
    """
    center_world = np.asarray(center_world, dtype=np.float64)

    center_xy, origin_ecef = ecef_points_to_local_enu_xy(
        center_world, origin_ecef=center_world
    )
    center_xy = center_xy[0]

    width_along_u = float(edge_length_along_short_direction)

    if image_aspect_ratio <= 0:
        image_aspect_ratio = 1.0
    height_along_v = width_along_u / image_aspect_ratio

    u = np.array(short_edge_direction, dtype=np.float64)
    if np.linalg.norm(u) < 1e-12:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = u / np.linalg.norm(u)

    v = np.array([-u[1], u[0]], dtype=np.float64)

    half_u = width_along_u / 2.0
    half_v = height_along_v / 2.0

    rect_enu = [
        center_xy + half_u * u + half_v * v,
        center_xy + half_u * u - half_v * v,
        center_xy - half_u * u - half_v * v,
        center_xy - half_u * u + half_v * v,
    ]

    lon0, lat0, _ = ecef_to_lonlatalt(*origin_ecef)
    R_enu_to_ecef = get_enu_to_ecef_rotation(lon0, lat0)

    rect_ecef = []
    for enu_xy in rect_enu:
        enu_vec = np.array([enu_xy[0], enu_xy[1], 0.0], dtype=np.float64)
        rect_ecef.append(origin_ecef + R_enu_to_ecef @ enu_vec)

    return rect_ecef


# =========================
# 主裁剪函数：ECEF 计算 + 4326 索引
# =========================
def crop_dsm_dom_angle(dom_data, dsm_data, dsm_transform,
                       K, pose_w2c, image_points, center_point, area_minZ,
                       crop_padding=0, target_ratio=None):
    """
    说明：
    1. 图像角点先投到 ECEF 参考高程面
    2. 在局部 ENU 里做短边方向 / 长方形构建
    3. 最终裁 DSM/DOM 时仍用 4326 的 lon/lat 去索引
    """
    world_points = []
    dsm_indices = []

    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)

        lon_i, lat_i, _ = ecef_to_lonlatalt(*xyz)
        idx = geo_coords_to_dsm_index(lon_i, lat_i, dsm_transform)
        dsm_indices.append(idx)

    center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)

    result = find_short_parallel_edge(world_points)
    if result is None:
        raise RuntimeError("无法从四边形角点中找到短边方向")

    short_len, short_dir, _, _, _ = result

    width = K[0, 2] * 2
    height = K[1, 2] * 2
    image_aspect_ratio = width / height

    rect_vertices_ecef = build_rotated_rectangle_from_center_and_edge(
        center_xyz,
        edge_length_along_short_direction=short_len,
        short_edge_direction=short_dir,
        image_aspect_ratio=image_aspect_ratio
    )

    rect_indices = []
    for vertex in rect_vertices_ecef:
        lon_v, lat_v, _ = ecef_to_lonlatalt(*vertex)
        row_v, col_v = geo_coords_to_dsm_index(lon_v, lat_v, dsm_transform)
        rect_indices.append((row_v, col_v))

    rows, cols = zip(*rect_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    if row_max <= row_min or col_max <= col_min:
        raise RuntimeError("裁剪范围无效，请检查姿态、DSM范围和投影结果")

    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c   # lon
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f   # lat
    zz = dsm_crop

    if zz.shape[0] != len(ys) or zz.shape[1] != len(xs):
        zz_fixed = np.full((len(ys), len(xs)), fill_value=area_minZ, dtype=np.float64)
        r = min(zz.shape[0], zz_fixed.shape[0])
        c = min(zz.shape[1], zz_fixed.shape[1])
        zz_fixed[:r, :c] = zz[:r, :c]
        zz = zz_fixed

    zz = np.nan_to_num(zz, nan=area_minZ)
    points_3d = build_ecef_grid_from_geo(xs, ys, zz)

    dsm_indices_xy = [(col, row) for row, col in dsm_indices]
    crop_offset = (row_min, col_min)

    rotated_rect_vertices_relative = []
    for row_v, col_v in rect_indices:
        rotated_rect_vertices_relative.append([
            col_v - col_min,
            row_v - row_min
        ])

    min_shape = min(col_max - col_min, row_max - row_min)

    return (
        dom_crop,
        points_3d,
        np.array(dsm_indices_xy),
        world_points,
        min_shape,
        crop_offset,
        np.array(rotated_rect_vertices_relative, dtype=np.int32)
    )


# =========================
# 主入口：生成 reference
# =========================
def generate_ref_map(name, query_intrinsics, query_poses, area_minZ,
                     dsm_data, dsm_transform, dom_data,
                     ref_rgb_path, ref_depth_path,
                     debug, crop_vis_save_path=None,
                     test_ratio=None, use_bbox=True):
    """
    当前默认只走 use_bbox=True 的分支
    即：
    - 中心点投影
    - 用梯形短边方向构建斜向长方形
    """
    K_w2c = query_intrinsics
    pose_w2c = query_poses

    width, height = K_w2c[0, 2] * 2, K_w2c[1, 2] * 2
    image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    center_point = (width // 2, height // 2)

    print('--------------', name, '--------------')

    if use_bbox:
        print("  使用中心点+短边构建斜向长方形裁剪")
        target_ratio = None
    else:
        target_ratio = test_ratio

    dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, crop_offset, rotated_rect_vertices_relative = crop_dsm_dom_angle(
        dom_data, dsm_data, dsm_transform,
        K_w2c, pose_w2c, image_points, center_point, area_minZ,
        crop_padding=0, target_ratio=target_ratio
    )

    if use_bbox:
        ratio_suffix = '_bbox'
    else:
        ratio_suffix = f'_ratio_{int(test_ratio * 100)}'

    ratio_img_path = os.path.join(ref_rgb_path, f'{name}{ratio_suffix}_dom.jpg')
    ratio_npy_path = os.path.join(ref_depth_path, f'{name}{ratio_suffix}_dom.npy')
    ratio_mask_path = os.path.join(ref_rgb_path, f'{name}{ratio_suffix}_dom_mask.png')

    os.makedirs(ref_rgb_path, exist_ok=True)
    os.makedirs(ref_depth_path, exist_ok=True)

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

    np.save(ratio_npy_path, point_cloud_crop)
    print(f"    💾 已保存: {ratio_img_path}")
    print(f"    💾 已保存: {ratio_npy_path}")
    print(f"    点云坐标系: ECEF XYZ")

    ref_h = dom_crop.shape[1]
    ref_w = dom_crop.shape[2] if len(dom_crop.shape) == 3 else dom_crop.shape[1]
    ref_total_area = ref_w * ref_h

    row_min, col_min = crop_offset
    actual_rect = (col_min, row_min, col_min + ref_w, row_min + ref_h)

    if isinstance(dsm_indices, np.ndarray):
        trapezoid_polygon = [(col, row) for col, row in dsm_indices]
    else:
        trapezoid_polygon = [(col, row) for col, row in dsm_indices]

    intersection_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, actual_rect)
    actual_ratio_percent = (intersection_area / ref_total_area) * 100 if ref_total_area > 0 else 0.0

    if debug and crop_vis_save_path:
        os.makedirs(crop_vis_save_path, exist_ok=True)

        ref_vis = np.transpose(dom_crop, (1, 2, 0)).copy()
        ref_height, ref_width = ref_vis.shape[0], ref_vis.shape[1]

        row_min, col_min = crop_offset
        ref_indices = dsm_indices.copy()
        ref_indices[:, 0] = ref_indices[:, 0] - col_min
        ref_indices[:, 1] = ref_indices[:, 1] - row_min

        min_dimension = min(ref_width, ref_height)
        if min_dimension < 500:
            adaptive_font_scale = 0.5
        elif min_dimension < 1000:
            adaptive_font_scale = 0.6
        else:
            adaptive_font_scale = 0.7

        adaptive_thickness = 1
        line_thickness = 2
        circle_radius = 5

        polygon_vertices = [(float(x), float(y)) for x, y in ref_indices]
        rect = (0.0, 0.0, float(ref_width - 1), float(ref_height - 1))
        n = len(polygon_vertices)

        for i in range(n):
            p1 = polygon_vertices[i]
            p2 = polygon_vertices[(i + 1) % n]
            clipped_edge = clip_line_to_rect(p1, p2, rect)
            if len(clipped_edge) >= 2:
                edge_points = np.array(clipped_edge, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(ref_vis, [edge_points], isClosed=False, color=(255, 0, 0), thickness=line_thickness)

        for idx, (x, y) in enumerate(ref_indices):
            x, y = float(x), float(y)
            if 0 <= x < ref_width and 0 <= y < ref_height:
                cv2.circle(ref_vis, (int(x), int(y)), radius=circle_radius, color=(0, 255, 255), thickness=-1)
                cv2.putText(ref_vis, str(idx), (int(x) + 5, int(y) - 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                            color=(0, 0, 0), thickness=adaptive_thickness, lineType=cv2.LINE_AA)

        if rotated_rect_vertices_relative is not None:
            rect_points = np.array(rotated_rect_vertices_relative, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(ref_vis, [rect_points], isClosed=True, color=(0, 255, 0),
                          thickness=line_thickness + 2, lineType=cv2.LINE_AA)

            for rect_idx, (x, y) in enumerate(rotated_rect_vertices_relative):
                x, y = float(x), float(y)
                if 0 <= x < ref_width and 0 <= y < ref_height:
                    cv2.circle(ref_vis, (int(x), int(y)), radius=circle_radius + 2,
                               color=(0, 255, 0), thickness=-1)
                    cv2.putText(ref_vis, f'R{rect_idx}', (int(x) + 5, int(y) - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                                color=(0, 0, 0), thickness=adaptive_thickness + 1, lineType=cv2.LINE_AA)

        try:
            mid_y_img = height / 2.0
            left_mid_uv = (0, mid_y_img)
            right_mid_uv = (width - 1, mid_y_img)

            world_left_mid = pixel_to_world(K_w2c, pose_w2c, left_mid_uv, target_z=area_minZ)
            world_right_mid = pixel_to_world(K_w2c, pose_w2c, right_mid_uv, target_z=area_minZ)

            midline_length_m = float(np.linalg.norm(world_right_mid - world_left_mid))

            lon_l, lat_l, _ = ecef_to_lonlatalt(*world_left_mid)
            lon_r, lat_r, _ = ecef_to_lonlatalt(*world_right_mid)

            row_ml, col_ml = geo_coords_to_dsm_index(lon_l, lat_l, dsm_transform)
            row_mr, col_mr = geo_coords_to_dsm_index(lon_r, lat_r, dsm_transform)

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

        ratio_text = f"A:{actual_ratio_percent:.1f}%"
        text_size, baseline = cv2.getTextSize(
            ratio_text, cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, adaptive_thickness + 1
        )
        text_width, text_height = text_size
        overlay = ref_vis.copy()
        cv2.rectangle(overlay, (5, 5), (text_width + 15, text_height + baseline + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, ref_vis, 0.4, 0, ref_vis)

        cv2.putText(ref_vis, ratio_text, (10, 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=adaptive_font_scale,
                    color=(255, 255, 255), thickness=adaptive_thickness + 1, lineType=cv2.LINE_AA)

        vis_save_path = os.path.join(crop_vis_save_path, f'{name}{ratio_suffix}.jpg')
        cv2.imwrite(vis_save_path, cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))
        print(f"    💾 保存可视化: {vis_save_path} ({ref_width}x{ref_height}, 实际比例:{actual_ratio_percent:.2f}%)")

    data = {
        f'{name}{ratio_suffix}': {
            'imgr_name': f'{name}{ratio_suffix}_dom',
            'exr_name': f'{name}{ratio_suffix}_dom',
            'min_shape': min_shape
        }
    }

    return data