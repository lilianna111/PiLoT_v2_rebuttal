import cv2
import numpy as np
import os
from osgeo import gdal
import time
from pathlib import Path
import torch
import pyproj
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

class TargetLocation():
    def __init__(self, config: Dict, use_dsm = False):
        self.config = config
        self.clicked_points = []
        self._dsm_cache = {}  # 缓存DSM数据，避免重复加载
        self._proj_cache = {}  # 缓存投影变换，避免重复构造 Transformer
        
        if use_dsm:
            DSM_path = self.config["ray_casting"]["DSM_path"]
            DSM_npy_path = self.config["ray_casting"]["DSM_npy_path"]
            geotransform_path = self.config["ray_casting"]["geotransform_path"]

            self.num_sample = self.config["ray_casting"]["num_sample"]
            self.area_minZ = self.config["ray_casting"]["area_minZ"]
            
            cur_dir = os.path.abspath(os.path.dirname(__file__))
            dsm_path = cur_dir + '/../' + DSM_path
            npy_path = cur_dir + '/../' + DSM_npy_path
            geotransform_path = cur_dir + "/../" + geotransform_path

            if os.path.isfile(npy_path):
                self.area = np.load(npy_path)
                self.geotransform = np.load(geotransform_path)
            else:
                self.area, self.geo_transform, self.area_minZ = self.dsm2npy(dsm_path, npy_path, geotransform_path)
            self._dsm_cache[DSM_path] = (self.area, self.geotransform, self.area_minZ)

    @staticmethod
    def dsm2npy(dsm_path, npy_path, geotransform_path):
        DSM_map = gdal.Open(dsm_path)
        band = DSM_map.GetRasterBand(1) 
        geotransform = DSM_map.GetGeoTransform() 
        area = band.ReadAsArray()
        del DSM_map
        mask = np.ma.masked_values(area, -9999)
        area_minZ = mask.min()
        np.save(npy_path, area)
        np.save(geotransform_path, geotransform)
        return area, geotransform, area_minZ

    def dms_to_dd(self, d, m, s):
        return d + (m / 60) + (s / 3600)

    def dd_to_dms(self, DD):
        degrees = int(DD)
        minutes = int((DD - degrees) * 60)
        seconds = (DD - degrees - minutes / 60) * 3600
        seconds = round(seconds, 2)
        dms = f"{degrees}\u00B0{minutes}\u2032{seconds}\u2033"
        return dms, degrees, minutes, seconds

    def interpolate_along_line(self, area, x, y, num_points):
        sample_values = map_coordinates(area, [y, x], order=1)
        sample_array = sample_values.reshape((num_points, ))
        return sample_array

    def pixel_to_world_coordinate(self, K, R, t, u, v):
        p_camera = np.array([[u], [v], [1]])
        p_camera = np.linalg.inv(K).dot(p_camera)
        p_world = R.dot(p_camera) + t
        return p_world

    def get_index_array(self, dataset):
        ds = dataset
        geotransform = ds.GetGeoTransform()
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
        rows, cols = ds.RasterYSize, ds.RasterXSize
        x = np.arange(cols) * x_pixel_size + x_origin
        y = np.arange(rows) * y_pixel_size + y_origin
        return x, y

    def line_equation_3d(self, point1, point2):
        p1 = np.array(point1)
        p2 = np.array(point2)
        direction = p2 - p1
        a, b, c = direction
        d = -(a * p1[0] + b * p1[1] + c * p1[2])
        return [a, b, c, d]

    def line_equation_2d(self, x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return [A, B, C]

    def line_equation(self, A, B, Z):
        x, y, z = A
        x1, y1, z1 = B
        t = np.array([x1 - x, y1 - y, z1 - z])
        ray = lambda k: np.array([x, y, z]) + k * t
        k = (Z - z) / t[2]
        projection = ray(k)[:2]
        return ray, projection

    def intersection(self, ray_eqn, Z):
        k = (Z - ray_eqn(0)[2]) / (ray_eqn(1)[2] - ray_eqn(0)[2])
        intersection_point = ray_eqn(k)
        return intersection_point

    def geo_coords_to_array_index(self, x, y, geotransform):
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
        col = ((x - x_origin) / x_pixel_size).astype(int)
        row = ((y - y_origin) / y_pixel_size).astype(int)
        return row, col

    def sample_points_on_line(self, line_equation, num_sample, x_minmax):
        A, B, C = line_equation[0], line_equation[1], line_equation[2]
        x_min, x_max = x_minmax[0], x_minmax[1]
        x = np.linspace(x_min, x_max, num_sample)
        y = (-A/B)*x - (C/B)
        return x, y

    def find_z(self, ray_eqn, points):
        k = (points[0] - ray_eqn(0)[0]) / (ray_eqn(1)[0] - ray_eqn(0)[0])
        z_values = [ray_eqn(k_i)[2] for k_i in k]
        return z_values

    def caculate_predictXYZ(self, K, pose, objPixelCoords, area, geotransform, area_minZ, num_sample):
        R = pose[:3, :3]
        t = pose[:3, 3].reshape([3,1])
        target = self.pixel_to_world_coordinate(K,R,t,objPixelCoords[0],objPixelCoords[1])
        ray_eqn, projection_eqn = self.line_equation(t, target, area_minZ)
        line2D_abcd = self.line_equation_2d(t[0], t[1], target[0], target[1])
        intersection_point = self.intersection(ray_eqn, area_minZ)
        x_minmax = [t[0],intersection_point[0]]
        x, y = self.sample_points_on_line(line2D_abcd, num_sample, x_minmax)
        row, col = self.geo_coords_to_array_index(x, y, geotransform)
        sampleHeight = self.interpolate_along_line(area, col ,row, num_sample)
        z_values = self.find_z(ray_eqn,[x, y])
        z_values = torch.tensor(np.asarray(z_values))
        z_values = z_values.squeeze()
        sampleHeight = torch.tensor(sampleHeight)
        abs_x = torch.abs(z_values - sampleHeight)
        min_val, min_idx = torch.min(abs_x, dim=0)
        result_x = x[min_idx]
        result_y = y[min_idx]
        result_z = z_values[min_idx]
        result_sampleHeight = sampleHeight[min_idx]
        origin_x = t[0].item()
        target_x = target[0].item()
        if abs(target_x - origin_x) > 1e-6:
            k_value = (result_x - origin_x) / (target_x - origin_x)
        else:
            origin_y = t[1].item()
            target_y = target[1].item()
            k_value = (result_y - origin_y) / (target_y - origin_y)
        return [result_x, result_y, result_z], k_value, result_sampleHeight 

    def get_intrinsic(self):
        image_width_px = 3840
        image_height_px = 2160
        sensor_width_mm = 3840
        sensor_height_mm = 2160
        f_mm = 2700
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
        cx = 1915.7
        cy = 1075.1
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        return K

    def get_query_intrinsic(self, camera):
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
        cx = image_width_px / 2
        cy = image_height_px / 2
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        return K

    def qvec2rotmat(self, qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

    def get_pose(self, ret):
        q, t = ret['qvec'], ret['tvec']
        R = np.asmatrix(self.qvec2rotmat(q)).transpose()
        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = -R.dot(t)
        return T

    # ================= 修改后的 predict_center_depth =================
# ================= 修改后的 predict_center_depth (含WGS84转换) =================
    def predict_center_depth(self, DSM_path, pose, num_sample=10000, object_pixel_coords=None):
        """
        预测图像中心点的深度，并输出4个关键指标（增加WGS84经纬度输出）
        """
        # 兼容旧接口，内部调用批量接口
        depths = self.predict_center_depth_batch(
            DSM_path, [pose], num_sample=num_sample, object_pixel_coords=object_pixel_coords
        )
        return depths[0]

    def _get_transformers(self, proj_crs="EPSG:4547"):
        """缓存 WGS84 <-> 投影的 Transformer，避免重复构造。"""
        if proj_crs not in self._proj_cache:
            transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
            transformer_back = pyproj.Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
            self._proj_cache[proj_crs] = (transformer, transformer_back)
        return self._proj_cache[proj_crs]

    def predict_center_depth_batch(self, DSM_path, poses, num_sample=10000, object_pixel_coords=None):
        """
        批量预测多个姿态的中心深度，减少重复IO与 Transformer 构造。
        poses: 可迭代对象，每个元素为 [lon, lat, alt, roll, pitch, yaw]
        返回: list[float]，与输入 poses 对应的深度
        """
        # 1. 准备数据
        if DSM_path in self._dsm_cache:
            area, geotransform, area_minZ = self._dsm_cache[DSM_path]
        else:
            DSM_npy_path = DSM_path.replace(".tif", ".npy")
            geotransform_path = DSM_path.replace(".tif", ".txt")
            
            if os.path.exists(DSM_npy_path) and os.path.exists(geotransform_path):
                area = np.load(DSM_npy_path)
                geotransform = np.load(geotransform_path)
                area_minZ = np.ma.masked_values(area, -9999).min()
            else:
                area, geotransform, area_minZ = self.dsm2npy(DSM_path, DSM_npy_path, geotransform_path)
            self._dsm_cache[DSM_path] = (area, geotransform, area_minZ)
        
        # 预备通用参数
        proj_crs = "EPSG:4547"  # 默认投影带
        transformer, transformer_back = self._get_transformers(proj_crs)
        coord_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        K = self.get_intrinsic()
        if object_pixel_coords is None:
            object_pixel_coords = [1920, 1080]
        depths = []

        # 支持 numpy 数组批量输入
        if isinstance(poses, np.ndarray):
            poses_iter = poses.tolist()
        else:
            poses_iter = list(poses)

        # 2. 计算变换矩阵 T
        for pose in poses_iter:
            raw_lon, raw_lat, raw_alt = pose[0], pose[1], pose[2]
            raw_roll, raw_pitch, raw_yaw = pose[3], pose[4], pose[5]

            x_meter, y_meter = transformer.transform(raw_lon, raw_lat)

            r_original = R.from_euler('xyz', [raw_pitch, raw_roll, raw_yaw], degrees=True)
            matrix_3x3 = r_original.as_matrix()

            T4x4 = np.eye(4)
            T4x4[:3, :3] = matrix_3x3
            T4x4[:3, 3] = [x_meter, y_meter, raw_alt]

            # 得到最终的 Camera -> World 矩阵
            T = T4x4 @ coord_transform

            R_new = T[:3, :3]

            # 3. 计算预测点 (Ray Casting) -> 得到投影坐标
            x_proj, _, _ = self.caculate_predictXYZ(
                K, T, object_pixel_coords, area, geotransform, area_minZ, num_sample=num_sample
            )

            # 格式转换
            P_center_list = []
            for item in x_proj:
                if torch.is_tensor(item):
                    P_center_list.append(item.item())
                elif isinstance(item, np.ndarray):
                    P_center_list.append(item.item())
                else:
                    P_center_list.append(item)
            P_center = np.array(P_center_list)  # [X_proj, Y_proj, Z]

            # --- 新增：将预测点转换为 WGS84 ---
            pred_lon, pred_lat = transformer_back.transform(P_center[0], P_center[1])
            pred_alt = P_center[2]

            # 4. 计算预测深度
            camera_point = R_new.T @ (P_center - T[0:3, 3])
            depth = camera_point[2]

            depths.append(depth)

        return depths


# # # # # ================= 主函数调用 =================
# if __name__ == "__main__":

#     config = {"ray_casting": { }}
#     locator = TargetLocation(config, use_dsm=False)
    
#     DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
#     npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.npy"
#     geotransform_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.txt"
#     area, geotransform, area_minZ = TargetLocation.dsm2npy(DSM_path, npy_path, geotransform_path)
#     # def dsm2npy(dsm_path, npy_path, geotransform_path):
    
#     # 你的 Pose
#     pose = [112.99913361737313, 28.29312016820301, 56.51444707904011, -0.009626261288695549, 20.457791201810448, -177.00059649573672]

#     # 【请在这里填入真实深度】(例如: 69.76 或者 None)
#     # 如果不知道真实深度，填 None，第3项就不会算出数值
#     real_depth = 167.5634
    
    
#     depth = locator.predict_center_depth(DSM_path, pose)
#     print(f"[4] 预测深度值:\n    {depth:.4f} 米")
#     print("-" * 60)




# if __name__ == "__main__":
#     # ================= 配置路径 =================
#     # 请确保路径正确
#     # DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"
#     # npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.npy"
#     # geotransform_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.txt"
#     # geotransform_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/DSM0.5.txt"
#     # DSM_path = "/home/amax/Documents/datasets/DEM/feicuiwandem.tif"
#     # npy_path = "/home/amax/Documents/datasets/DEM/fcw_hangtian.npy"

#     # ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/feicuiwan_mix.npy"
#     # DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DOM.tif"
#     # DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DSM.tif"
#     # npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/feicuiwan.npy"


#     # ================= 1. 加载 DSM 数据 =================
#     # 实例化类主要是为了使用它的工具函数
#     config = {"ray_casting": { }}
#     locator = TargetLocation(config, use_dsm=False)

#     print("[Info] Loading DSM data...")
#     if os.path.exists(npy_path) and os.path.exists(geotransform_path):
#         area = np.load(npy_path)
#         geotransform = np.load(geotransform_path)
#         print("[Info] Loaded from .npy cache.")
#     else:
#         print("[Info] Generating .npy from TIF...")
#         area, geotransform, area_minZ = TargetLocation.dsm2npy(DSM_path, npy_path, geotransform_path)

#     print('geotransform', geotransform)
#     print('area', area)
#     print('area_minZ', area_minZ)
    
#     # ================= 2. 输入坐标 (与DSM坐标系一致) =================
#     # 在这里输入你的 X, Y 坐标
#     # 例如：投影坐标系下的米制坐标
#     input_x = 401809.87
#     input_y = 3131367.68

#     print("-" * 50)
#     print(f"Querying Height for Coordinate:\n  X: {input_x}\n  Y: {input_y}")

#     # ================= 3. 坐标转换 (World -> Pixel) =================
#     # GeoTransform 格式: [top_left_x, w_e_pixel_res, rotation_0, top_left_y, rotation_1, n_s_pixel_resolution]
#     x_origin = geotransform[0]
#     x_pixel_size = geotransform[1]
#     y_origin = geotransform[3]
#     y_pixel_size = geotransform[5]

#     # 计算浮点数索引 (Col, Row)
#     # Col = (X - X_origin) / Pixel_Width
#     # Row = (Y - Y_origin) / Pixel_Height
#     col_float = (input_x - x_origin) / x_pixel_size
#     row_float = (input_y - y_origin) / y_pixel_size

#     # ================= 4. 双线性插值获取高度 =================
#     rows, cols = area.shape
    
#     # 检查坐标是否在图片范围内
#     if 0 <= col_float < cols and 0 <= row_float < rows:
#         # map_coordinates 需要的输入顺序是 [y(row), x(col)]
#         # 且需要传入数组形式
#         coords_y = np.array([row_float])
#         coords_x = np.array([col_float])
        
#         # 调用类中的插值函数
#         # 注意：area 是 (H, W) 的数组
#         height_array = locator.interpolate_along_line(area, coords_x, coords_y, num_points=1)
        
#         dsm_height = height_array[0]
        
#         print(f"\n[Result] DSM Height: {dsm_height:.4f} meters")
#         print(f"(Pixel Coordinates: Row={row_float:.2f}, Col={col_float:.2f})")
#     else:
#         print(f"\n[Error] Coordinate out of bounds!")
#         print(f"  Image Size: {cols}x{rows}")
#         print(f"  Calculated Pixel: Row={row_float:.2f}, Col={col_float:.2f}")

#     print("-" * 50)


if __name__ == "__main__":
    # ================= 1. 用户配置输入 =================
    # DSM 文件路径
    DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
    
    # 目标点的 WGS84 坐标 (经度, 纬度)
    # 例如: 长沙某地
    target_lon = 113.002454
    target_lat = 28.291432

    print(f"==========================================")
    print(f"任务: 查询 WGS84 坐标 ({target_lon}, {target_lat}) 的 DSM 高度")

    # ================= 2. 加载 DSM 数据 (Area & GeoTransform) =================
    # 自动推导缓存路径
    npy_path = DSM_path.replace(".tif", ".npy")
    txt_path = DSM_path.replace(".tif", ".txt")
    
    # 初始化工具类
    config = {"ray_casting": { }}
    locator = TargetLocation(config, use_dsm=False)

    print("[1/4] 加载 DSM 数据...")
    if os.path.exists(npy_path) and os.path.exists(txt_path):
        area = np.load(npy_path)
        geotransform = np.load(txt_path)
        print("      已加载 .npy 缓存文件")
    else:
        print("      未找到缓存，正在从 TIF 读取并生成 .npy ...")
        area, geotransform, _ = TargetLocation.dsm2npy(DSM_path, npy_path, txt_path)
    
    # ================= 3. 构建坐标转换器 (WGS84 -> DSM投影系) =================
    print("[2/4] 解析坐标系投影...")
    ds = gdal.Open(DSM_path)
    if ds is None:
        raise FileNotFoundError(f"无法打开 DSM 文件: {DSM_path}")
    
    # 获取 DSM 的投影 WKT
    proj_wkt = ds.GetProjection()
    
    # 构建 pyproj 转换器
    # 源: EPSG:4326 (WGS84 经纬度)
    # 目标: DSM 自身的投影 (从 WKT 读取)
    crs_src = pyproj.CRS.from_epsg(4326)
    try:
        crs_dst = pyproj.CRS.from_wkt(proj_wkt)
        transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        print(f"      检测到 DSM 投影: {crs_dst.name}")
    except Exception as e:
        # 如果读取失败，回退到用户常用的 EPSG:4547 (CGCS2000 / 3-degree Gauss-Kruger CM 114E)
        print(f"      [警告] 无法自动解析投影，回退到默认 EPSG:4547。错误: {e}")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)

    # 执行转换: (Lon, Lat) -> (Projected X, Projected Y)
    proj_x, proj_y = transformer.transform(target_lon, target_lat)
    print(f"      投影坐标 (米): X={proj_x:.2f}, Y={proj_y:.2f}")

    # ================= 4. 像素坐标计算与高度插值 =================
    print("[3/4] 计算像素位置并查询高度...")
    
    # GeoTransform: [top_left_x, w_e_res, rot_0, top_left_y, rot_1, n_s_res]
    x_origin = geotransform[0]
    x_res = geotransform[1]
    y_origin = geotransform[3]
    y_res = geotransform[5]

    # 计算浮点数像素坐标
    # Col = (X - X0) / dX
    # Row = (Y - Y0) / dY
    col_float = (proj_x - x_origin) / x_res
    row_float = (proj_y - y_origin) / y_res

    rows, cols = area.shape

    # 边界检查
    if 0 <= col_float < cols and 0 <= row_float < rows:
        # 使用 scipy.ndimage.map_coordinates 进行双线性插值
        # 注意: map_coordinates 输入顺序为 [Row(Y), Col(X)]
        coords_y = np.array([row_float])
        coords_x = np.array([col_float])
        
        # 调用类中现有的插值方法 (或者直接用 map_coordinates)
        # 此处复用类方法以保持一致性
        height_values = locator.interpolate_along_line(area, coords_x, coords_y, num_points=1)
        dsm_height = height_values[0]

        print("-" * 50)
        print(f"[结果] DSM 高度 (Z): {dsm_height:.4f} 米")
        print("-" * 50)
        print(f"详细信息:")
        print(f"  输入经纬度: {target_lon}, {target_lat}")
        print(f"  像素坐标  : Row={row_float:.2f}, Col={col_float:.2f}")
    else:
        print("-" * 50)
        print(f"[错误] 坐标超出地图范围!")
        print(f"  地图尺寸: {cols}x{rows}")
        print(f"  计算位置: Row={row_float:.1f}, Col={col_float:.1f}")
        print("-" * 50)