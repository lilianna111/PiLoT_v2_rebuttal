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
        image_width_px = 3840/2
        image_height_px = 2160/2
        sensor_width_mm = 3840/2
        sensor_height_mm = 2160/2
        f_mm = 2700/2
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
        cx = 1915.7/2
        cy = 1075.1/2
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

    def predict_center_alt(self, DSM_path, pose, ref_npy_path, geotransform, K, num_sample, object_pixel_coords):
        """
        预测图像中心点的深度，并输出4个关键指标（增加WGS84经纬度输出）
        """
        # 1. 准备数据
        # DSM_npy_path = ref_npy_path
        # geotransform_path = DSM_path.replace(".tif", ".txt")
        
        # if os.path.exists(DSM_npy_path) and os.path.exists(geotransform_path):
        #      area = np.load(DSM_npy_path)
        #      geotransform = np.load(geotransform_path)
        #      area_minZ = np.ma.masked_values(area, -9999).min()
        # else:
        #      area, geotransform, area_minZ = self.dsm2npy(DSM_path, DSM_npy_path, geotransform_path)
        
        area = np.load(ref_npy_path)
        area_minZ = np.ma.masked_values(area, -9999).min()
        
        # 2. 计算变换矩阵 T
        raw_lon, raw_lat, raw_alt = pose[0], pose[1], pose[2]
        raw_roll, raw_pitch, raw_yaw = pose[3], pose[4], pose[5]
        
        # 定义正向变换 (WGS84 -> 投影) 和 反向变换 (投影 -> WGS84)
        proj_crs = "EPSG:4547" # 您之前代码中使用的投影带
        transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
        transformer_back = pyproj.Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)

        x_meter, y_meter = transformer.transform(raw_lon, raw_lat)
        
        coord_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        r_original = R.from_euler('xyz', [raw_pitch, raw_roll, raw_yaw], degrees=True)
        matrix_3x3 = r_original.as_matrix()
        
        T4x4 = np.eye(4)
        T4x4[:3, :3] = matrix_3x3
        T4x4[:3, 3] = [x_meter, y_meter, raw_alt]
        
        # 得到最终的 Camera -> World 矩阵
        T = T4x4 @ coord_transform
        
        R_new = T[:3, :3]
        t_new = T[:3, 3]
        
        
        if object_pixel_coords is None:
            object_pixel_coords = [0, 0]
        
        # 3. 计算预测点 (Ray Casting) -> 得到投影坐标
        x_proj, k_value, sampleHeight = self.caculate_predictXYZ(
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
        P_center = np.array(P_center_list) # [X_proj, Y_proj, Z]
        
        return P_center
    def predict_center_depth(self, DSM_path, pose, gt_depth_val=None, num_sample=10000, object_pixel_coords=None):
        """
        预测图像中心点的深度，并输出4个关键指标（增加WGS84经纬度输出）
        """
        # 1. 准备数据
        DSM_npy_path = DSM_path.replace(".tif", ".npy")
        geotransform_path = DSM_path.replace(".tif", ".txt")
        
        if os.path.exists(DSM_npy_path) and os.path.exists(geotransform_path):
             area = np.load(DSM_npy_path)
             geotransform = np.load(geotransform_path)
             area_minZ = np.ma.masked_values(area, -9999).min()
        else:
             area, geotransform, area_minZ = self.dsm2npy(DSM_path, DSM_npy_path, geotransform_path)
        
        # 2. 计算变换矩阵 T
        raw_lon, raw_lat, raw_alt = pose[0], pose[1], pose[2]
        raw_roll, raw_pitch, raw_yaw = pose[3], pose[4], pose[5]
        
        # 定义正向变换 (WGS84 -> 投影) 和 反向变换 (投影 -> WGS84)
        proj_crs = "EPSG:4547" # 您之前代码中使用的投影带
        transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
        transformer_back = pyproj.Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)

        x_meter, y_meter = transformer.transform(raw_lon, raw_lat)
        
        coord_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        r_original = R.from_euler('xyz', [raw_pitch, raw_roll, raw_yaw], degrees=True)
        matrix_3x3 = r_original.as_matrix()
        
        T4x4 = np.eye(4)
        T4x4[:3, :3] = matrix_3x3
        T4x4[:3, 3] = [x_meter, y_meter, raw_alt]
        
        # 得到最终的 Camera -> World 矩阵
        T = T4x4 @ coord_transform
        
        R_new = T[:3, :3]
        t_new = T[:3, 3]
        
        K = self.get_intrinsic()
        
        if object_pixel_coords is None:
            object_pixel_coords = [1915.7, 1075.1]
        
        # 3. 计算预测点 (Ray Casting) -> 得到投影坐标
        x_proj, k_value, sampleHeight = self.caculate_predictXYZ(
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
        P_center = np.array(P_center_list) # [X_proj, Y_proj, Z]
        
        # --- 新增：将预测点转换为 WGS84 ---
        pred_lon, pred_lat = transformer_back.transform(P_center[0], P_center[1])
        pred_alt = P_center[2]
        
        # 4. 计算预测深度
        camera_point = R_new.T @ (P_center - T[0:3, 3])
        depth = camera_point[2]
        
        # 5. 计算真实中心点 3D Pose (反投影)
        gt_center_str = "未提供真实深度 (None)"
        gt_wgs84_str = "N/A"
        
        if gt_depth_val is not None:
            # 像素 -> 归一化相机坐标
            K_np = np.array(K)
            uv_homo = np.array([object_pixel_coords[0], object_pixel_coords[1], 1.0])
            norm_cam = np.linalg.inv(K_np).dot(uv_homo)
            
            # 归一化 -> 相机坐标系 (x*d, y*d, d)
            P_cam_gt = norm_cam * gt_depth_val
            
            # 相机坐标系 -> 世界投影坐标系
            P_world_gt = R_new.dot(P_cam_gt) + t_new
            gt_center_str = f"[{P_world_gt[0]:.4f}, {P_world_gt[1]:.4f}, {P_world_gt[2]:.4f}]"
            
            # --- 新增：将真实点转换为 WGS84 ---
            gt_lon, gt_lat = transformer_back.transform(P_world_gt[0], P_world_gt[1])
            gt_alt = P_world_gt[2]
            gt_wgs84_str = f"[Lon: {gt_lon:.8f}, Lat: {gt_lat:.8f}, Alt: {gt_alt:.4f}]"

        # ==================== 输出结果 ====================
        print("-" * 60)
        # 1. 无人机 Pose
        print(f"[1] 无人机 Pose:\n    Position: {pose}")
        
        # 2. 预测中心点
        print(f"[2] 预测中心点 3D Pose (DSM):")
        print(f"    (投影坐标): {P_center}")
        print(f"    (WGS84):   [Lon: {pred_lon:.8f}, Lat: {pred_lat:.8f}, Alt: {pred_alt:.4f}]")
        
        # 3. 真实中心点
        print(f"[3] 真实中心点 3D Pose (GT Back-project):")
        print(f"    (投影坐标): {gt_center_str}")
        print(f"    (WGS84):   {gt_wgs84_str}")
        
        # 4. 预测深度值
        print(f"[4] 预测深度值:\n    {depth:.4f} 米")
        print("-" * 60)
        
        return depth
# # ================= 主函数调用 =================
# if __name__ == "__main__":

#     config = {"ray_casting": { }}
#     locator = TargetLocation(config, use_dsm=False)
    
#     DSM_path = "/home/amax/Documents/datasets/DEM/feicuiwan1210/feicuiwandem.tif"
#     npy_path = "/home/amax/Documents/datasets/DEM/feicuiwan1210/feicuiwandem.npy"
#     geotransform_path = "/home/amax/Documents/datasets/DEM/feicuiwan1210/feicuiwandem.txt"
#     area, geotransform, area_minZ = TargetLocation.dsm2npy(DSM_path, npy_path, geotransform_path)
    # def dsm2npy(dsm_path, npy_path, geotransform_path):
    
#     # 你的 Pose
#     pose = [112.991595, 28.290454, 76.4, 0.0, 34.0, 162.1]
    
#     # 【请在这里填入真实深度】(例如: 69.76 或者 None)
#     # 如果不知道真实深度，填 None，第3项就不会算出数值
#     real_depth = 69.76
    
    
    # depth = locator.predict_center_depth(DSM_path, pose, gt_depth_val=real_depth)