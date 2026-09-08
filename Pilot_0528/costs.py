import torch
from typing import Optional, Tuple
from torch import Tensor
import copy
import numpy as np
import pyproj
import os
from osgeo import gdal
from scipy.ndimage import map_coordinates
import time
from scipy.spatial.transform import Rotation as SciRotation
from . import Pose, Camera
from .optimization import J_normalization, skew_symmetric, so3exp_map
from .interpolation import Interpolator
import torch.nn.functional as F
from .transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
@torch.jit.script
def J_project( p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=1e-3)
        J = torch.stack([
            1/z, zero, -x / z**2,
            zero, 1/z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1]+(2, 3))
        return J  # N x 2 x 3
@torch.jit.script
def transform_p3d(body2view_pose_data, p3d):
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)
@torch.jit.script
def project_p3d(camera_data, p3d):
    eps=1e-4

    z = p3d[..., -1]
    valid1 = z > eps
    z = z.clamp(min=eps)
    p2d = p3d[..., :-1] / z.unsqueeze(-1)

    f = camera_data[..., 2:4]
    c = camera_data[..., 4:6]
    p2d = p2d * f.unsqueeze(-2) + c.unsqueeze(-2)

    size = camera_data[..., :2]
    size = size.unsqueeze(-2)
    # valid2 = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
    valid2 = torch.logical_and(p2d >= 0, p2d <= (size - 1))
    valid2 = torch.logical_and(valid2[..., 0], valid2[..., 1])
    valid = torch.logical_and(valid1, valid2)
    return p2d, valid
@torch.jit.script
def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M
class DirectAbsoluteCost2:
    def __init__(self):
        self.iter = 2
        self._ecef_to_wgs84 = None
        self._dsm_cache = {}
        self._wgs84_to_ecef = None
        self.use_gpu_dsm = torch.cuda.is_available()
        # 深度约束（DSM prior）：True=开启，False=关闭
        self.enable_depth_prior = True  # 改为 False 可关闭深度约束
        # 启用角度约束（roll/pitch prior）
        self.enable_angle_prior = True
        # LM 中角度项强度：提高角度先验在迭代中的实际影响
        self.angle_scale_fixed = 0.5
        # 角度梯度相对重投影梯度的目标比例（自适应缩放）
        self.angle_target_ratio = 0.3
        # 自适应缩放的钳制范围，防止跑飞
        self.angle_scale_min = 0.05
        self.angle_scale_max = 20.0
        # 角度残差门控（单位：度）：残差越大，越可能是定义/同步偏差，自动减弱角度项
        self.angle_gate_start_deg = 8.0
        self.angle_gate_end_deg = 25.0
        self.angle_gate_min = 0.2
        # 最终选 pose 时也纳入角度（按有效点数缩放到与重投影可比量级）
        self.angle_selection_weight = 1.0
        self.angle_debug_print = True
        self._angle_prev_mean_deg = None
        # 深度先验强度（与角度先验同样思路：按梯度范数自适应缩放）
        # Follow angle-prior style adaptive scaling for depth.
        self.depth_scale_fixed = 0.5
        self.depth_target_ratio = 0.3
        self.depth_scale_min = 0.05
        self.depth_scale_max = 20.0  # 与 angle_scale_max 一致，否则 raw_scale 被压死、contrib 永远≈0
        # Depth residual gate (meters), same philosophy as angle prior:
        # large mismatch likely indicates outlier/model mismatch, so reduce
        # depth influence smoothly instead of abrupt disable.
        self.depth_gate_start_m = 4.0
        self.depth_gate_end_m = 20.0
        self.depth_gate_min = 0.35
        # Temporal smoothing for adaptive scale to avoid frame-to-frame jumps.
        self.depth_scale_ema_momentum = 0.8
        self._depth_prev_scale_mean = None
        # Depth prior mainly corrects translation/height; keep rotation from reprojection.
        self.depth_translation_only = True
        # Use a dedicated robust-kernel width for depth.
        # Reusing projection truncate(0.1) makes depth weight collapse when residuals are in meter scale.
        self.depth_loss_truncate = 2.0
        # Finite-difference step sizes for depth Jacobian (engineering scale).
        # Too small values cause near-zero Jacobian and make depth prior ineffective.
        self.depth_fd_eps_rot_deg = 0.2
        self.depth_fd_eps_trans = 0.2
        # 最终选 pose 时纳入深度项（按有效点数缩放）
        self.depth_selection_weight = 0.0
        # Debug policy: collect per-call stats and print one summary at the end.
        self.depth_debug_print = True
        self.depth_debug_verbose = False
        self.depth_verify_query_print = False
        self._depth_debug_state = None

    def begin_depth_debug_session(self, tag: str = ""):
        self._depth_debug_state = {
            "tag": tag,
            "calls_total": 0,
            "calls_with_depth": 0,
            "res_abs_mean_m": [],
            "res_abs_p90_m": [],
            "valid_ratio": [],
            "gate_mean": [],
            "hard_reject_ratio": [],
            "effective_ratio": [],
            "scale_mean": [],
            "scale_max": [],
            "grad_ratio_depth_over_proj": [],
            "grad_ratio_depth_over_total": [],
            "inactive_calls": 0,
        }

    def end_depth_debug_session(self, print_summary: bool = True) -> str:
        s = self._depth_debug_state
        if not self.depth_debug_print or s is None:
            return ""
        if s["calls_total"] == 0:
            msg = "[depth_prior_summary] no calls recorded."
        elif s["calls_with_depth"] == 0:
            msg = (
                "[depth_prior_summary] calls={} depth_calls=0 (depth branch not entered)."
                .format(s["calls_total"])
            )
        else:
            inactive_ratio = float(s["inactive_calls"]) / float(s["calls_with_depth"])
            msg = (
                "[depth_prior_summary] tag={} calls={} depth_calls={} | "
                "res|m| mean={:.3f} p90={:.3f} | valid_ratio mean={:.3f} | "
                "gate mean={:.3f} | hard_reject mean={:.3f} | "
                "effective mean={:.3f} inactive_calls={:.1%} | "
                "scale mean={:.3f} max={:.3f} | grad(depth/proj) mean={:.3f} | "
                "contrib(depth/total) mean={:.3f}"
            ).format(
                s["tag"],
                s["calls_total"],
                s["calls_with_depth"],
                float(np.mean(s["res_abs_mean_m"])),
                float(np.mean(s["res_abs_p90_m"])),
                float(np.mean(s["valid_ratio"])),
                float(np.mean(s["gate_mean"])),
                float(np.mean(s["hard_reject_ratio"])),
                float(np.mean(s["effective_ratio"])),
                inactive_ratio,
                float(np.mean(s["scale_mean"])),
                float(np.max(s["scale_max"])),
                float(np.mean(s["grad_ratio_depth_over_proj"])),
                float(np.mean(s["grad_ratio_depth_over_total"])),
            )
            if inactive_ratio > 0.5:
                msg += " [WARN depth mostly inactive]"
            if float(np.mean(s["effective_ratio"])) < 0.9:
                msg += " [WARN depth participation dropped]"
        # if print_summary:
        #     print(msg)
        return msg

    def _poses_to_wgs84(self, pose_data_q: Tensor, render_T_ecef) -> list:
        # Convert candidate poses from local (camera) coordinates to WGS84.
        t_c2w_ecef = (
            render_T_ecef.to(dtype=torch.float64)
            if isinstance(render_T_ecef, torch.Tensor)
            else torch.from_numpy(render_T_ecef).to(device=pose_data_q.device, dtype=torch.float64)
        )
        t_ecef = t_c2w_ecef.detach().cpu().numpy()
        if t_ecef.ndim == 1:
            t_ecef = t_ecef[None, :]
        if self._ecef_to_wgs84 is None:
            self._ecef_to_wgs84 = pyproj.Transformer.from_crs(
                {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
                "EPSG:4326",
                always_xy=True,
            )
        transformer = self._ecef_to_wgs84
        lon_all, lat_all, alt_all = transformer.transform(
            t_ecef[:, 0], t_ecef[:, 1], t_ecef[:, 2], radians=False
        )
        if np.ndim(lon_all) == 0:
            lon_all = np.array([lon_all])
            lat_all = np.array([lat_all])
            alt_all = np.array([alt_all])

        wgs84_poses = []
        num_init_pose = pose_data_q.shape[1]
        for i in range(num_init_pose):
            lon = float(lon_all[i] if len(lon_all) > 1 else lon_all[0])
            lat = float(lat_all[i] if len(lat_all) > 1 else lat_all[0])
            alt = float(alt_all[i] if len(alt_all) > 1 else alt_all[0])

            lat_rad, lon_rad = np.radians(lat), np.radians(lon)
            up = np.array(
                [
                    np.cos(lon_rad) * np.cos(lat_rad),
                    np.sin(lon_rad) * np.cos(lat_rad),
                    np.sin(lat_rad),
                ]
            )
            east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
            north = np.cross(up, east)
            rot_enu_in_ecef = np.column_stack([east, north, up])

            R_w2c_q = pose_data_q[0, i, :9].view(3, 3).to(dtype=torch.float64)
            R_c2w_prep = R_w2c_q.transpose(-1, -2)
            R_c2w = R_c2w_prep.clone()
            R_c2w[:, 1] *= -1
            R_c2w[:, 2] *= -1

            R_c2w_np = R_c2w.detach().cpu().numpy()
            R_pose_in_enu = rot_enu_in_ecef.T @ R_c2w_np
            euler_angles = SciRotation.from_matrix(R_pose_in_enu).as_euler("xyz", degrees=True)
            pitch, roll, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
            wgs84_poses.append([lon, lat, alt, roll, pitch, yaw])

        return wgs84_poses

    def _pose_translations_ecef(self, pose_data_q: Tensor, origin, dd) -> Tensor:
        pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
        R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
        t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
        t_c2w_local = -torch.bmm(R_w2c.transpose(1, 2), t_w2c.unsqueeze(-1)).squeeze(-1)
        origin_t = torch.as_tensor(origin, device=pose_data_q.device, dtype=torch.float64)
        dd_t = torch.as_tensor(dd, device=pose_data_q.device, dtype=torch.float64)
        return t_c2w_local + origin_t + dd_t

    def _center_points_from_wgs84_poses(
        self,
        wgs84_poses: list,
        depth,
        K: Optional[list] = None,
        mul: Optional[float] = None,
    ) -> list:
        # Estimate the 3D point at image center for each WGS84 pose and depth.
        if len(wgs84_poses) == 0:
            return []
        if K is None:
            K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
        _, _, fx, fy, cx, cy = K

        if torch.is_tensor(depth):
            depth = float(depth.detach().cpu().reshape(-1)[0].item())
        elif isinstance(depth, np.ndarray):
            depth = float(depth.reshape(-1)[0])
        else:
            depth = float(depth)
        if mul is not None:
            depth = depth * float(mul)
        poses_np = np.asarray(wgs84_poses, dtype=np.float64)
        lon = poses_np[:, 0]
        lat = poses_np[:, 1]
        alt = poses_np[:, 2]
        roll = poses_np[:, 3]
        pitch = poses_np[:, 4]
        yaw = poses_np[:, 5]

        if self._wgs84_to_ecef is None:
            self._wgs84_to_ecef = pyproj.Transformer.from_crs(
                "EPSG:4326",
                {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
                always_xy=True,
            )
        x_ecef, y_ecef, z_ecef = self._wgs84_to_ecef.transform(lon, lat, alt, radians=False)
        t_c2w = np.stack([x_ecef, y_ecef, z_ecef], axis=1)

        euler_angles = np.stack([pitch, roll, yaw], axis=1)
        rot_pose_in_enu = SciRotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        up = np.stack(
            [
                np.cos(lon_rad) * np.cos(lat_rad),
                np.sin(lon_rad) * np.cos(lat_rad),
                np.sin(lat_rad),
            ],
            axis=1,
        )
        east = np.stack(
            [
                -np.sin(lon_rad),
                np.cos(lon_rad),
                np.zeros_like(lon_rad),
            ],
            axis=1,
        )
        north = np.cross(up, east)
        rot_enu_in_ecef = np.stack([east, north, up], axis=2)

        R_c2w = np.einsum("nij,njk->nik", rot_enu_in_ecef, rot_pose_in_enu)
        R_c2w[:, :, 1] *= -1
        R_c2w[:, :, 2] *= -1

        K_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_inv = np.linalg.inv(K_matrix)
        pixel_center = np.array([cx, cy, 1.0])
        cam_vec = K_inv @ (depth * pixel_center)

        point_3d_ecef = np.einsum("nij,j->ni", R_c2w, cam_vec) + t_c2w
        world_points = ECEF_to_WGS84(point_3d_ecef)
        return world_points.tolist()

    def _center_points_from_pose_ecef(
        self,
        pose_data_q: Tensor,
        depth,
        K: Optional[list],
        origin,
        dd,
        mul: Optional[float] = None,
    ) -> list:
        # Directly compute center-ray 3D point from pose in ECEF without
        # Euler/WGS84 round-trips to avoid convention errors.
        if K is None:
            K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
        _, _, fx, fy, cx, cy = K

        if torch.is_tensor(depth):
            depth = float(depth.detach().cpu().reshape(-1)[0].item())
        elif isinstance(depth, np.ndarray):
            depth = float(depth.reshape(-1)[0])
        else:
            depth = float(depth)
        if mul is not None:
            depth = depth * float(mul)

        pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
        R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
        t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
        R_c2w = R_w2c.transpose(1, 2)
        C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)

        origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
        dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
        C_ecef = C_local + origin_t + dd_t

        K_matrix = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=pose.device,
            dtype=torch.float64,
        )
        K_inv = torch.inverse(K_matrix)
        pixel_center = torch.tensor([cx, cy, 1.0], device=pose.device, dtype=torch.float64)
        cam_vec = K_inv @ (depth * pixel_center)
        cam_vec = cam_vec.view(1, 3, 1).repeat(R_c2w.shape[0], 1, 1)
        ray_world = torch.bmm(R_c2w, cam_vec).squeeze(-1)
        point_ecef = C_ecef + ray_world

        world_points = ECEF_to_WGS84(point_ecef.detach().cpu().numpy())
        return world_points.tolist()

    def _query_dsm_heights(
        self,
        wgs84_points: list,
        dsm_path: str,
        proj_crs: str = "EPSG:4547",
        use_gpu: bool = False,
    ) -> list:
        # Query DSM heights by lon/lat from WGS84 points using bilinear interpolation.
        cache = self._dsm_cache.get(dsm_path)
        if cache is None or cache.get("proj_crs") != proj_crs:
            dsm_npy_path = dsm_path.replace(".tif", ".npy")
            geotransform_path = dsm_path.replace(".tif", ".txt")
            if os.path.exists(dsm_npy_path) and os.path.exists(geotransform_path):
                area = np.load(dsm_npy_path)
                geotransform = np.load(geotransform_path)
            else:
                dsm_map = gdal.Open(dsm_path)
                if dsm_map is None:
                    raise FileNotFoundError(f"Unable to open DSM file: {dsm_path}")
                band = dsm_map.GetRasterBand(1)
                geotransform = dsm_map.GetGeoTransform()
                area = band.ReadAsArray()
                del dsm_map
                np.save(dsm_npy_path, area)
                np.save(geotransform_path, geotransform)
            rows, cols = area.shape
            x_origin = geotransform[0]
            x_pixel_size = geotransform[1]
            y_origin = geotransform[3]
            y_pixel_size = geotransform[5]
            min_x_proj = x_origin
            max_x_proj = x_origin + cols * x_pixel_size
            max_y_proj = y_origin
            min_y_proj = y_origin + rows * y_pixel_size

            transformer_back = pyproj.Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
            corner_x = np.array([min_x_proj, max_x_proj, min_x_proj, max_x_proj])
            corner_y = np.array([min_y_proj, min_y_proj, max_y_proj, max_y_proj])
            corner_lon, corner_lat = transformer_back.transform(corner_x, corner_y)
            lon_min, lon_max = float(np.min(corner_lon)), float(np.max(corner_lon))
            lat_min, lat_max = float(np.min(corner_lat)), float(np.max(corner_lat))

            cache = {
                "area": area,
                "geotransform": geotransform,
                "proj_crs": proj_crs,
                "rows": rows,
                "cols": cols,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "transformer_wgs84_to_proj": pyproj.Transformer.from_crs(
                    "EPSG:4326", proj_crs, always_xy=True
                ),
            }
            self._dsm_cache[dsm_path] = cache

        if use_gpu and torch.cuda.is_available():
            gpu_cache = cache.get("gpu")
            if gpu_cache is None:
                map_tensor = (
                    torch.from_numpy(cache["area"])
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to("cuda")
                )
                gpu_cache = {
                    "map_tensor": map_tensor,
                }
                cache["gpu"] = gpu_cache

            # 先将WGS84转换为投影坐标系（EPSG:4547）
            transformer = cache["transformer_wgs84_to_proj"]
            geotransform = cache["geotransform"]
            x_origin = geotransform[0]
            x_pixel_size = geotransform[1]
            y_origin = geotransform[3]
            y_pixel_size = geotransform[5]
            rows, cols = cache["rows"], cache["cols"]

            points_array = np.asarray(wgs84_points, dtype=np.float64)
            lons = points_array[:, 0]
            lats = points_array[:, 1]
            # WGS84 -> EPSG:4547投影坐标
            x_proj, y_proj = transformer.transform(lons, lats)
            
            # 投影坐标 -> 像素坐标
            col_float = (x_proj - x_origin) / x_pixel_size
            row_float = (y_proj - y_origin) / y_pixel_size
            
            # 转换为GPU张量
            col_tensor = torch.from_numpy(col_float).to("cuda", dtype=torch.float32)
            row_tensor = torch.from_numpy(row_float).to("cuda", dtype=torch.float32)
            
            # 检查有效性
            valid = (col_tensor >= 0) & (col_tensor < cols) & (row_tensor >= 0) & (row_tensor < rows)
            
            # 归一化到[-1, 1]用于grid_sample
            grid_u = (col_tensor / (cols - 1)) * 2.0 - 1.0
            grid_v = (row_tensor / (rows - 1)) * 2.0 - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).view(1, -1, 1, 2)
            
            # 采样
            sample = F.grid_sample(gpu_cache["map_tensor"], grid, align_corners=True, padding_mode="border")
            heights = sample.view(-1)
            heights = heights.masked_fill(~valid, float("nan"))
            return heights.detach().cpu().numpy().tolist()

        transformer = cache["transformer_wgs84_to_proj"]

        geotransform = cache["geotransform"]
        area = cache["area"]
        x_origin = geotransform[0]
        x_pixel_size = geotransform[1]
        y_origin = geotransform[3]
        y_pixel_size = geotransform[5]
        rows, cols = cache["rows"], cache["cols"]

        points_array = np.asarray(wgs84_points, dtype=np.float64)
        lons = points_array[:, 0]
        lats = points_array[:, 1]
        x_proj, y_proj = transformer.transform(lons, lats)
        col_float = (x_proj - x_origin) / x_pixel_size
        row_float = (y_proj - y_origin) / y_pixel_size

        heights = np.full((len(wgs84_points),), np.nan, dtype=np.float64)
        valid = (col_float >= 0) & (col_float < cols) & (row_float >= 0) & (row_float < rows)
        if np.any(valid):
            coords_x = col_float[valid]
            coords_y = row_float[valid]
            height_array = map_coordinates(area, [coords_y, coords_x], order=1)
            heights[valid] = height_array

        return heights.tolist()
    def bilinear_grid_sample(self, im, grid, align_corners=False):
        n, c, h, w = im.shape
        gn, gh, gw, _ = grid.shape
        assert n == gn
    
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]
    
        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2
    
        x = x.view(n, -1)
        y = y.view(n, -1)
    
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1
    
        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)
    
        # Apply default for grid_sample function zero padding
        im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0.0)
        padded_h = h + 2
        padded_w = w + 2
    
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1
    
        # Clip coordinates to padded image size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
        x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
        x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
        x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
        y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
        y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
        y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
        y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)
    
        im_padded = im_padded.view(n, c, -1)
    
        x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    
        Ia = torch.gather(im_padded, 2, x0_y0)
        Ib = torch.gather(im_padded, 2, x0_y1)
        Ic = torch.gather(im_padded, 2, x1_y0)
        Id = torch.gather(im_padded, 2, x1_y1)
    
        return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)
    def interpolate_feature_map(
                            self,
                            feature: torch.Tensor,
                            p2d: torch.Tensor,
                            return_gradients: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interpolation_pad = 4
        b, c, h, w = feature.shape
        scale = torch.tensor([w-1, h-1]).to(p2d)
        pts = (p2d / scale) * 2  - 1
        pts = pts.clamp(min=-2.0, max=2.0)
        # fp = torch.nn.functional.grid_sample(feature, pts[:, None], mode='bilinear', align_corners=True)
        fp = self.bilinear_grid_sample(feature, pts[:, None], align_corners=True)
        fp = fp.reshape(b, c, -1).transpose(-1, -2)
        
        image_size_ = torch.tensor([w-interpolation_pad-1, h-interpolation_pad-1]).to(pts)
        valid0 = torch.logical_and(p2d >= interpolation_pad, p2d <= image_size_)
        valid = torch.logical_and(valid0[..., 0], valid0[..., 1])
        # valid = torch.all((p2d >= interpolation_pad) & (p2d <= image_size_), -1)
        if return_gradients:
            dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
            dx, dy = dxdy.chunk(2, dim=0)
            pts_d = torch.cat([pts-dx, pts+dx, pts-dy, pts+dy], 1)
            # tensor_d = torch.nn.functional.grid_sample(
            #         feature, pts_d[:, None], mode='bilinear', align_corners=True)
            tensor_d = self.bilinear_grid_sample(
                    feature, pts_d[:, None], align_corners=True)
            tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
            tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
            gradients = torch.stack([
                (tensor_x1 - tensor_x0)/2, (tensor_y1 - tensor_y0)/2], dim=-1)
        else:
            gradients = torch.zeros(b, pts.shape[1], c, 2).to(feature)

        return fp, valid, gradients

    def loss_fn1(
            self,
            cost: Tensor,
            alpha: float = 0.0,
            truncate: float = 0.1,
            eps: float = 1e-7,
            derivatives: bool = True
        ) -> Tuple[Tensor, Tensor, Tensor]:  
        # x = cost / (truncate**2)
        # alpha = cost.new_tensor(alpha)[None]
        x = cost / (truncate ** 2)
        alpha = torch.full((1,), alpha, dtype=cost.dtype, device=cost.device)

        loss_two = x
        # loss_zero = 2 * torch.log1p(torch.clamp(0.5*x, max=33e37))
        loss_zero = 2 * torch.log(torch.clamp(0.5*x, max=33e37)+1)

        # The loss when not in one of the above special cases.
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.abs(alpha - 2.).clamp(min=eps)

        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        alpha_safe = alpha_safe * torch.abs(alpha).clamp(min=eps)

        loss_otherwise = 2 * (beta_safe / alpha_safe) * (
            torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == 0, loss_zero,
            torch.where(alpha == 2, loss_two, loss_otherwise))
        dummy = torch.zeros_like(x)

        loss_two_d1 = torch.ones_like(x)
        loss_zero_d1 = 2 / (x + 2)
        loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)
        loss_d1 = torch.where(
            alpha == 0, loss_zero_d1,
            torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

        loss, loss_d1, loss_d2 = loss, loss_d1, dummy

        
        return loss*(truncate**2), loss_d1, loss_d2/(truncate**2)

    def residual_jacobian_batch_quat(
        self, pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D: Tensor,
        c_ref, c_query, p2d_r, visible_r, gt_depth=None, gt_roll=None, gt_pitch=None,
        dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None,
        w_depth=None, w_angle=None):
        """
        对多候选查询位姿 batch 计算重投影残差、深度/角度先验的残差与雅可比，
        并返回用于高斯-牛顿的负梯度、Hessian、鲁棒权重等。

        融合深度/角度时使用可学习标量 w_depth、w_angle（由调用方传入，如 nn.Parameter）。
        未传时视为 1.0。深度残差/雅可比用固定常数归一化（不参与训练）。
        """
        # -------- 参数与参考帧准备 --------
        num_init_pose = pose_data_q.shape[1]
        cam_K = None
        if cam_data_q is not None and torch.is_tensor(cam_data_q) and cam_data_q.numel() >= 6:
            cam_K = cam_data_q[0, 0, :6].detach().cpu().tolist()

        # -------- Step 1: 参考帧特征与有效 mask --------
        # 在参考帧 2D 点 p2d_r 上插值特征，得到 [1, N, C]
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)
        valid_ref_mask = (valid_r & visible_r).to(torch.float32)  # [1, N]
        mask = valid_ref_mask.transpose(1, 0)  # [N, 1]

        fp_r = fp_r[0] * mask   # [N, C]，无效点置 0
        p3D = p3D[0] * mask     # [N, 3]
        fp_r = fp_r.repeat(1, num_init_pose, 1, 1)
        p3D = p3D.repeat(1, num_init_pose, 1, 1)
        fp_r = torch.nn.functional.normalize(fp_r, dim=-1)

        # -------- Step 2: 用候选位姿把参考 3D 点变到查询相机并投影 --------
        p3d_q = transform_p3d(pose_data_q, p3D)   # 体到相机的 3D
        p2d_q, visible_q = project_p3d(cam_data_q, p3d_q)
        p2d_q = p2d_q * mask.T.view(1, 1, -1, 1)  # 无效点投影坐标置 0

        # -------- Step 3: 在查询帧上按 p2d_q 插值特征与图像梯度 --------
        fp_q, valid_q, J_f = self.interpolate_feature_map(
            f_q, p2d_q.reshape(1, -1, 2), return_gradients=True
        )
        fp_q = fp_q.view(1, num_init_pose, -1, fp_q.shape[-1])
        valid_q = valid_q.view(1, num_init_pose, -1)
        J_f = J_f.view(1, num_init_pose, -1, J_f.shape[-2], J_f.shape[-1])
        visible_q = visible_q.view(1, num_init_pose, -1)
        res = fp_q - fp_r  # 重投影残差 [1, num_init_pose, N, C]

        # -------- Step 4: 重投影残差对位姿的雅可比 J = J_f @ J_p2d_p3d @ J_p3d_pose --------
        R = pose_data_q[..., :9].view(1, -1, 3, 3)
        J_p3d_rot = R[:, :, None].matmul(-skew_symmetric(p3D))
        J_p3d_tran = R[:, :, None].expand(-1, -1, J_p3d_rot.shape[2], -1, -1)
        J_p3d_pose = torch.cat((J_p3d_rot, J_p3d_tran), dim=-1)

        fx, fy = cam_data_q[:, :, 2], cam_data_q[:, :, 3]
        zero = torch.zeros_like(fx)
        f_diag_embed = torch.stack([fx, zero, zero, fy], dim=-1).reshape(1, -1, 1, 2, 2)
        J_p2d_p3d = f_diag_embed @ J_project(p3d_q)
        J = J_f @ J_p2d_p3d @ J_p3d_pose  # [1, num_init_pose, N, C, 6]，特征对 6DoF 的雅可比

        # =========深度约束=========
        # === Step 6: Soft depth + angle priors (adaptive scaling) ===
        if (
            render_T_ecef is not None
            and gt_depth is not None
            and dsm_path is not None
        ):
            # === 验证查询帧位姿（第一个候选）===
            if self.depth_verify_query_print and self.iter <= 3:
                # 获取真实深度值（米），不需要乘 mul
                if torch.is_tensor(gt_depth):
                    depth_real = float(gt_depth.detach().cpu().item())
                else:
                    depth_real = float(gt_depth)
                
                # print(f"\n[验证查询帧] pose_data_q[0,0], depth={depth_real:.2f}m (不乘mul)")
                #
                # 提取第一个候选位姿
                pose_q_first = pose_data_q[:, 0:1, :]  # [1, 1, 12]
                #
                # 转换查询帧位姿到WGS84
                if origin is not None and dd is not None:
                    t_c2w_ecef_q = self._pose_translations_ecef(pose_q_first, origin, dd)
                    wgs84_poses_q = self._poses_to_wgs84(pose_q_first, t_c2w_ecef_q)
                else:
                    wgs84_poses_q = self._poses_to_wgs84(pose_q_first, render_T_ecef)
                #
                lon_q, lat_q, alt_q, roll_q, pitch_q, yaw_q = wgs84_poses_q[0]
                # print(f"  查询帧WGS84: lon={lon_q:.6f}, lat={lat_q:.6f}, alt={alt_q:.2f}m")
                # print(f"  查询帧姿态: roll={roll_q:.4f}°, pitch={pitch_q:.4f}°, yaw={yaw_q:.2f}°")
                #
                # 计算中心点3D坐标，传入真实深度（不乘mul）
                center_q = self._center_points_from_wgs84_poses(
                    wgs84_poses_q, depth_real, K=cam_K
                )
                #
                lon_c, lat_c, z_c = center_q[0]
                # print(f"  中心点3D: lon={lon_c:.6f}, lat={lat_c:.6f}, z={z_c:.2f}m")
                #
                # 查询DSM高度
                dsm_q = self._query_dsm_heights(center_q, dsm_path, use_gpu=self.use_gpu_dsm)
                # print(f"  DSM高度: {dsm_q[0]:.2f}m")
                # print(f"  中心点z - DSM = {z_c - dsm_q[0]:.2f}m (应该接近0)")
                # print(f"[验证结束]\n")

        # === Step 5: 计算（可选的）先验约束（当前启用深度，角度关闭） ===
        depth_res = None
        J_depth_tensor = None
        valid_depth = None
        depth_weight = 1.0
        # 深度残差/雅可比归一化到与投影同量级，便于共享鲁棒核与自适应缩放。
        DEPTH_NORM = 10.0

        if (
            self.enable_depth_prior
            and render_T_ecef is not None
            and gt_depth is not None
            and dsm_path is not None
        ):
            if torch.is_tensor(gt_depth):
                depth_real = float(gt_depth.detach().cpu().reshape(-1)[0].item())
            elif isinstance(gt_depth, np.ndarray):
                depth_real = float(gt_depth.reshape(-1)[0])
            else:
                depth_real = float(gt_depth)

            if cam_K is None:
                cam_K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
            _, _, fx, fy, cx, cy = cam_K

            pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
            R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
            t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
            R_c2w = R_w2c.transpose(1, 2)

            C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)
            if origin is not None and dd is not None:
                origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
                dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
                C_ecef = C_local + origin_t + dd_t
            else:
                C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)

            if origin is not None and dd is not None:
                center_points = self._center_points_from_pose_ecef(
                    pose_data_q, depth_real, K=cam_K, origin=origin, dd=dd, mul=None
                )
            else:
                # Keep depth geometry fully consistent with pose update:
                # compute center-ray 3D point directly in ECEF from pose,
                # instead of WGS84/Euler round-trip conversion.
                center_points = self._center_points_from_pose_ecef(
                    pose_data_q,
                    depth_real,
                    K=cam_K,
                    origin=render_T_ecef,
                    dd=np.zeros(3, dtype=np.float64),
                    mul=None,
                )

            dsm_heights = self._query_dsm_heights(
                center_points, dsm_path, use_gpu=self.use_gpu_dsm
            )
            center_points_np = np.asarray(center_points, dtype=np.float64)
            dsm_heights_np = np.asarray(dsm_heights, dtype=np.float64)

            base_residuals = center_points_np[:, 2] - dsm_heights_np
            valid_mask = ~np.isnan(base_residuals)
            base_residuals = np.nan_to_num(base_residuals, nan=0.0)

            # Numerical Jacobian for depth residual w.r.t. the SAME pose update rule
            # used by optimizer: T <- T @ T_delta.
            # This avoids convention/sign mismatch from closed-form derivatives.
            base_residuals_raw = base_residuals.copy()
            eps_fd_rot = np.deg2rad(float(self.depth_fd_eps_rot_deg))
            eps_fd_trans = float(self.depth_fd_eps_trans)
            J_depth = np.zeros((num_init_pose, 6), dtype=np.float64)

            R_base = R_w2c
            t_base = t_w2c

            def _depth_residual_for_pose(pose_in: Tensor) -> np.ndarray:
                if origin is not None and dd is not None:
                    center_pts = self._center_points_from_pose_ecef(
                        pose_in, depth_real, K=cam_K, origin=origin, dd=dd, mul=None
                    )
                else:
                    center_pts = self._center_points_from_pose_ecef(
                        pose_in,
                        depth_real,
                        K=cam_K,
                        origin=render_T_ecef,
                        dd=np.zeros(3, dtype=np.float64),
                        mul=None,
                    )
                dsm_k = self._query_dsm_heights(
                    center_pts, dsm_path, use_gpu=self.use_gpu_dsm
                )
                center_pts_np = np.asarray(center_pts, dtype=np.float64)
                dsm_k_np = np.asarray(dsm_k, dtype=np.float64)
                r_k = center_pts_np[:, 2] - dsm_k_np
                return r_k

            for k in range(6):
                dR = torch.eye(3, device=pose.device, dtype=torch.float64).unsqueeze(0).repeat(
                    num_init_pose, 1, 1
                )
                dt = torch.zeros(num_init_pose, 3, device=pose.device, dtype=torch.float64)
                if k < 3:
                    dvec = torch.zeros(num_init_pose, 3, device=pose.device, dtype=torch.float64)
                    dvec[:, k] = eps_fd_rot
                    dR = so3exp_map(dvec)
                    eps_used = eps_fd_rot
                else:
                    dt[:, k - 3] = eps_fd_trans
                    eps_used = eps_fd_trans

                # Right-multiplied increment in camera frame:
                # R' = R @ dR, t' = R @ dt + t
                R_pert = torch.bmm(R_base, dR)
                t_pert = torch.bmm(R_base, dt.unsqueeze(-1)).squeeze(-1) + t_base
                pose_pert = torch.cat([R_pert.reshape(num_init_pose, -1), t_pert], dim=-1).unsqueeze(0)

                r_pert = _depth_residual_for_pose(pose_pert)
                valid_pert = ~np.isnan(r_pert)
                valid_both = valid_mask & valid_pert
                col = np.zeros((num_init_pose,), dtype=np.float64)
                if np.any(valid_both):
                    col[valid_both] = (
                        (r_pert[valid_both] - base_residuals_raw[valid_both]) / eps_used
                    )
                J_depth[:, k] = col

            J_depth = np.nan_to_num(J_depth, nan=0.0)
            if self.depth_translation_only:
                # Keep depth prior from directly driving rotation, which can cause yaw runaway.
                J_depth[:, :3] = 0.0

            base_residuals = base_residuals / DEPTH_NORM
            J_depth = J_depth / DEPTH_NORM

            depth_res = torch.from_numpy(base_residuals).to(device=J.device, dtype=J.dtype).view(
                1, num_init_pose, 1
            )
            J_depth_tensor = torch.from_numpy(J_depth).to(device=J.device, dtype=J.dtype).view(
                1, num_init_pose, 1, 6
            )
            valid_depth = torch.from_numpy(valid_mask.astype(np.float32)).to(
                device=J.device, dtype=J.dtype
            ).view(1, num_init_pose, 1)

            # if self.depth_debug_verbose and self.iter <= 3:
            #     r_m = base_residuals * DEPTH_NORM
            #     print(
            #         "[depth_prior] res_m min={:.3f} mean={:.3f} max={:.3f} valid={}/{}".format(
            #             float(np.min(r_m)),
            #             float(np.mean(np.abs(r_m))),
            #             float(np.max(r_m)),
            #             int(valid_mask.sum()),
            #             int(valid_mask.shape[0]),
            #         )
            #     )

        # 角度约束与深度约束可同时启用（enable_angle_prior=True 且 enable_depth_prior=True）。
        angle_res = None
        J_angle_tensor = None
        angle_weight = 1.0

        # === Step 6: Angle prior（当 enable_angle_prior 且传入 render_T_ecef、gt_roll/gt_pitch 时生效）===
        #
        # 角度 prior 的目标：
        # - 使用外部提供的 gt_roll / gt_pitch（单位：度）作为参考，约束当前候选位姿的 roll/pitch
        # - roll/pitch 的定义来自“相机姿态在局部 ENU 坐标系中的欧拉角（xyz/Tait-Bryan）”
        # - 该 prior 只对旋转自由度起作用，所以它的 Jacobian 只填前 3 维（旋转），平移导数为 0
        if (
            self.enable_angle_prior
            and render_T_ecef is not None
            and (gt_roll is not None or gt_pitch is not None)
        ):
            # === 角度约束：roll 和 pitch 使用 gt_roll 和 gt_pitch（解析法）===
            # （若未进入此分支：检查 render_T_ecef 是否传入、gt_roll/gt_pitch 是否有值）
            # 解析法计算 d(roll)/d(ω_c) 和 d(pitch)/d(ω_c)，其中 ω_c 是“相机坐标系下”的旋转增量（李代数）。
            # 我们不用数值差分（finite difference），而是显式写出链式法则：
            #
            #   ω_c  ->  R_w2c  ->  R_c2w  ->  R_pose_in_enu  ->  (roll, pitch)
            #
            # 好处：稳定、可控、不会引入差分步长超参。
            
            # 提取位姿信息
            pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
            R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)  # [num_init_pose, 3, 3]
            
            # 计算相机中心在 ECEF 坐标系中的位置（用于确定局部 ENU 的朝向）。
            # 这里 pose 内部的平移是 t_w2c：世界点 p_w 变换到相机点 p_c 的平移项：
            #   p_c = R_w2c * p_w + t_w2c
            # 对应相机中心（世界坐标）：
            #   C_w = -R_c2w * t_w2c, 其中 R_c2w = R_w2c^T
            t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
            R_c2w = R_w2c.transpose(-1, -2)  # [num_init_pose, 3, 3]
            C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # [num_init_pose, 3]
            
            if origin is not None and dd is not None:
                origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
                dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
                C_ecef = C_local + origin_t + dd_t
            else:
                C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
            
            # 转换到 WGS84 获取经纬度（只用于构建 ENU 基底）。
            # 注意：这里通过 pyproj 把 ECEF -> (lon, lat)，再由 (lon, lat) 构建局部 ENU 的 east/north/up 三个单位向量。
            C_ecef_np = C_ecef.detach().cpu().numpy()
            if C_ecef_np.ndim == 1:
                C_ecef_np = C_ecef_np[None, :]
            
            if self._ecef_to_wgs84 is None:
                self._ecef_to_wgs84 = pyproj.Transformer.from_crs(
                    {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
                    "EPSG:4326",
                    always_xy=True,
                )
            transformer = self._ecef_to_wgs84
            lon_all, lat_all, _ = transformer.transform(
                C_ecef_np[:, 0], C_ecef_np[:, 1], C_ecef_np[:, 2], radians=False
            )
            if np.ndim(lon_all) == 0:
                lon_all = np.array([lon_all])
                lat_all = np.array([lat_all])
            
            # 计算 ENU->ECEF 的旋转矩阵（对每个位姿一套 ENU）。
            # 约定：rot_enu_in_ecef 的列向量依次为 [east, north, up]（都在 ECEF 表达）
            # 则 ECEF 向量 v_ecef 转到 ENU：v_enu = rot_enu_in_ecef^T * v_ecef
            lat_rad = np.radians(lat_all)  # 纬度
            lon_rad = np.radians(lon_all)  # 经度
            lat_rad_t = torch.from_numpy(lat_rad).to(device=pose.device, dtype=torch.float64)
            lon_rad_t = torch.from_numpy(lon_rad).to(device=pose.device, dtype=torch.float64)
            
            # ENU到ECEF的旋转矩阵
            up = torch.stack([
                torch.cos(lon_rad_t) * torch.cos(lat_rad_t),
                torch.sin(lon_rad_t) * torch.cos(lat_rad_t),
                torch.sin(lat_rad_t)
            ], dim=-1)  # [num_init_pose, 3]
            east = torch.stack([
                -torch.sin(lon_rad_t),
                torch.cos(lon_rad_t),
                torch.zeros_like(lon_rad_t)
            ], dim=-1)  # [num_init_pose, 3]
            north = torch.cross(up, east, dim=-1)  # [num_init_pose, 3]
            rot_enu_in_ecef = torch.stack([east, north, up], dim=-1)  # [num_init_pose, 3, 3]
            
            # 计算姿态在 ENU 下的旋转矩阵：
            # 先得到相机->世界（ECEF）的旋转 R_c2w，然后左乘 rot_enu_in_ecef^T 把“世界基底”换成 ENU 基底：
            #   R_pose_in_enu = rot_enu_in_ecef^T * R_c2w_prep
            #
            # 这里对 R_c2w 的第 2/3 列取反（y,z）是为了匹配本工程里“相机坐标系与 ENU/世界坐标系”的轴向约定。
            # 这属于坐标系约定修正，不影响下面链式法则推导的形式，但会影响 roll/pitch 的符号与数值。
            R_c2w_prep = R_c2w.clone()
            R_c2w_prep[:, :, 1] *= -1
            R_c2w_prep[:, :, 2] *= -1
            
            # 计算 R_pose_in_enu = rot_enu_in_ecef^T @ R_c2w
            rot_enu_in_ecef_T = rot_enu_in_ecef.transpose(-1, -2)  # [num_init_pose, 3, 3]
            R_pose_in_enu = torch.bmm(rot_enu_in_ecef_T, R_c2w_prep)  # [num_init_pose, 3, 3]
            
            # 从 R_pose_in_enu 提取 roll / pitch（与下方雅可比一致：roll=asin(R[2,0]), pitch_euler=atan2(-R[2,1],R[2,2])）。
            # 项目约定：0° = 纯俯视，90° = 水平向前。atan2(-R[2,1], R[2,2]) 与 GT 符号相反，故取反。
            R_pose_20 = R_pose_in_enu[:, 2, 0]  # R[2,0]
            R_pose_21 = R_pose_in_enu[:, 2, 1]  # R[2,1]
            R_pose_22 = R_pose_in_enu[:, 2, 2]  # R[2,2]
            roll_base = torch.asin(R_pose_20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))  # [num_init_pose]
            pitch_euler_from_R = torch.atan2(-R_pose_21, R_pose_22)  # [num_init_pose]
            pitch_base = -pitch_euler_from_R  # 项目定义：0°=俯视，90°=水平向前（与 GT 符号一致）
            
            # 角度 Jacobian 改为数值差分，确保与当前优化更新方式 T = T @ T_delta 一致，
            # 避免解析推导在坐标系/右乘扰动约定下出现符号或方向偏差导致累计漂移。
            eps_fd = 1e-4

            def _extract_roll_pitch_from_R_pose(R_pose_enu):
                r20 = R_pose_enu[:, 2, 0]
                r21 = R_pose_enu[:, 2, 1]
                r22 = R_pose_enu[:, 2, 2]
                roll = torch.asin(r20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
                pitch_euler = torch.atan2(-r21, r22)
                pitch = -pitch_euler
                return roll, pitch

            J_roll_cols = []
            J_pitch_cols = []
            base_roll, base_pitch = _extract_roll_pitch_from_R_pose(R_pose_in_enu)
            for k in range(3):
                dvec = torch.zeros(num_init_pose, 3, device=pose.device, dtype=torch.float64)
                dvec[:, k] = eps_fd
                dR = so3exp_map(dvec)  # [num_init_pose, 3, 3]
                R_w2c_pert = torch.bmm(R_w2c, dR)  # 与 T = T @ T_delta 的右乘增量一致
                R_c2w_pert = R_w2c_pert.transpose(-1, -2)
                R_c2w_pert_prep = R_c2w_pert.clone()
                R_c2w_pert_prep[:, :, 1] *= -1
                R_c2w_pert_prep[:, :, 2] *= -1
                R_pose_pert = torch.bmm(rot_enu_in_ecef_T, R_c2w_pert_prep)
                roll_pert, pitch_pert = _extract_roll_pitch_from_R_pose(R_pose_pert)

                # 使用 wrap 后差值，避免跨越 ±pi 时数值跳变
                droll = torch.remainder(roll_pert - base_roll + np.pi, 2.0 * np.pi) - np.pi
                dpitch = torch.remainder(pitch_pert - base_pitch + np.pi, 2.0 * np.pi) - np.pi
                J_roll_cols.append(droll / eps_fd)
                J_pitch_cols.append(dpitch / eps_fd)

            J_roll = torch.stack(J_roll_cols, dim=-1)   # [num_init_pose, 3]
            J_pitch = torch.stack(J_pitch_cols, dim=-1) # [num_init_pose, 3]

            # 角度残差的单位与 wrap：
            # - 输入 gt_roll/gt_pitch 通常以“度”为单位（来自 IMU 或外部姿态源）
            # - 为了做 wrap（跨越 180 度时差值连续），先在“度”域做差并 wrap 到 [-180, 180]
            # - 最后把残差转为“弧度”，作为优化里的 angle_res（与旋转增量 ω 的单位一致，都是 rad）
            roll_base_deg = torch.rad2deg(roll_base)
            pitch_base_deg = torch.rad2deg(pitch_base)

            def _to_deg_tensor(x, *, device):
                if x is None:
                    return torch.zeros(1, device=device, dtype=torch.float64)
                if torch.is_tensor(x):
                    t = x.detach().to(device=device, dtype=torch.float64)
                    return t.reshape(-1)
                if isinstance(x, np.ndarray):
                    v = float(np.reshape(x, -1)[0])
                    return torch.tensor([v], device=device, dtype=torch.float64)
                return torch.tensor([float(x)], device=device, dtype=torch.float64)

            gt_roll_deg_t = _to_deg_tensor(gt_roll, device=pose.device)
            gt_pitch_deg_t = _to_deg_tensor(gt_pitch, device=pose.device)
            if gt_roll_deg_t.numel() == 1:
                gt_roll_deg_t = gt_roll_deg_t.expand(num_init_pose)
            if gt_pitch_deg_t.numel() == 1:
                gt_pitch_deg_t = gt_pitch_deg_t.expand(num_init_pose)

            roll_diff_deg = roll_base_deg - gt_roll_deg_t
            pitch_diff_deg = pitch_base_deg - gt_pitch_deg_t
            # wrap 到 [-180, 180]
            roll_diff_deg = (roll_diff_deg + 180.0) % 360.0 - 180.0
            pitch_diff_deg = (pitch_diff_deg + 180.0) % 360.0 - 180.0

            angle_r = torch.stack(
                [torch.deg2rad(roll_diff_deg), torch.deg2rad(pitch_diff_deg)],
                dim=-1,
            )  # [num_init_pose, 2]
            J_angle_rot = torch.stack([J_roll, J_pitch], dim=1)  # [num_init_pose, 2, 3]

            # 转换为与 J 相同的 dtype
            angle_res = angle_r.to(dtype=J.dtype)  # [num_init_pose, 2]
            J_angle_rot = J_angle_rot.to(dtype=J.dtype)  # [num_init_pose, 2, 3]
            
            # 扩展 J_angle 到完整的 6 自由度（后 3 个平移分量为 0）：
            # 角度 prior 只约束旋转，因此：
            #   d(roll,pitch)/d(t) = 0
            #   d(roll,pitch)/d(ω) = J_angle_rot
            J_angle_full = torch.zeros(num_init_pose, 2, 6, device=J.device, dtype=J.dtype)
            J_angle_full[:, :, :3] = J_angle_rot  # 只有旋转部分有值
            J_angle_tensor = J_angle_full.view(1, num_init_pose, 2, 6)
            angle_res = angle_res.view(1, num_init_pose, 2)
            
            # 打印候选pose的角度和角度残差（用于调试）
            roll_base_deg_np = roll_base_deg.detach().cpu().numpy()
            pitch_base_deg_np = pitch_base_deg.detach().cpu().numpy()
            gt_roll_deg_np = gt_roll_deg_t.detach().cpu().numpy()
            gt_pitch_deg_np = gt_pitch_deg_t.detach().cpu().numpy()
            roll_diff_deg_np = roll_diff_deg.detach().cpu().numpy()
            pitch_diff_deg_np = pitch_diff_deg.detach().cpu().numpy()
            
            # print(f"\n[角度约束] 候选pose数量: {num_init_pose}")
            # for i in range(min(num_init_pose, 10)):  # 最多打印前10个候选pose
            #     print(f"  Pose {i}: roll={roll_base_deg_np[i]:.3f}°, pitch={pitch_base_deg_np[i]:.3f}° | "
            #           f"GT: roll={gt_roll_deg_np[i]:.3f}°, pitch={gt_pitch_deg_np[i]:.3f}° | "
            #           f"残差: roll={roll_diff_deg_np[i]:.3f}°, pitch={pitch_diff_deg_np[i]:.3f}°")
            # if num_init_pose > 10:
            #     print(f"  ... (还有 {num_init_pose - 10} 个候选pose未显示)")
            # print(f"  平均角度残差: roll={np.mean(np.abs(roll_diff_deg_np)):.3f}°, "
            #       f"pitch={np.mean(np.abs(pitch_diff_deg_np)):.3f}°\n")

        # === Step 5: loss 和加权 ===
        cost_proj = (res**2).sum(-1)
        # cost, w_loss_proj, _ = self.loss_fn1(cost)
    
        valid_query_for_selection = (valid_q & visible_q).float()
        valid_count = valid_query_for_selection.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # 候选打分时把“有效点数量差异”归一化，避免低有效点候选因总和更小被误选
        valid_norm = float(cost_proj.shape[-1]) / valid_count
        cost_proj_for_selection = cost_proj * valid_query_for_selection

        # 角度与深度可同时存在：先统一 proj+角度 的鲁棒权重，再按需加深度
        w_loss_angle = None
        if angle_res is not None:
            cost_angle = (angle_res ** 2).sum(-1, keepdim=True)  # [1, num_init_pose, 1]
            cost_all = torch.cat([cost_proj_for_selection, cost_angle], dim=-1)
            cost_all, w_loss_all, _ = self.loss_fn1(cost_all)
            w_loss_proj = w_loss_all[..., :-1]
            w_loss_angle = w_loss_all[..., -1:]
            cost_proj_robust = cost_all[..., :-1] * valid_norm
            cost_angle_robust = cost_all[..., -1:]
            if depth_res is not None:
                cost_depth = depth_res ** 2
                cost_depth_robust, w_loss_depth, _ = self.loss_fn1(
                    cost_depth, truncate=float(self.depth_loss_truncate)
                )
                depth_abs_m_sel = depth_res.abs() * DEPTH_NORM
                sel_gate = torch.ones_like(depth_abs_m_sel)
                sel_linear = 1.0 - (depth_abs_m_sel - self.depth_gate_start_m) / (
                    self.depth_gate_end_m - self.depth_gate_start_m + 1e-12
                )
                sel_linear = sel_linear.clamp(min=0.1, max=1.0)
                sel_gate = torch.where(depth_abs_m_sel >= self.depth_gate_start_m, sel_linear, sel_gate)
                cost_depth_robust = cost_depth_robust * sel_gate
                cost = torch.cat([
                    cost_proj_robust,
                    float(self.depth_selection_weight) * valid_count * cost_depth_robust,
                    float(self.angle_selection_weight) * valid_count * cost_angle_robust,
                ], dim=-1)
            else:
                w_loss_depth = None
                cost = torch.cat([
                    cost_proj_robust,
                    float(self.angle_selection_weight) * valid_count * cost_angle_robust,
                ], dim=-1)
        elif depth_res is not None:
            cost_proj_robust, w_loss_proj, _ = self.loss_fn1(cost_proj_for_selection)
            cost_depth = depth_res ** 2
            cost_depth_robust, w_loss_depth, _ = self.loss_fn1(
                cost_depth, truncate=float(self.depth_loss_truncate)
            )
            depth_abs_m_sel = depth_res.abs() * DEPTH_NORM
            sel_gate = torch.ones_like(depth_abs_m_sel)
            sel_linear = 1.0 - (depth_abs_m_sel - self.depth_gate_start_m) / (
                self.depth_gate_end_m - self.depth_gate_start_m + 1e-12
            )
            sel_linear = sel_linear.clamp(min=0.1, max=1.0)
            sel_gate = torch.where(depth_abs_m_sel >= self.depth_gate_start_m, sel_linear, sel_gate)
            cost_depth_robust = cost_depth_robust * sel_gate
            cost_proj_robust = cost_proj_robust * valid_norm
            cost = torch.cat([
                cost_proj_robust,
                float(self.depth_selection_weight) * valid_count * cost_depth_robust,
            ], dim=-1)
        else:
            cost, w_loss_proj, _ = self.loss_fn1(cost_proj_for_selection)
            cost = cost * valid_norm
            w_loss_depth = None

        valid_query = (valid_q & visible_q).float()
        weight_loss_proj = w_loss_proj * valid_query

        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r * mask.T.view(1, -1, 1)  # [N, C]
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [B, num_init_pose, N]
        weights_proj = weight_loss_proj * w_unc

        grad_proj = (J * res[..., :, :, None]).sum(dim=-2)  # [1, num_init_pose, N, 6]
        grad_proj = weights_proj[:, :, :, None] * grad_proj
        grad_proj = grad_proj.sum(-2)  # [1, num_init_pose, 6]

        Hess_proj = J.permute(0, 1, 2, 4, 3) @ J  # [1, num_init_pose, N, 6, 6]
        Hess_proj = weights_proj[:, :, :, None, None] * Hess_proj
        Hess_proj = Hess_proj.sum(-3)  # [1, num_init_pose, 6, 6]

        # === 深度约束的梯度和 Hessian（如果存在） ===
        if depth_res is not None and w_loss_depth is not None:
            # 与角度先验一致：gate 不乘在 weights 上，只乘在 scale 上；否则 gn_depth 被压小、raw_scale 虚高再被 clamp 压死。
            weights_depth = w_loss_depth * valid_depth * depth_weight
            depth_abs_m = (depth_res.abs() * DEPTH_NORM)
            hard_valid = torch.isfinite(depth_abs_m)
            gate = torch.ones_like(depth_abs_m)
            linear = 1.0 - (depth_abs_m - self.depth_gate_start_m) / (
                self.depth_gate_end_m - self.depth_gate_start_m + 1e-12
            )
            linear = linear.clamp(min=self.depth_gate_min, max=1.0)
            gate = torch.where(depth_abs_m >= self.depth_gate_start_m, linear, gate)
            # 仅用 hard_valid 过滤无效值，gate 只用于后面的 scale（与 angle 一致）
            weights_depth = weights_depth * hard_valid.to(weights_depth.dtype)

            grad_depth = torch.einsum(
                "bnij,bni->bnj", J_depth_tensor, depth_res * weights_depth
            )  # [1, num_init_pose, 6]
            Hess_depth = (
                torch.einsum("bnri,bnrj->bnij", J_depth_tensor, J_depth_tensor)
                * weights_depth[..., None]
            )  # [1, num_init_pose, 6, 6]

            eps_bal = 1e-12
            gn_proj = grad_proj.norm(dim=-1, keepdim=True)
            gn_depth = grad_depth.norm(dim=-1, keepdim=True)
            cos_align = (
                (grad_proj * grad_depth).sum(dim=-1, keepdim=True)
                / (gn_proj * gn_depth + eps_bal)
            ).clamp(min=-1.0, max=1.0)
            raw_scale = (
                self.depth_scale_fixed
                * (gn_proj / (gn_depth + eps_bal))
                * self.depth_target_ratio
            )
            # 先 clamp 再乘 gate（与 angle 一致）；残差大时 gate 减小，避免错误先验拉偏。
            depth_scale_per_pose = raw_scale.clamp(
                min=self.depth_scale_min, max=self.depth_scale_max
            ) * gate
            # 冲突时减权：深度与重投影反向时少用深度，避免“打架”拉偏（角度没有显式做，但深度 log 里 conflict 高）。
            align_factor = ((1.0 + cos_align) / 2.0).clamp(min=0.2, max=1.0)  # cos<0 时压到 [0.2,1]
            depth_scale_per_pose = (depth_scale_per_pose * align_factor).clamp(
                min=self.depth_scale_min, max=self.depth_scale_max
            )
            # 去掉 EMA：角度没有 EMA；EMA 在 scale 常顶帽时会把历史压小，进一步削弱深度。

            if self.depth_debug_print:
                if self._depth_debug_state is not None:
                    st = self._depth_debug_state
                    st["calls_total"] += 1
                    st["calls_with_depth"] += 1
                    dep_m = (depth_res * DEPTH_NORM).detach().cpu().numpy().ravel()
                    gate_np = gate.detach().cpu().numpy().ravel()
                    hard_np = hard_valid.detach().cpu().numpy().ravel().astype(np.float32)
                    scale_np = depth_scale_per_pose.detach().cpu().numpy().ravel()
                    valid_np = valid_depth.detach().cpu().numpy().ravel()
                    eff_np = (weights_depth > 1e-8).detach().cpu().numpy().ravel().astype(np.float32)
                    gn_proj_np = gn_proj.detach().cpu().numpy().ravel()
                    gn_depth_np = gn_depth.detach().cpu().numpy().ravel()
                    ratio_np = gn_depth_np / (gn_proj_np + 1e-6)
                    ratio_np = np.clip(ratio_np, 0.0, 1e3)
                    gn_depth_scaled_np = (
                        (grad_depth * depth_scale_per_pose)
                        .norm(dim=-1, keepdim=True)
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()
                    )
                    gn_total_np = (
                        (grad_proj + grad_depth * depth_scale_per_pose)
                        .norm(dim=-1, keepdim=True)
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()
                    )
                    contrib_np = gn_depth_scaled_np / (gn_total_np + 1e-6)
                    contrib_np = np.clip(contrib_np, 0.0, 1.0)
                    st["res_abs_mean_m"].append(float(np.mean(np.abs(dep_m))))
                    st["res_abs_p90_m"].append(float(np.percentile(np.abs(dep_m), 90)))
                    st["valid_ratio"].append(float(np.mean(valid_np)))
                    st["gate_mean"].append(float(np.mean(gate_np)))
                    st["hard_reject_ratio"].append(float(1.0 - np.mean(hard_np)))
                    st["effective_ratio"].append(float(np.mean(eff_np)))
                    st["scale_mean"].append(float(np.mean(scale_np)))
                    st["scale_max"].append(float(np.max(scale_np)))
                    st["grad_ratio_depth_over_proj"].append(float(np.mean(ratio_np)))
                    st["grad_ratio_depth_over_total"].append(float(np.mean(contrib_np)))
                    if float(np.mean(eff_np)) < 0.05:
                        st["inactive_calls"] += 1
            # if self.depth_debug_verbose:
            #     np_proj = gn_proj.detach().cpu().numpy().ravel()
            #     np_depth = gn_depth.detach().cpu().numpy().ravel()
            #     np_scale = depth_scale_per_pose.detach().cpu().numpy().ravel()
            #     dep_m = (depth_res * DEPTH_NORM).detach().cpu().numpy().ravel()
            #     print(
            #         "[depth_prior] res_m mean={:.3f} | "
            #         "norm_proj mean={:.4f} norm_depth mean={:.6f} "
            #         "scale_adapt mean={:.3f} max={:.3f}".format(
            #             float(np.mean(np.abs(dep_m))),
            #             float(np.mean(np_proj)),
            #             float(np.mean(np_depth)),
            #             float(np.mean(np_scale)),
            #             float(np.max(np_scale)),
            #         )
            #     )

            grad_depth_scaled = grad_depth * depth_scale_per_pose
            Hess_depth_scaled = Hess_depth * depth_scale_per_pose.unsqueeze(-1)
            grad = grad_proj + grad_depth_scaled
            Hess = Hess_proj + Hess_depth_scaled
        else:
            if self.depth_debug_print and self._depth_debug_state is not None:
                self._depth_debug_state["calls_total"] += 1
            grad = grad_proj
            Hess = Hess_proj

        # === 角度约束的梯度和 Hessian（与 costs_angle_0218 完全一致）===
        if angle_res is not None and w_loss_angle is not None:
            angle_weight = 1.0
            weights_angle = w_loss_angle * angle_weight  # [1, num_init_pose, 1]
            grad_angle = torch.einsum(
                "bnij,bni->bnj", J_angle_tensor, angle_res * weights_angle
            )  # [1, num_init_pose, 6]
            Hess_angle = (
                torch.einsum("bnri,bnrj->bnij", J_angle_tensor, J_angle_tensor)
                * weights_angle[..., None]
            )  # [1, num_init_pose, 6, 6]
            eps_bal = 1e-12
            gn_proj = grad_proj.norm(dim=-1, keepdim=True)   # 与 costs_angle_0218 一致：用 grad_proj
            gn_angle = grad_angle.norm(dim=-1, keepdim=True)
            raw_scale = self.angle_scale_fixed * (gn_proj / (gn_angle + eps_bal)) * self.angle_target_ratio
            angle_scale_per_pose = raw_scale.clamp(min=self.angle_scale_min, max=self.angle_scale_max)
            angle_norm_deg = torch.rad2deg(angle_res.norm(dim=-1, keepdim=True))  # [1,N,1]
            start = float(self.angle_gate_start_deg)
            end = float(self.angle_gate_end_deg)
            if end <= start:
                end = start + 1e-3
            gate = 1.0 - (angle_norm_deg - start) / (end - start)
            gate = gate.clamp(min=0.0, max=1.0)
            gate = gate * (1.0 - float(self.angle_gate_min)) + float(self.angle_gate_min)
            angle_scale_per_pose = angle_scale_per_pose * gate
            grad_angle_scaled = grad_angle * angle_scale_per_pose
            Hess_angle_scaled = Hess_angle * angle_scale_per_pose.unsqueeze(-1)
            grad = grad + grad_angle_scaled
            Hess = Hess + Hess_angle_scaled

        w_loss = w_loss_proj
        return -grad, Hess, w_loss, valid_query, p2d_q, cost










        # # -------- Step 5: 深度/角度约束的占位与归一化常数 --------
        # # 深度：固定常数 DEPTH_NORM 做 r_norm = r_m / DEPTH_NORM，使 cost_depth 与单点 cost_proj 同量级。
        # # 角度：残差保持弧度。融合强度由 w_depth、w_angle 控制。
        # depth_res = None
        # J_depth_tensor = None
        # valid_depth = None
        # angle_res = None
        # J_angle_tensor = None
        # # 量纲匹配：cost_proj 单点约 0.01–1，cost_depth=(r_m/DEPTH_NORM)^2。取 DEPTH_NORM=10 时，
        # # 1m→0.01, 5m→0.25, 10m→1，与重投影同量级，w_depth 易落在 [0.1,10]。取 1 则靠 w_depth 学尺度。
        # DEPTH_NORM = 10.0  # 米，不参与训练

        # # -------- Step 6: 深度 + 角度先验（仅当 ECEF、gt_depth、dsm_path 都有时） --------
        # if (
        #     render_T_ecef is not None
        #     and gt_depth is not None
        #     and dsm_path is not None
        # ):
        #     # === 验证查询帧位姿（第一个候选）===
        #     if self.iter <= 3:
        #         # 获取真实深度值（米），不需要乘 mul
        #         if torch.is_tensor(gt_depth):
        #             depth_real = float(gt_depth.detach().cpu().item())
        #         else:
        #             depth_real = float(gt_depth)
                
        #         print(f"\n[验证查询帧] pose_data_q[0,0], depth={depth_real:.2f}m (不乘mul)")
                
        #         # 提取第一个候选位姿
        #         pose_q_first = pose_data_q[:, 0:1, :]  # [1, 1, 12]
                
        #         # 转换查询帧位姿到WGS84
        #         if origin is not None and dd is not None:
        #             t_c2w_ecef_q = self._pose_translations_ecef(pose_q_first, origin, dd)
        #             wgs84_poses_q = self._poses_to_wgs84(pose_q_first, t_c2w_ecef_q)
        #         else:
        #             wgs84_poses_q = self._poses_to_wgs84(pose_q_first, render_T_ecef)
                
        #         lon_q, lat_q, alt_q, roll_q, pitch_q, yaw_q = wgs84_poses_q[0]
        #         print(f"  查询帧WGS84: lon={lon_q:.6f}, lat={lat_q:.6f}, alt={alt_q:.2f}m")
        #         print(f"  查询帧姿态: roll={roll_q:.4f}°, pitch={pitch_q:.4f}°, yaw={yaw_q:.2f}°")
                
        #         # 计算中心点3D坐标，传入真实深度（不乘mul）
        #         center_q = self._center_points_from_wgs84_poses(
        #             wgs84_poses_q, depth_real, K=cam_K
        #         )
                
        #         lon_c, lat_c, z_c = center_q[0]
        #         print(f"  中心点3D: lon={lon_c:.6f}, lat={lat_c:.6f}, z={z_c:.2f}m")
        #         print(f"  相机alt到中心点z的距离: {alt_q - z_c:.2f}m (应该≈depth={depth_real:.2f}m)")
                
        #         # 查询DSM高度
        #         dsm_q = self._query_dsm_heights(center_q, dsm_path, use_gpu=self.use_gpu_dsm)
        #         print(f"  DSM高度: {dsm_q[0]:.2f}m")
        #         print(f"  中心点z - DSM = {z_c - dsm_q[0]:.2f}m (应该接近0)")
        #         print(f"[验证结束]\n")

        #     # -------- 深度约束：解析法雅可比 -------------
        #     # 残差定义: r = z_center - h_dsm（中心射线落点 WGS84 高度 - DSM 高度，单位米）
        #     # 对位姿的雅可比: dr/dpose = dz_center/dpose = [(u×v)^T, -u^T]（6 维）
        #     #   u：ECEF 下“天”方向单位向量（由中心点 lon/lat 得）；v：中心射线在 ECEF 下的方向。
        #     # 旋转部分 (u×v)^T，平移部分 -u^T。
            
        #     # 获取真实深度值（米），用于射线与雅可比计算
        #     if torch.is_tensor(gt_depth):
        #         depth_real = float(gt_depth.detach().cpu().item())
        #     else:
        #         depth_real = float(gt_depth)
            
        #     # 相机内参 [w, h, fx, fy, cx, cy]，缺省时用默认值
        #     if cam_K is None:
        #         cam_K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
        #     w_img, h_img, fx, fy, cx, cy = cam_K
            
        #     # 当前 batch 的位姿：R_w2c（世界到相机）、t_w2c，以及 R_c2w
        #     pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
        #     R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
        #     t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
        #     R_c2w = R_w2c.transpose(1, 2)
            
        #     # 相机中心在局部/ECEF 中的位置 C = R_c2w @ (-t_w2c) + origin + dd
        #     C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)
        #     if origin is not None and dd is not None:
        #         origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
        #         dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
        #         C_ecef = C_local + origin_t + dd_t
        #     else:
        #         C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
            
        #     # 计算中心像素的相机坐标系射线
        #     # 注意：深度单位需要与t_w2c一致
        #     if mul is not None:
        #         depth_for_jac = depth_real * float(mul)  # 转换单位
        #     else:
        #         depth_for_jac = depth_real
            
        #     K_matrix = torch.tensor(
        #         [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        #         device=pose.device, dtype=torch.float64
        #     )
        #     K_inv = torch.inverse(K_matrix)
        #     pixel_center = torch.tensor([cx, cy, 1.0], device=pose.device, dtype=torch.float64)
        #     cam_vec = K_inv @ (depth_for_jac * pixel_center)  # 相机系中心射线（未单位化，长度=深度）
            
        #     # 射线方向在 ECEF 下：v = R_c2w @ cam_vec，用于雅可比中 (u×v)
        #     cam_vec_exp = cam_vec.unsqueeze(0).expand(num_init_pose, -1)
        #     v = torch.bmm(R_c2w, cam_vec_exp.unsqueeze(-1)).squeeze(-1)  # [N, 3]
            
        #     # 由位姿+深度得到中心射线落点的 WGS84 (lon,lat,z)，用于 DSM 查询与 u 的计算
        #     if origin is not None and dd is not None:
        #         wgs84_poses = self._poses_to_wgs84(pose_data_q, C_ecef)
        #         center_points = self._center_points_from_pose_ecef(
        #             pose_data_q, depth_real, K=cam_K, origin=origin, dd=dd, mul=None
        #         )
        #     else:
        #         wgs84_poses = self._poses_to_wgs84(pose_data_q, render_T_ecef)
        #         center_points = self._center_points_from_wgs84_poses(
        #             wgs84_poses, depth_real, K=cam_K
        #         )
            
        #     # DSM 在中心点 (lon,lat) 处插值得到高程 h_dsm
        #     dsm_heights = self._query_dsm_heights(center_points, dsm_path, use_gpu=self.use_gpu_dsm)
            
        #     # 深度残差 r = z_center - h_dsm（WGS84 高度 - DSM），NaN 表示越界或无效
        #     center_points_np = np.array(center_points, dtype=np.float64)
        #     dsm_heights_np = np.array(dsm_heights, dtype=np.float64)
        #     estimated_z = center_points_np[:, 2]  # WGS84 高度
        #     base_residuals = estimated_z - dsm_heights_np
        #     valid_mask = ~np.isnan(base_residuals)
        #     base_residuals = np.nan_to_num(base_residuals, nan=0.0)
            
        #     # “天”方向单位向量 u（ECEF），由中心点经纬度得到，用于 J_depth = [(u×v)^T, -u^T]
        #     lon_rad = np.radians(center_points_np[:, 0])
        #     lat_rad = np.radians(center_points_np[:, 1])
        #     u_np = np.stack([
        #         np.cos(lat_rad) * np.cos(lon_rad),
        #         np.cos(lat_rad) * np.sin(lon_rad),
        #         np.sin(lat_rad)
        #     ], axis=-1)
        #     u = torch.from_numpy(u_np).to(device=pose.device, dtype=torch.float64)
            
        #     # 解析雅可比: 每行 6 维 [旋转3, 平移3] = [(u×v)^T, -u^T]
        #     u_cross_v = torch.cross(u, v, dim=-1)  # [N, 3]
        #     J_depth = torch.cat([u_cross_v, -u], dim=-1)  # [N, 6]
        #     J_depth = J_depth.detach().cpu().numpy()
        #     J_depth = np.nan_to_num(J_depth, nan=0.0)
            
        #     base_residuals_raw = base_residuals.copy()  # 诊断打印用，不归一化
            
        #     # 残差与雅可比同除以 DEPTH_NORM，使量纲与鲁棒核匹配（固定常数，不训练）
        #     base_residuals = base_residuals / DEPTH_NORM
        #     J_depth = J_depth / DEPTH_NORM
            
        #     # 转为与重投影一致的张量形状 [1, num_init_pose, 1] / [1, num_init_pose, 1, 6]
        #     depth_res = torch.from_numpy(base_residuals).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)
        #     J_depth_tensor = torch.from_numpy(J_depth).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1, 6)
        #     valid_depth = torch.from_numpy(valid_mask.astype(np.float32)).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)
            
        #     # -------- 角度约束：roll/pitch 对旋转的雅可比（解析法）--------
        #     # 残差: (roll - gt_roll), (pitch - gt_pitch)，单位弧度。
        #     # 位姿用 ω_c（相机系旋转向量，即李代数），只对旋转求导，平移为 0。
        #     # 链式: d(roll,pitch)/d(ω_c) = d(roll,pitch)/d(R_pose_in_enu) * d(R_pose_in_enu)/d(ω_c)。
            
        #     pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
        #     R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)  # [num_init_pose, 3, 3]
        #     t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
        #     R_c2w = R_w2c.transpose(-1, -2)  # [num_init_pose, 3, 3]
        #     # 相机中心 C 在 ECEF 中，用于得到 lon/lat，进而得到 ENU↔ECEF 的 rot_enu_in_ecef
        #     C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # [num_init_pose, 3]
            
        #     if origin is not None and dd is not None:
        #         origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
        #         dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
        #         C_ecef = C_local + origin_t + dd_t
        #     else:
        #         C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
            
        #     # 相机位置 ECEF -> WGS84，得到每个候选的 (lon, lat)，用于构造 ENU 基
        #     C_ecef_np = C_ecef.detach().cpu().numpy()
        #     if C_ecef_np.ndim == 1:
        #         C_ecef_np = C_ecef_np[None, :]
            
        #     if self._ecef_to_wgs84 is None:
        #         self._ecef_to_wgs84 = pyproj.Transformer.from_crs(
        #             {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        #             "EPSG:4326",
        #             always_xy=True,
        #         )
        #     transformer = self._ecef_to_wgs84
        #     lon_all, lat_all, _ = transformer.transform(
        #         C_ecef_np[:, 0], C_ecef_np[:, 1], C_ecef_np[:, 2], radians=False
        #     )
        #     if np.ndim(lon_all) == 0:
        #         lon_all = np.array([lon_all])
        #         lat_all = np.array([lat_all])
            
        #     # 按相机位置 (lon, lat) 构造 ENU 基：east, north, up，并组成 rot_enu_in_ecef
        #     lat_rad = np.radians(lat_all)
        #     lon_rad = np.radians(lon_all)
        #     lat_rad_t = torch.from_numpy(lat_rad).to(device=pose.device, dtype=torch.float64)
        #     lon_rad_t = torch.from_numpy(lon_rad).to(device=pose.device, dtype=torch.float64)
            
        #     # ENU 三轴在 ECEF 下的分量 -> 3x3 旋转矩阵 [east, north, up]
        #     up = torch.stack([
        #         torch.cos(lon_rad_t) * torch.cos(lat_rad_t),
        #         torch.sin(lon_rad_t) * torch.cos(lat_rad_t),
        #         torch.sin(lat_rad_t)
        #     ], dim=-1)  # [num_init_pose, 3]
        #     east = torch.stack([
        #         -torch.sin(lon_rad_t),
        #         torch.cos(lon_rad_t),
        #         torch.zeros_like(lon_rad_t)
        #     ], dim=-1)  # [num_init_pose, 3]
        #     north = torch.cross(up, east, dim=-1)
        #     rot_enu_in_ecef = torch.stack([east, north, up], dim=-1)  # [num_init_pose, 3, 3]
            
        #     # 代码中相机朝向约定：R_c2w 的 1、2 列取反后才是“标准”ENU 下的机身朝向
        #     R_c2w_prep = R_c2w.clone()
        #     R_c2w_prep[:, :, 1] *= -1
        #     R_c2w_prep[:, :, 2] *= -1
            
        #     # 相机在 ENU 下的姿态矩阵 R_pose_in_enu = (ENU->ECEF)^T @ R_c2w_prep
        #     rot_enu_in_ecef_T = rot_enu_in_ecef.transpose(-1, -2)
        #     R_pose_in_enu = torch.bmm(rot_enu_in_ecef_T, R_c2w_prep)  # [num_init_pose, 3, 3]
            
        #     # 按 xyz（Tait-Bryan）从 R_pose_in_enu 取出 roll/pitch：
        #     # roll = asin(R[2,0])，pitch = atan2(-R[2,1], R[2,2])
        #     R_pose_20 = R_pose_in_enu[:, 2, 0]  # R[2,0]
        #     R_pose_21 = R_pose_in_enu[:, 2, 1]  # R[2,1]
        #     R_pose_22 = R_pose_in_enu[:, 2, 2]  # R[2,2]
            
        #     roll_base = torch.asin(R_pose_20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))  # [num_init_pose]
        #     pitch_base = torch.atan2(-R_pose_21, R_pose_22)  # [num_init_pose]
            
        #     # 第一层链式：d(roll)/d(R[2,0])、d(pitch)/d(R[2,1]), R[2,2]
        #     d_roll_dR20 = 1.0 / torch.sqrt(1.0 - R_pose_20.clamp(-1.0 + 1e-7, 1.0 - 1e-7).pow(2) + 1e-8)
        #     denom_pitch = R_pose_21.pow(2) + R_pose_22.pow(2) + 1e-8
        #     d_pitch_dR21 = -R_pose_22 / denom_pitch
        #     d_pitch_dR22 = R_pose_21 / denom_pitch
            
        #     # 第二层链式：d(R_pose_in_enu)/d(ω_c)。扰动在相机系：R_w2c'=(I+[δω_c]×)R_w2c
        #     # => R_c2w'=R_c2w(I-[δω_c]×) => R_pose_in_enu'=rot_enu_in_ecef^T @ R_c2w @ (I-[δω_c]×)
        #     # 故 d(R_pose_in_enu)/d(ω_c_i) = rot_enu_in_ecef^T @ R_c2w_prep @ (-[e_i]×)
            
        #     # 对轴角 e_i（i=0,1,2）建反对称阵 [e_i]×，再算 R_c2w_prep @ (-[e_i]×)
        #     eye_3 = torch.eye(3, device=pose.device, dtype=torch.float64)
        #     skew_e0 = skew_symmetric(eye_3[0:1])  # [1, 3, 3]
        #     skew_e1 = skew_symmetric(eye_3[1:2])  # [1, 3, 3]
        #     skew_e2 = skew_symmetric(eye_3[2:3])  # [1, 3, 3]
            
        #     # 计算 d(R_pose_in_enu)/d(ω_c) 对每个轴
        #     # d(R_pose_in_enu)/d(ω_c[i]) = rot_enu_in_ecef^T @ R_c2w_prep @ (-[e_i]×)
        #     # 对于每个位姿，计算 R_c2w_prep @ (-[e_i]×)
        #     skew_e0_neg = -skew_e0.squeeze(0)  # [3, 3]
        #     skew_e1_neg = -skew_e1.squeeze(0)  # [3, 3]
        #     skew_e2_neg = -skew_e2.squeeze(0)  # [3, 3]
            
        #     # 批量 R_c2w_prep @ (-[e_j]×)，得到 d(R_pose_in_enu)/d(ω_c_j) 的矩阵形式
        #     R_c2w_skew0 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e0_neg)  # [num_init_pose, 3, 3]
        #     R_c2w_skew1 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e1_neg)  # [num_init_pose, 3, 3]
        #     R_c2w_skew2 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e2_neg)  # [num_init_pose, 3, 3]
            
        #     # d(R_pose_in_enu)/d(ω_c_j) = rot_enu_in_ecef_T @ R_c2w_skew_j
        #     dR_pose_dw0 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew0)
        #     dR_pose_dw1 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew1)
        #     dR_pose_dw2 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew2)
            
        #     # 取出第 3 行元素对 ω_c 的导数，用于链式乘 d(roll)/d(R20)、d(pitch)/d(R21),d(R22)
        #     dR20_dw0 = dR_pose_dw0[:, 2, 0]  # [num_init_pose]
        #     dR20_dw1 = dR_pose_dw1[:, 2, 0]  # [num_init_pose]
        #     dR20_dw2 = dR_pose_dw2[:, 2, 0]  # [num_init_pose]
            
        #     dR21_dw0 = dR_pose_dw0[:, 2, 1]  # [num_init_pose]
        #     dR21_dw1 = dR_pose_dw1[:, 2, 1]  # [num_init_pose]
        #     dR21_dw2 = dR_pose_dw2[:, 2, 1]  # [num_init_pose]
            
        #     dR22_dw0 = dR_pose_dw0[:, 2, 2]  # [num_init_pose]
        #     dR22_dw1 = dR_pose_dw1[:, 2, 2]  # [num_init_pose]
        #     dR22_dw2 = dR_pose_dw2[:, 2, 2]  # [num_init_pose]
            
        #     # 链式得到 d(roll)/d(ω_c) 和 d(pitch)/d(ω_c)，均为 [num_init_pose, 3]
        #     J_roll = torch.stack([
        #         d_roll_dR20 * dR20_dw0,
        #         d_roll_dR20 * dR20_dw1,
        #         d_roll_dR20 * dR20_dw2
        #     ], dim=-1)
        #     J_pitch = torch.stack([
        #         d_pitch_dR21 * dR21_dw0 + d_pitch_dR22 * dR22_dw0,
        #         d_pitch_dR21 * dR21_dw1 + d_pitch_dR22 * dR22_dw1,
        #         d_pitch_dR21 * dR21_dw2 + d_pitch_dR22 * dR22_dw2
        #     ], dim=-1)  # [num_init_pose, 3]

        #     # 获取 GT 值（如果提供），并转换为弧度
        #     if gt_roll is not None:
        #         # 转换为弧度并处理不同输入类型
        #         if torch.is_tensor(gt_roll):
        #             gt_roll_val = gt_roll.detach().to(device=pose.device, dtype=torch.float64)
        #             if gt_roll_val.ndim == 0:
        #                 gt_roll_val = gt_roll_val.unsqueeze(0)
        #             gt_roll_rad = torch.deg2rad(gt_roll_val)
        #         elif isinstance(gt_roll, np.ndarray):
        #             gt_roll_val = np.reshape(gt_roll, -1)[0]
        #             gt_roll_rad = torch.deg2rad(torch.tensor([float(gt_roll_val)], device=pose.device, dtype=torch.float64))
        #         else:
        #             gt_roll_rad = torch.deg2rad(torch.tensor([float(gt_roll)], device=pose.device, dtype=torch.float64))
        #     else:
        #         gt_roll_rad = torch.zeros(1, device=pose.device, dtype=torch.float64)
        #         print(f"gt_roll is None")
            
        #     if gt_pitch is not None:
        #         # 转换为弧度并处理不同输入类型
        #         if torch.is_tensor(gt_pitch):
        #             gt_pitch_val = gt_pitch.detach().to(device=pose.device, dtype=torch.float64)
        #             if gt_pitch_val.ndim == 0:
        #                 gt_pitch_val = gt_pitch_val.unsqueeze(0)
        #             gt_pitch_rad = torch.deg2rad(gt_pitch_val)
        #         elif isinstance(gt_pitch, np.ndarray):
        #             gt_pitch_val = np.reshape(gt_pitch, -1)[0]
        #             gt_pitch_rad = torch.deg2rad(torch.tensor([float(gt_pitch_val)], device=pose.device, dtype=torch.float64))
        #         else:
        #             gt_pitch_rad = torch.deg2rad(torch.tensor([float(gt_pitch)], device=pose.device, dtype=torch.float64))
        #     else:
        #         gt_pitch_rad = torch.zeros(1, device=pose.device, dtype=torch.float64)
        #         print(f"gt_pitch is None")

        #     # 残差是 (roll - gt_roll) 和 (pitch - gt_pitch)，归一化到 [-π, π]
        #     # 否则 361° 与 1° 的差会变成 360° 而非 0°（角度周期 2π）
        #     if gt_roll_rad.numel() == 1:
        #         gt_roll_rad = gt_roll_rad.expand(num_init_pose)
        #     if gt_pitch_rad.numel() == 1:
        #         gt_pitch_rad = gt_pitch_rad.expand(num_init_pose)
        #     def _wrap_angle_rad(diff):
        #         return torch.atan2(torch.sin(diff), torch.cos(diff))
        #     angle_r = torch.stack([
        #         _wrap_angle_rad(roll_base - gt_roll_rad),
        #         _wrap_angle_rad(pitch_base - gt_pitch_rad)
        #     ], dim=-1)  # [num_init_pose, 2]
        #     J_angle_rot = torch.stack([J_roll, J_pitch], dim=1)  # [num_init_pose, 2, 3]

        #     # 转换为与 J 相同的 dtype
        #     angle_res = angle_r.to(dtype=J.dtype)  # [num_init_pose, 2]
        #     J_angle_rot = J_angle_rot.to(dtype=J.dtype)  # [num_init_pose, 2, 3]
            
        #     # 扩展 J_angle 到完整的 6 自由度（后3个平移分量为0）
        #     J_angle_full = torch.zeros(num_init_pose, 2, 6, device=J.device, dtype=J.dtype)
        #     J_angle_full[:, :, :3] = J_angle_rot  # 只有旋转部分有值
        #     J_angle_tensor = J_angle_full.view(1, num_init_pose, 2, 6)
        #     angle_res = angle_res.view(1, num_init_pose, 2)

        #     # 打印：WGS84 位姿、反投影中心点高度、DSM 中心高度、深度残差、角度残差
        #     if self.iter <= 3:
        #         print(f"\n=== [深度/角度残差诊断] iter={self.iter} ===")
        #         for idx in range(num_init_pose):
        #             lon, lat, alt, roll, pitch, yaw = wgs84_poses[idx]
        #             z_center = float(center_points_np[idx, 2])
        #             h_dsm = float(dsm_heights_np[idx])
        #             r_depth = float(base_residuals_raw[idx])
        #             roll_res_rad = angle_res[0, idx, 0].item()
        #             pitch_res_rad = angle_res[0, idx, 1].item()
        #             roll_res_deg = np.degrees(roll_res_rad)
        #             pitch_res_deg = np.degrees(pitch_res_rad)
        #             print(f"  候选 {idx}:")
        #             print(f"    WGS84 pose: lon={lon:.6f}, lat={lat:.6f}, alt={alt:.2f}m, roll={roll:.4f}°, pitch={pitch:.4f}°, yaw={yaw:.4f}°")
        #             print(f"    反投影中心点3D高度 z_center = {z_center:.2f} m")
        #             print(f"    查表DSM中心点高度 h_dsm    = {h_dsm:.2f} m")
        #             print(f"    深度残差 r_depth = z_center - h_dsm = {r_depth:.4f} m")
        #             print(f"    角度残差 roll(rad)={roll_res_rad:.6f}, pitch(rad)={pitch_res_rad:.6f} | roll(deg)={roll_res_deg:.4f}°, pitch(deg)={pitch_res_deg:.4f}°")
        #         print(f"=== [诊断结束] ===\n", flush=True)
        
        # # -------- 统一 loss 与鲁棒权重 --------
        # # cost 定义：投影为 per-point 的 (res^2).sum(dim=C)；深度、角度为标量平方。
        # cost_proj = (res**2).sum(-1)  # [1, num_init_pose, N]

        # if depth_res is not None:
        #     cost_depth = depth_res ** 2  # [1, num_init_pose, 1]
        #     cost_angle = (angle_res ** 2).sum(-1, keepdim=True)  # [1, num_init_pose, 1]
        #     # 投影|深度|角度拼成 cost_all，一次 loss_fn1 得到统一鲁棒核的一阶导 w_loss_all
        #     cost_all = torch.cat([cost_proj, cost_depth, cost_angle], dim=-1)  # [1, num_init_pose, N+2]
        #     cost_all, w_loss_all, _ = self.loss_fn1(cost_all)
        #     w_loss_proj = w_loss_all[..., :-2]    # [1, num_init_pose, N]
        #     w_loss_depth = w_loss_all[..., -2:-1]  # [1, num_init_pose, 1]
        #     w_loss_angle = w_loss_all[..., -1:]    # [1, num_init_pose, 1]
        #     cost = cost_all[..., :-2]  # 对外只返回投影部分 cost
        # else:
        #     cost, w_loss_proj, _ = self.loss_fn1(cost_proj)
        #     w_loss_depth = None
        #     w_loss_angle = None

        # # -------- 投影约束的逐点权重 --------
        # # 有效掩码 + 鲁棒权重 + 不确定性权重。无效点、不可见点权重为 0。
        # valid_query = (valid_q & visible_q).float()
        # weight_loss_proj = w_loss_proj * valid_query

        # # 在 c_query、c_ref 上对 p2d_q、p2d_r 插值得到“置信度”weight_q/weight_r，相乘作为 w_unc
        # weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        # weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])
        # weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        # weight_r = weight_r * mask.T.view(1, -1, 1)
        # weight_r = weight_r.repeat(1, num_init_pose, 1, 1)
        # w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [1, num_init_pose, N]
        # weights_proj = weight_loss_proj * w_unc  # 最终投影逐点权重

        # # -------- 投影约束的梯度与 Hessian --------
        # # grad_proj = sum_n weights_proj[n] * J[n].T @ res[n]；Hess_proj = sum_n weights_proj[n] * J[n].T @ J[n]
        # grad_proj = (J * res[..., :, :, None]).sum(dim=-2)  # [1, num_init_pose, N, 6]
        # grad_proj = weights_proj[:, :, :, None] * grad_proj
        # grad_proj = grad_proj.sum(-2)  # [1, num_init_pose, 6]

        # Hess_proj = J.permute(0, 1, 2, 4, 3) @ J  # [1, num_init_pose, N, 6, 6]
        # Hess_proj = weights_proj[:, :, :, None, None] * Hess_proj
        # Hess_proj = Hess_proj.sum(-3)  # [1, num_init_pose, 6, 6]

        # # -------- 深度/角度约束的梯度与 Hessian，及自适应权重融合 --------
        # # 
        # # 【核心设计】目标是定位准确，残差大 = 偏离GT远 = 需要更强矫正
        # #   - 让"单位残差产生的梯度"相等（归一化雅可比差异）
        # #   - 这样残差越大，梯度自然越大，矫正力度自然越强
        # #   - 完全自动，无需手动阈值
        # #
        # if depth_res is not None and w_loss_depth is not None:
        #     # 深度：weights_depth = w_loss_depth * valid_depth（鲁棒核输出，已对异常值降权）
        #     weights_depth = w_loss_depth * valid_depth  # [1, num_init_pose, 1]
        #     grad_depth = J_depth_tensor * (depth_res * weights_depth)[..., None]
        #     grad_depth = grad_depth.sum(-2)  # [1, num_init_pose, 6]
        #     Hess_depth = J_depth_tensor[..., :, None] * J_depth_tensor[..., None, :]
        #     Hess_depth = weights_depth[..., None, None] * Hess_depth
        #     Hess_depth = Hess_depth.sum(-3)

        #     # 角度：weights_angle = w_loss_angle
        #     weights_angle = w_loss_angle  # [1, num_init_pose, 1]
        #     grad_angle = torch.einsum("bnij,bni->bnj", J_angle_tensor, angle_res * weights_angle)
        #     Hess_angle = torch.einsum("bnri,bnrj->bnij", J_angle_tensor, J_angle_tensor) * weights_angle[..., None]

        #     # ============ 自适应权重设计：只用角度约束 ============
        #     # 
        #     # 【设计思想】
        #     #   1. 残差归一化：把角度残差归一化到"几个标准差"的无量纲尺度
        #     #   2. 梯度范数均衡：补偿点数差异，让角度有合理的梯度贡献
        #     #   3. 残差驱动：残差越大（偏离越远），权重越大，矫正力度越强
        #     #
        #     # 【公式】
        #     #   w = scale_factor * residual_factor
        #     #   - scale_factor = gn_proj / gn_angle （补偿梯度范数差异）
        #     #   - residual_factor = normalized_residual（残差越大，权重越大）
        #     #
        #     eps_bal = 1e-8
            
        #     # ---- 1. 残差归一化（解决单位问题）----
        #     SIGMA_ANGLE = 0.05   # 角度典型误差：0.05弧度 ≈ 3度
            
        #     # 计算归一化残差（无量纲，表示"几个标准差"）
        #     angle_res_norm = torch.abs(angle_res) / SIGMA_ANGLE  # [1, num_init_pose, 2]
        #     angle_deviation = angle_res_norm.mean()  # 标量
            
        #     # ---- 2. 梯度范数均衡（解决点数问题）----
        #     gn_proj = torch.norm(grad_proj) + eps_bal
        #     gn_angle = torch.norm(grad_angle) + eps_bal
            
        #     # scale_factor：补偿梯度范数差异
        #     scale_angle = gn_proj / gn_angle
            
        #     # ---- 3. 残差驱动的自适应权重 ----
        #     # 
        #     # 核心思想：
        #     #   - 残差大（偏离远）→ 需要更强矫正 → 权重大
        #     #   - 残差小（接近正确）→ 不需要过度干预 → 权重小
        #     #
        #     CAP = 2.0  # 2个标准差以上视为"严重偏离"
        #     residual_factor_angle = torch.clamp(angle_deviation / CAP, min=0.1, max=1.0)
            
        #     # ---- 4. 最终权重 = scale_factor * residual_factor ----
        #     w_a_auto = scale_angle * residual_factor_angle
        #     w_a_auto = w_a_auto.clamp(0.01, 100.0)

        #     # 合并梯度和 Hessian（只有重投影 + 角度，不用深度）
        #     grad = grad_proj + w_a_auto * grad_angle
        #     Hess = Hess_proj + w_a_auto * Hess_angle
        #     w_loss = w_loss_proj
            
        #     # ============ Loss 只用重投影（用于候选选择）============
        #     cost_total = cost_proj  # [1, num_init_pose, N]

        #     # 打印诊断
        #     if self.iter <= 3:
        #         gn_total = torch.norm(grad).item() + eps_bal
        #         ratio_proj = torch.norm(grad_proj).item() / gn_total * 100
        #         ratio_angle = torch.norm(w_a_auto * grad_angle).item() / gn_total * 100
                
        #         # 角度残差统计（度）
        #         roll_res_all = torch.rad2deg(angle_res[..., 0]).squeeze(0)
        #         pitch_res_all = torch.rad2deg(angle_res[..., 1]).squeeze(0)
        #         roll_abs_mean = roll_res_all.abs().mean().item()
        #         pitch_abs_mean = pitch_res_all.abs().mean().item()
                
        #         print(f"\n=== [角度约束自适应权重] iter={self.iter} ===")
        #         print(f"  【归一化残差】angle={angle_deviation.item():.2f}σ")
        #         print(f"  【残差因子】angle={residual_factor_angle.item():.2f}")
        #         print(f"  【梯度范数】proj={gn_proj.item():.4f} | angle={gn_angle.item():.4f}")
        #         print(f"  【自适应权重】w_a={w_a_auto.item():.2f}")
        #         print(f"  【角度残差】roll={roll_abs_mean:.2f}° | pitch={pitch_abs_mean:.2f}°")
        #         print(f"  【梯度占比】重投影={ratio_proj:.1f}% | 角度={ratio_angle:.1f}%")
        #         print(f"=== [诊断结束] ===\n", flush=True)
        # else:
        #     # 无深度/角度先验时，仅用投影的梯度与 Hessian
        #     grad = grad_proj
        #     Hess = Hess_proj
        #     w_loss = w_loss_proj
        #     cost_total = cost  # 只有投影 cost
        
        # # 返回：负梯度、Hessian、投影鲁棒权重、有效掩码、查询帧 2D 点、融合后总 cost
        # return -grad, Hess, w_loss, valid_query, p2d_q, cost_total
    
