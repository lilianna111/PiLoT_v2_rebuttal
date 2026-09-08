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
from .optimization import J_normalization, skew_symmetric
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
        c_ref,c_query,p2d_r,visible_r, gt_depth=None, gt_roll=None, gt_pitch=None, dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None):


        num_init_pose = pose_data_q.shape[1]
        # Use the actual camera intrinsics for the center-ray computation.
        cam_K = None
        if cam_data_q is not None and torch.is_tensor(cam_data_q) and cam_data_q.numel() >= 6:
            cam_K = cam_data_q[0, 0, :6].detach().cpu().tolist()
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)
        # print("fp_r: ",fp_r.data)

        valid_ref_mask = (valid_r & visible_r).to(torch.float32)  # shape: [1, N]
        mask = valid_ref_mask.transpose(1, 0)  # shape: [N, 1]，避免 unsqueeze

        fp_r = fp_r[0] * mask  # [N, C]
        p3D = p3D[0] * mask    # [N, 3]

        fp_r = fp_r.repeat(1, num_init_pose, 1, 1)
        p3D = p3D.repeat(1, num_init_pose, 1, 1)
        
        fp_r = torch.nn.functional.normalize(fp_r, dim=-1)
        

        # === Step 2: 查询帧中投影 ===
        p3d_q = transform_p3d(pose_data_q, p3D)
        p2d_q, visible_q = project_p3d(cam_data_q, p3d_q)

        p2d_q = p2d_q * mask.T.view(1, 1, -1, 1)  # [B, N, 2] * [1, N, 1]  # [B, N, 2]，广播避免 unsqueeze
        
        # === Step 3: 查询帧特征采样 ===
        fp_q, valid_q, J_f = self.interpolate_feature_map(f_q, p2d_q.reshape(1, -1, 2), return_gradients=True)
        fp_q = fp_q.view(1, num_init_pose, -1, fp_q.shape[-1])
        # print("fp_q: ",fp_q.data)
        valid_q = valid_q.view(1, num_init_pose, -1)
        J_f = J_f.view(1, num_init_pose, -1, J_f.shape[-2], J_f.shape[-1])
        visible_q = visible_q.view(1, num_init_pose, -1)
        
        res = fp_q - fp_r

        # === Step 4: Jacobians ===
        R = pose_data_q[..., :9].view(1, -1, 3, 3)
        J_p3d_rot = R[:, :, None].matmul(-skew_symmetric(p3D))
        J_p3d_tran = R[:, :, None].expand(-1, -1, J_p3d_rot.shape[2], -1, -1)
        J_p3d_pose = torch.cat((J_p3d_rot, J_p3d_tran), dim=-1)

        fx, fy = cam_data_q[:, :, 2], cam_data_q[:, :, 3]
        zero = torch.zeros_like(fx)
        f_diag_embed = torch.stack([fx, zero, zero, fy], dim=-1).reshape(1, -1, 1, 2, 2)

        J_p2d_p3d = f_diag_embed @ J_project(p3d_q)  # [B, N, 2, 3]
        J = J_f @ J_p2d_p3d @ J_p3d_pose             # [B, N, 2, 6]

        # === Step 5: 计算深度约束和角度约束的残差和Jacobian（前置计算） ===
        depth_res = None
        J_depth_tensor = None
        valid_depth = None
        angle_res = None
        J_angle_tensor = None
        
        depth_weight = 3.0        # 深度约束基础权重
        angle_weight = 3.0        # 角度约束基础权重（增加以提升影响）
        depth_sigma = 100.0       # 深度归一化因子
        angle_sigma = 10.0        # 角度归一化因子（度）
        depth_truncate = 5.0      # 深度鲁棒核截断阈值
        angle_truncate = 5.0      # 角度鲁棒核截断阈值（度）

        # === Step 6: Soft depth + angle priors (adaptive scaling) ===
        if (
            render_T_ecef is not None
            and gt_depth is not None
            and dsm_path is not None
        ):
            # === 验证查询帧位姿（第一个候选）===
            if self.iter <= 3:
                # 获取真实深度值（米），不需要乘 mul
                if torch.is_tensor(gt_depth):
                    depth_real = float(gt_depth.detach().cpu().item())
                else:
                    depth_real = float(gt_depth)
                
                print(f"\n[验证查询帧] pose_data_q[0,0], depth={depth_real:.2f}m (不乘mul)")
                
                # 提取第一个候选位姿
                pose_q_first = pose_data_q[:, 0:1, :]  # [1, 1, 12]
                
                # 转换查询帧位姿到WGS84
                if origin is not None and dd is not None:
                    t_c2w_ecef_q = self._pose_translations_ecef(pose_q_first, origin, dd)
                    wgs84_poses_q = self._poses_to_wgs84(pose_q_first, t_c2w_ecef_q)
                else:
                    wgs84_poses_q = self._poses_to_wgs84(pose_q_first, render_T_ecef)
                
                lon_q, lat_q, alt_q, roll_q, pitch_q, yaw_q = wgs84_poses_q[0]
                print(f"  查询帧WGS84: lon={lon_q:.6f}, lat={lat_q:.6f}, alt={alt_q:.2f}m")
                print(f"  查询帧姿态: roll={roll_q:.4f}°, pitch={pitch_q:.4f}°, yaw={yaw_q:.2f}°")
                
                # 计算中心点3D坐标，传入真实深度（不乘mul）
                center_q = self._center_points_from_wgs84_poses(
                    wgs84_poses_q, depth_real, K=cam_K
                )
                
                lon_c, lat_c, z_c = center_q[0]
                print(f"  中心点3D: lon={lon_c:.6f}, lat={lat_c:.6f}, z={z_c:.2f}m")
                print(f"  相机alt到中心点z的距离: {alt_q - z_c:.2f}m (应该≈depth={depth_real:.2f}m)")
                
                # 查询DSM高度
                dsm_q = self._query_dsm_heights(center_q, dsm_path, use_gpu=self.use_gpu_dsm)
                print(f"  DSM高度: {dsm_q[0]:.2f}m")
                print(f"  中心点z - DSM = {z_c - dsm_q[0]:.2f}m (应该接近0)")
                print(f"[验证结束]\n")
            
            depth_weight = 3.0        # 基础权重
            depth_truncate = 5.0      # 鲁棒核函数的截断阈值

            # === 解析法计算雅可比矩阵 ===
            # 残差: r = z_center - h_dsm
            # 雅可比: dr/dpose = dz_center/dpose = [(u × v)^T, -u^T]
            # 其中 u 是"天"方向单位向量，v 是射线方向
            
            # 获取真实深度值（米）
            if torch.is_tensor(gt_depth):
                depth_real = float(gt_depth.detach().cpu().item())
            else:
                depth_real = float(gt_depth)
            
            # 获取相机内参
            if cam_K is None:
                cam_K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0]
            w_img, h_img, fx, fy, cx, cy = cam_K
            
            # 提取位姿信息
            pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
            R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
            t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
            R_c2w = R_w2c.transpose(1, 2)
            
            # 计算相机中心在ECEF坐标系中的位置
            C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)
            if origin is not None and dd is not None:
                origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
                dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
                C_ecef = C_local + origin_t + dd_t
            else:
                C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
            
            # 计算中心像素的相机坐标系射线
            # 注意：深度单位需要与t_w2c一致
            if mul is not None:
                depth_for_jac = depth_real * float(mul)  # 转换单位
            else:
                depth_for_jac = depth_real
            
            K_matrix = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                device=pose.device, dtype=torch.float64
            )
            K_inv = torch.inverse(K_matrix)
            pixel_center = torch.tensor([cx, cy, 1.0], device=pose.device, dtype=torch.float64)
            cam_vec = K_inv @ (depth_for_jac * pixel_center)  # [3]
            
            # 计算ECEF坐标系中的射线方向 v = R_c2w @ cam_vec
            cam_vec_exp = cam_vec.unsqueeze(0).expand(num_init_pose, -1)
            v = torch.bmm(R_c2w, cam_vec_exp.unsqueeze(-1)).squeeze(-1)  # [N, 3]
            
            # 计算3D中心点和WGS84坐标
            if origin is not None and dd is not None:
                wgs84_poses = self._poses_to_wgs84(pose_data_q, C_ecef)
                center_points = self._center_points_from_pose_ecef(
                    pose_data_q, gt_depth, K=cam_K, origin=origin, dd=dd, mul=mul
                )
            else:
                wgs84_poses = self._poses_to_wgs84(pose_data_q, render_T_ecef)
                center_points = self._center_points_from_wgs84_poses(
                    wgs84_poses, depth_real, K=cam_K
                )
            
            # 查询DSM高度
            dsm_heights = self._query_dsm_heights(center_points, dsm_path, use_gpu=self.use_gpu_dsm)
            
            # 计算残差
            center_points_np = np.array(center_points, dtype=np.float64)
            dsm_heights_np = np.array(dsm_heights, dtype=np.float64)
            estimated_z = center_points_np[:, 2]  # WGS84高度
            base_residuals = estimated_z - dsm_heights_np
            valid_mask = ~np.isnan(base_residuals)
            base_residuals = np.nan_to_num(base_residuals, nan=0.0)
            
            # 计算"天"方向单位向量 u（ECEF坐标系）
            lon_rad = np.radians(center_points_np[:, 0])
            lat_rad = np.radians(center_points_np[:, 1])
            u_np = np.stack([
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad)
            ], axis=-1)
            u = torch.from_numpy(u_np).to(device=pose.device, dtype=torch.float64)
            
            # 解析雅可比: J = [(u × v)^T, -u^T]
            u_cross_v = torch.cross(u, v, dim=-1)  # [N, 3]
            J_depth = torch.cat([u_cross_v, -u], dim=-1)  # [N, 6]
            J_depth = J_depth.detach().cpu().numpy()
            J_depth = np.nan_to_num(J_depth, nan=0.0)
            
            # 保存原始残差用于打印
            base_residuals_raw = base_residuals.copy()
            
            # 归一化残差和 Jacobian
            base_residuals = base_residuals / depth_sigma
            J_depth = J_depth / depth_sigma
            
            # 转换为统一格式 [1, num_init_pose, 1]
            depth_res = torch.from_numpy(base_residuals).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)
            J_depth_tensor = torch.from_numpy(J_depth).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1, 6)
            valid_depth = torch.from_numpy(valid_mask.astype(np.float32)).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)
            
            # === 角度约束：roll 和 pitch 使用 gt_roll 和 gt_pitch（解析法）===
            # 解析法计算 d(roll)/d(ω) 和 d(pitch)/d(ω)，其中 ω 是旋转向量（李代数）
            
            # 提取位姿信息
            pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
            R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)  # [num_init_pose, 3, 3]
            
            # 计算相机中心在ECEF坐标系中的位置（用于计算ENU旋转矩阵）
            t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
            R_c2w = R_w2c.transpose(-1, -2)  # [num_init_pose, 3, 3]
            C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # [num_init_pose, 3]
            
            if origin is not None and dd is not None:
                origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
                dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
                C_ecef = C_local + origin_t + dd_t
            else:
                C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
            
            # 转换到WGS84获取经纬度
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
            
            # 计算ENU到ECEF的旋转矩阵（对每个位姿）
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
            
            # 计算 R_c2w（注意：需要对第1列和第2列取反）
            R_c2w_prep = R_c2w.clone()
            R_c2w_prep[:, :, 1] *= -1
            R_c2w_prep[:, :, 2] *= -1
            
            # 计算 R_pose_in_enu = rot_enu_in_ecef^T @ R_c2w
            rot_enu_in_ecef_T = rot_enu_in_ecef.transpose(-1, -2)  # [num_init_pose, 3, 3]
            R_pose_in_enu = torch.bmm(rot_enu_in_ecef_T, R_c2w_prep)  # [num_init_pose, 3, 3]
            
            # 从 R_pose_in_enu 提取 roll 和 pitch（xyz 欧拉角顺序）
            # 对于 "xyz" 顺序（Tait-Bryan角）：
            # pitch (x轴) = atan2(-R[2,1], R[2,2])
            # roll (y轴) = asin(R[2,0])
            # yaw (z轴) = atan2(-R[1,0], R[0,0])
            R_pose_20 = R_pose_in_enu[:, 2, 0]  # R[2,0]
            R_pose_21 = R_pose_in_enu[:, 2, 1]  # R[2,1]
            R_pose_22 = R_pose_in_enu[:, 2, 2]  # R[2,2]
            
            # 计算 roll 和 pitch
            roll_base = torch.asin(R_pose_20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))  # [num_init_pose]
            pitch_base = torch.atan2(-R_pose_21, R_pose_22)  # [num_init_pose]
            
            # 计算 d(roll)/d(R_pose_in_enu) 和 d(pitch)/d(R_pose_in_enu)
            # d(roll)/d(R[2,0]) = 1 / sqrt(1 - R[2,0]^2)
            d_roll_dR20 = 1.0 / torch.sqrt(1.0 - R_pose_20.clamp(-1.0 + 1e-7, 1.0 - 1e-7).pow(2) + 1e-8)
            
            # d(pitch)/d(R[2,1]) = -R[2,2] / (R[2,1]^2 + R[2,2]^2)
            # d(pitch)/d(R[2,2]) = R[2,1] / (R[2,1]^2 + R[2,2]^2)
            denom_pitch = R_pose_21.pow(2) + R_pose_22.pow(2) + 1e-8
            d_pitch_dR21 = -R_pose_22 / denom_pitch
            d_pitch_dR22 = R_pose_21 / denom_pitch
            
            # 计算 d(R_pose_in_enu)/d(ω_c)，其中 ω_c 是相机坐标系中的旋转向量
            # 当在相机坐标系中施加扰动 δω_c 时：
            # R_w2c_new = (I + [δω_c]×) @ R_w2c
            # R_c2w_new = R_w2c_new^T = R_w2c^T @ (I - [δω_c]×) ≈ R_c2w @ (I - [δω_c]×)
            # R_pose_in_enu_new = rot_enu_in_ecef^T @ R_c2w_new ≈ rot_enu_in_ecef^T @ R_c2w @ (I - [δω_c]×)
            # 所以：d(R_pose_in_enu)/d(ω_c) = rot_enu_in_ecef^T @ R_c2w @ (-[·]×)
            
            # 构建单位反对称矩阵 [e_i]×，其中 e_i 是单位向量
            eye_3 = torch.eye(3, device=pose.device, dtype=torch.float64)
            skew_e0 = skew_symmetric(eye_3[0:1])  # [1, 3, 3]
            skew_e1 = skew_symmetric(eye_3[1:2])  # [1, 3, 3]
            skew_e2 = skew_symmetric(eye_3[2:3])  # [1, 3, 3]
            
            # 计算 d(R_pose_in_enu)/d(ω_c) 对每个轴
            # d(R_pose_in_enu)/d(ω_c[i]) = rot_enu_in_ecef^T @ R_c2w_prep @ (-[e_i]×)
            # 对于每个位姿，计算 R_c2w_prep @ (-[e_i]×)
            skew_e0_neg = -skew_e0.squeeze(0)  # [3, 3]
            skew_e1_neg = -skew_e1.squeeze(0)  # [3, 3]
            skew_e2_neg = -skew_e2.squeeze(0)  # [3, 3]
            
            # 对每个位姿计算 R_c2w_prep[i] @ (-[e_j]×)
            # R_c2w_prep: [num_init_pose, 3, 3], skew_e*_neg: [3, 3]
            # 使用 einsum 进行批量矩阵乘法
            R_c2w_skew0 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e0_neg)  # [num_init_pose, 3, 3]
            R_c2w_skew1 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e1_neg)  # [num_init_pose, 3, 3]
            R_c2w_skew2 = torch.einsum('bij,jk->bik', R_c2w_prep, skew_e2_neg)  # [num_init_pose, 3, 3]
            
            # 然后计算 rot_enu_in_ecef_T @ R_c2w_skew
            dR_pose_dw0 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew0)  # [num_init_pose, 3, 3]
            dR_pose_dw1 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew1)  # [num_init_pose, 3, 3]
            dR_pose_dw2 = torch.bmm(rot_enu_in_ecef_T, R_c2w_skew2)  # [num_init_pose, 3, 3]
            
            # 提取 d(R_pose[2,0])/d(ω_c) 和 d(R_pose[2,1])/d(ω_c), d(R_pose[2,2])/d(ω_c)
            dR20_dw0 = dR_pose_dw0[:, 2, 0]  # [num_init_pose]
            dR20_dw1 = dR_pose_dw1[:, 2, 0]  # [num_init_pose]
            dR20_dw2 = dR_pose_dw2[:, 2, 0]  # [num_init_pose]
            
            dR21_dw0 = dR_pose_dw0[:, 2, 1]  # [num_init_pose]
            dR21_dw1 = dR_pose_dw1[:, 2, 1]  # [num_init_pose]
            dR21_dw2 = dR_pose_dw2[:, 2, 1]  # [num_init_pose]
            
            dR22_dw0 = dR_pose_dw0[:, 2, 2]  # [num_init_pose]
            dR22_dw1 = dR_pose_dw1[:, 2, 2]  # [num_init_pose]
            dR22_dw2 = dR_pose_dw2[:, 2, 2]  # [num_init_pose]
            
            # 通过链式法则计算 d(roll)/d(ω_c) 和 d(pitch)/d(ω_c)
            # d(roll)/d(ω_c) = d(roll)/d(R[2,0]) * d(R[2,0])/d(ω_c)
            J_roll = torch.stack([
                d_roll_dR20 * dR20_dw0,
                d_roll_dR20 * dR20_dw1,
                d_roll_dR20 * dR20_dw2
            ], dim=-1)  # [num_init_pose, 3]
            
            # d(pitch)/d(ω_c) = d(pitch)/d(R[2,1]) * d(R[2,1])/d(ω_c) + d(pitch)/d(R[2,2]) * d(R[2,2])/d(ω_c)
            J_pitch = torch.stack([
                d_pitch_dR21 * dR21_dw0 + d_pitch_dR22 * dR22_dw0,
                d_pitch_dR21 * dR21_dw1 + d_pitch_dR22 * dR22_dw1,
                d_pitch_dR21 * dR21_dw2 + d_pitch_dR22 * dR22_dw2
            ], dim=-1)  # [num_init_pose, 3]

            # 获取 GT 值（如果提供），并转换为弧度
            if gt_roll is not None:
                # 转换为弧度并处理不同输入类型
                if torch.is_tensor(gt_roll):
                    gt_roll_val = gt_roll.detach().to(device=pose.device, dtype=torch.float64)
                    if gt_roll_val.ndim == 0:
                        gt_roll_val = gt_roll_val.unsqueeze(0)
                    gt_roll_rad = torch.deg2rad(gt_roll_val)
                elif isinstance(gt_roll, np.ndarray):
                    gt_roll_val = np.reshape(gt_roll, -1)[0]
                    gt_roll_rad = torch.deg2rad(torch.tensor([float(gt_roll_val)], device=pose.device, dtype=torch.float64))
                else:
                    gt_roll_rad = torch.deg2rad(torch.tensor([float(gt_roll)], device=pose.device, dtype=torch.float64))
            else:
                gt_roll_rad = torch.zeros(1, device=pose.device, dtype=torch.float64)
                print(f"gt_roll is None")
            
            if gt_pitch is not None:
                # 转换为弧度并处理不同输入类型
                if torch.is_tensor(gt_pitch):
                    gt_pitch_val = gt_pitch.detach().to(device=pose.device, dtype=torch.float64)
                    if gt_pitch_val.ndim == 0:
                        gt_pitch_val = gt_pitch_val.unsqueeze(0)
                    gt_pitch_rad = torch.deg2rad(gt_pitch_val)
                elif isinstance(gt_pitch, np.ndarray):
                    gt_pitch_val = np.reshape(gt_pitch, -1)[0]
                    gt_pitch_rad = torch.deg2rad(torch.tensor([float(gt_pitch_val)], device=pose.device, dtype=torch.float64))
                else:
                    gt_pitch_rad = torch.deg2rad(torch.tensor([float(gt_pitch)], device=pose.device, dtype=torch.float64))
            else:
                gt_pitch_rad = torch.zeros(1, device=pose.device, dtype=torch.float64)
                print(f"gt_pitch is None")

            # 残差是 (roll - gt_roll) 和 (pitch - gt_pitch)
            # 确保 gt_roll_rad 和 gt_pitch_rad 可以广播到 num_init_pose
            if gt_roll_rad.numel() == 1:
                gt_roll_rad = gt_roll_rad.expand(num_init_pose)
            if gt_pitch_rad.numel() == 1:
                gt_pitch_rad = gt_pitch_rad.expand(num_init_pose)
            
            angle_r = torch.stack([roll_base - gt_roll_rad, pitch_base - gt_pitch_rad], dim=-1)  # [num_init_pose, 2]
            J_angle_rot = torch.stack([J_roll, J_pitch], dim=1)  # [num_init_pose, 2, 3]

            # 转换为与 J 相同的 dtype
            angle_res = angle_r.to(dtype=J.dtype)  # [num_init_pose, 2]
            J_angle_rot = J_angle_rot.to(dtype=J.dtype)  # [num_init_pose, 2, 3]
            
            # 扩展 J_angle 到完整的 6 自由度（后3个平移分量为0）
            J_angle_full = torch.zeros(num_init_pose, 2, 6, device=J.device, dtype=J.dtype)
            J_angle_full[:, :, :3] = J_angle_rot  # 只有旋转部分有值
            J_angle_tensor = J_angle_full.view(1, num_init_pose, 2, 6)
            angle_res = angle_res.view(1, num_init_pose, 2)
        
        # === Step 6: 统一计算投影、深度和角度的 loss 和加权 ===
        # 投影约束的 cost
        cost_proj = (res**2).sum(-1)  # [1, num_init_pose, N]

        if depth_res is not None:
            # 深度约束的 cost
            cost_depth = depth_res ** 2  # [1, num_init_pose, 1]
            # 角度约束的 cost (roll^2 + pitch^2)
            cost_angle = (angle_res ** 2).sum(-1, keepdim=True)  # [1, num_init_pose, 1]
            # 拼接所有 cost，统一计算 loss
            cost_all = torch.cat([cost_proj, cost_depth, cost_angle], dim=-1)  # [1, num_init_pose, N+2]
            cost_all, w_loss_all, _ = self.loss_fn1(cost_all)
            # 分离投影、深度和角度的 loss 权重
            w_loss_proj = w_loss_all[..., :-2]    # [1, num_init_pose, N]
            w_loss_depth = w_loss_all[..., -2:-1] # [1, num_init_pose, 1]
            w_loss_angle = w_loss_all[..., -1:]   # [1, num_init_pose, 1]
            cost = cost_all[..., :-2]  # 返回投影部分的 cost
        else:
            cost, w_loss_proj, _ = self.loss_fn1(cost_proj)
            w_loss_depth = None
            w_loss_angle = None

        # === 投影约束的权重计算 ===
        valid_query = (valid_q & visible_q).float()
        weight_loss_proj = w_loss_proj * valid_query

        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r * mask.T.view(1, -1, 1)  # [N, C]
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [B, num_init_pose, N]
        weights_proj = weight_loss_proj * w_unc

        # === 投影约束的梯度和 Hessian ===
        grad_proj = (J * res[..., :, :, None]).sum(dim=-2)  # [1, num_init_pose, N, 6]
        grad_proj = weights_proj[:, :, :, None] * grad_proj
        grad_proj = grad_proj.sum(-2)  # [1, num_init_pose, 6]

        Hess_proj = J.permute(0, 1, 2, 4, 3) @ J  # [1, num_init_pose, N, 6, 6]
        Hess_proj = weights_proj[:, :, :, None, None] * Hess_proj
        Hess_proj = Hess_proj.sum(-3)  # [1, num_init_pose, 6, 6]

        # === 深度约束和角度约束的梯度和 Hessian（如果存在） ===
        if depth_res is not None and w_loss_depth is not None:
            # === 深度约束 ===
            weights_depth = w_loss_depth * valid_depth * depth_weight  # [1, num_init_pose, 1]
            grad_depth = J_depth_tensor * (depth_res * weights_depth)[..., None]  # [1, num_init_pose, 1, 6]
            grad_depth = grad_depth.sum(-2)  # [1, num_init_pose, 6]
            Hess_depth = J_depth_tensor[..., :, None] * J_depth_tensor[..., None, :]  # [1, num_init_pose, 1, 6, 6]
            Hess_depth = weights_depth[..., None, None] * Hess_depth
            Hess_depth = Hess_depth.sum(-3)  # [1, num_init_pose, 6, 6]

            # === 角度约束 ===
            weights_angle = w_loss_angle * angle_weight  # [1, num_init_pose, 1]
            grad_angle = torch.einsum("bnij,bni->bnj", J_angle_tensor, angle_res * weights_angle)  # [1, num_init_pose, 6]
            Hess_angle = torch.einsum("bnri,bnrj->bnij", J_angle_tensor, J_angle_tensor) * weights_angle[..., None]  # [1, num_init_pose, 6, 6]

            # 自适应缩放
            reproj_grad_norm = torch.norm(grad_proj)
            depth_grad_norm = torch.norm(grad_depth)
            angle_grad_norm = torch.norm(grad_angle)
            
            target_ratio = 0.2  # 让先验梯度的贡献大约是视觉梯度的 20%
            depth_scale = (reproj_grad_norm / (depth_grad_norm + 1e-8)) * target_ratio
            angle_scale = (reproj_grad_norm / (angle_grad_norm + 1e-8)) * target_ratio
            depth_scale = depth_scale.clamp(max=500.0)
            angle_scale = angle_scale.clamp(max=200.0)  # 增加上限以允许更大的缩放

            # 合并梯度和 Hessian
            grad = grad_proj + grad_depth * depth_scale + grad_angle * angle_scale
            Hess = Hess_proj + Hess_depth * depth_scale + Hess_angle * angle_scale
            w_loss = w_loss_proj  # 返回投影部分的 w_loss
            
            # 打印权重诊断信息
            if self.iter <= 3:
                print(f"\n=== [约束权重诊断] iter={self.iter} ===")
                print(f"[重投影约束]")
                print(f"  梯度范数(原始): {reproj_grad_norm.item():.6f}")
                print(f"  梯度占比: {reproj_grad_norm.item()/(torch.norm(grad).item()+1e-8)*100:.2f}%")
                
                print(f"[深度约束]")
                print(f"  残差(原始): mean={np.abs(base_residuals_raw).mean():.4f}m, max={np.abs(base_residuals_raw).max():.4f}m")
                depth_res_norm_np = depth_res.detach().cpu().numpy()
                print(f"  残差(归一化): mean={np.abs(depth_res_norm_np).mean():.6f}, sigma={depth_sigma}m")
                print(f"  有效数量: {valid_mask.sum()}/{len(valid_mask)}")
                print(f"  梯度范数(原始): {depth_grad_norm.item():.6f}")
                print(f"  梯度范数(加权后): {torch.norm(grad_depth * depth_scale).item():.6f}")
                print(f"  梯度占比: {torch.norm(grad_depth * depth_scale).item()/(torch.norm(grad).item()+1e-8)*100:.2f}% (目标: {target_ratio*100:.0f}%)")
                print(f"  缩放因子: {depth_scale.item():.4f}")
                
                print(f"[角度约束]")
                roll_base_deg = np.rad2deg(roll_base.detach().cpu().numpy())
                pitch_base_deg = np.rad2deg(pitch_base.detach().cpu().numpy())
                # 重新计算GT值（确保在诊断打印时可以访问）
                if gt_roll is not None:
                    if torch.is_tensor(gt_roll):
                        gt_roll_deg = float(gt_roll.detach().cpu().item())
                    elif isinstance(gt_roll, np.ndarray):
                        gt_roll_deg = float(gt_roll.reshape(-1)[0])
                    else:
                        gt_roll_deg = float(gt_roll)
                else:
                    gt_roll_deg = 0.0
                if gt_pitch is not None:
                    if torch.is_tensor(gt_pitch):
                        gt_pitch_deg = float(gt_pitch.detach().cpu().item())
                    elif isinstance(gt_pitch, np.ndarray):
                        gt_pitch_deg = float(gt_pitch.reshape(-1)[0])
                    else:
                        gt_pitch_deg = float(gt_pitch)
                else:
                    gt_pitch_deg = 0.0
                roll_residual_deg = roll_base_deg - gt_roll_deg
                pitch_residual_deg = pitch_base_deg - gt_pitch_deg
                print(f"  GT值: roll={gt_roll_deg:.4f}°, pitch={gt_pitch_deg:.4f}°")
                print(f"  roll残差(原始): mean={np.abs(roll_residual_deg).mean():.4f}°, max={np.abs(roll_residual_deg).max():.4f}°")
                print(f"  pitch残差(原始): mean={np.abs(pitch_residual_deg).mean():.4f}°, max={np.abs(pitch_residual_deg).max():.4f}°")
                angle_res_np = angle_res.detach().cpu().numpy()
                print(f"  残差(弧度): mean={np.abs(angle_res_np).mean():.6f}rad")
                print(f"  有效数量: {num_init_pose}/{num_init_pose}")
                print(f"  梯度范数(原始): {angle_grad_norm.item():.6f}")
                print(f"  梯度范数(加权后): {torch.norm(grad_angle * angle_scale).item():.6f}")
                print(f"  梯度占比: {torch.norm(grad_angle * angle_scale).item()/(torch.norm(grad).item()+1e-8)*100:.2f}% (目标: {target_ratio*100:.0f}%)")
                print(f"  缩放因子: {angle_scale.item():.4f}")
                
                print(f"[总体]")
                print(f"  总梯度范数: {torch.norm(grad).item():.6f}")
                print(f"  比例: 重投影:{reproj_grad_norm.item()/(torch.norm(grad).item()+1e-8)*100:.1f}% | 深度:{torch.norm(grad_depth * depth_scale).item()/(torch.norm(grad).item()+1e-8)*100:.1f}% | 角度:{torch.norm(grad_angle * angle_scale).item()/(torch.norm(grad).item()+1e-8)*100:.1f}%")
                print(f"=== [权重诊断结束] ===\n", flush=True)
        else:
            grad = grad_proj
            Hess = Hess_proj
            w_loss = w_loss_proj
            
            # 打印权重诊断信息
            # if self.iter <= 3:
            #     print(f"\n=== [深度约束权重诊断] iter={self.iter} ===")
            #     print(f"[残差信息]")
            #     print(f"  深度残差(原始): mean={np.abs(base_residuals_raw).mean():.4f}m, max={np.abs(base_residuals_raw).max():.4f}m")
            #     print(f"  深度残差(归一化): mean={np.abs(base_residuals).mean():.6f}, max={np.abs(base_residuals).max():.6f}, sigma={depth_sigma}m")
            #     print(f"  有效深度约束数量: {valid_mask.sum()}/{len(valid_mask)}")
                
            #     print(f"[权重信息]")
            #     print(f"  depth_weight (基础权重): {depth_weight:.2f}")
            #     print(f"  w_loss (鲁棒核权重): mean={w_loss.mean().item():.6f}, max={w_loss.max().item():.6f}, min={w_loss.min().item():.6f}")
            #     print(f"  最终权重 (w_loss * depth_weight): mean={(w_loss * depth_weight).mean().item():.6f}, max={(w_loss * depth_weight).max().item():.6f}")
                
            #     print(f"[雅可比信息]")
            #     print(f"  J_depth范数: {np.linalg.norm(J_depth):.4f}")
            #     print(f"  J_depth均值: {np.abs(J_depth).mean():.6f}, 最大值: {np.abs(J_depth).max():.6f}")
                
            #     print(f"[梯度信息]")
            #     print(f"  重投影梯度范数: {reproj_grad_norm.item():.4f}")
            #     print(f"  深度梯度范数(加权前): {torch.norm(grad_depth).item():.4f}")
            #     print(f"  自适应缩放因子 (depth_scale): {depth_scale.item():.4f}")
            #     print(f"  深度梯度范数(加权后): {torch.norm(grad_depth * depth_scale).item():.4f}")
            #     print(f"  深度梯度占比: {torch.norm(grad_depth * depth_scale).item()/(torch.norm(grad).item()+1e-8)*100:.2f}% (目标: {target_ratio*100:.0f}%)")
                
            #     print(f"[Hessian信息]")
            #     print(f"  重投影Hessian范数: {torch.norm(Hess_reproj_only).item():.4f}")
            #     print(f"  深度Hessian范数(加权后): {torch.norm(Hess_depth * depth_scale).item():.4f}")
            #     print(f"  深度Hessian占比: {torch.norm(Hess_depth * depth_scale).item()/(torch.norm(Hess).item()+1e-8)*100:.2f}%")
                
            #     print(f"=== [权重诊断结束] ===\n", flush=True)
            # grad = grad + grad_angle * angle_scale * angle_gate
            # Hess = Hess + Hess_angle * angle_scale * angle_gate
        
        return -grad, Hess, w_loss, valid_query, p2d_q, cost   # torch.Size([64, 6]) , torch.Size([64, 6, 6]), torch.Size([64, 447]), torch.Size([64, 447])
    
