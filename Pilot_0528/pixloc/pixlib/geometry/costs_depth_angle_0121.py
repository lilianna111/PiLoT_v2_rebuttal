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
                lon_min = torch.tensor(cache["lon_min"], device="cuda", dtype=torch.float32)
                lon_scale = torch.tensor(cache["lon_max"] - cache["lon_min"], device="cuda", dtype=torch.float32)
                lat_min = torch.tensor(cache["lat_min"], device="cuda", dtype=torch.float32)
                lat_scale = torch.tensor(cache["lat_max"] - cache["lat_min"], device="cuda", dtype=torch.float32)
                gpu_cache = {
                    "map_tensor": map_tensor,
                    "lon_min": lon_min,
                    "lon_scale": lon_scale,
                    "lat_min": lat_min,
                    "lat_scale": lat_scale,
                }
                cache["gpu"] = gpu_cache

            points_array = np.asarray(wgs84_points, dtype=np.float32)
            lons = torch.from_numpy(points_array[:, 0]).to("cuda", dtype=torch.float32)
            lats = torch.from_numpy(points_array[:, 1]).to("cuda", dtype=torch.float32)
            valid = (lons >= cache["lon_min"]) & (lons <= cache["lon_max"]) & (lats >= cache["lat_min"]) & (lats <= cache["lat_max"])

            u = (lons - gpu_cache["lon_min"]) / gpu_cache["lon_scale"]
            v = (lats - gpu_cache["lat_min"]) / gpu_cache["lat_scale"]
            grid_u = u * 2.0 - 1.0
            grid_v = (1.0 - v) * 2.0 - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).view(1, -1, 1, 2)
            sample = F.grid_sample(gpu_cache["map_tensor"], grid, align_corners=False, padding_mode="border")
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
        c_ref,c_query,p2d_r,visible_r, gt_depth=None, dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None):


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
        angle_weight = 1.0        # 角度约束基础权重
        depth_sigma = 100.0       # 深度归一化因子
        eps_alt = 0.5             # meters
        eps_ang = 1e-3            # degrees
        eps_rot = 1e-3            # radians for angle Jacobian

        if (
            render_T_ecef is not None
            and gt_depth is not None
            and dsm_path is not None
        ):
            # 构建扰动位姿 - 批量并行化
            # 构建6个扰动方向的 delta_aa 和 delta_t [6, 3]
            delta_aa_all = torch.zeros(6, 3, device=pose_data_q.device, dtype=pose_data_q.dtype)
            delta_t_all = torch.zeros(6, 3, device=pose_data_q.device, dtype=pose_data_q.dtype)
            delta_aa_all[0, 0] = eps_ang  # rot x
            delta_aa_all[1, 1] = eps_ang  # rot y
            delta_aa_all[2, 2] = eps_ang  # rot z
            delta_t_all[3, 0] = eps_alt   # trans x
            delta_t_all[4, 1] = eps_alt   # trans y
            delta_t_all[5, 2] = eps_alt   # trans z

            # 基础位姿 [num_init_pose, 12]
            base_poses = Pose(pose_data_q[0])  # [num_init_pose]
            
            # 批量构建扰动位姿
            pose_list_all = [base_poses._data]  # 原始位姿
            for j in range(6):
                delta_pose = Pose.from_aa(
                    delta_aa_all[j:j+1].expand(num_init_pose, -1),
                    delta_t_all[j:j+1].expand(num_init_pose, -1)
                )
                pert_poses = base_poses.compose(delta_pose)
                pose_list_all.append(pert_poses._data)
            
            
            pose_batch = torch.stack(pose_list_all, dim=1).view(1, num_init_pose * 7, -1)
            if origin is not None and dd is not None:
                t_c2w_ecef_batch = self._pose_translations_ecef(pose_batch, origin, dd)
                wgs84_batch = self._poses_to_wgs84(pose_batch, t_c2w_ecef_batch)
            else:
                wgs84_batch = self._poses_to_wgs84(pose_batch, render_T_ecef)
            if origin is not None and dd is not None:
                centers_batch = self._center_points_from_pose_ecef(
                    pose_batch, gt_depth, K=cam_K, origin=origin, dd=dd, mul=mul
                )
            else:
                centers_batch = self._center_points_from_wgs84_poses(
                    wgs84_batch, gt_depth, K=cam_K, mul=mul
                )
            dsm_heights_batch = self._query_dsm_heights(
                centers_batch, dsm_path, use_gpu=self.use_gpu_dsm
            )

            # 计算深度残差
            centers_np = np.asarray(centers_batch, dtype=np.float64)
            dsm_np = np.asarray(dsm_heights_batch, dtype=np.float64)
            # 使用 Base Pose 的 DSM 高度来避免 DSM 斜坡主导 Jacobian 方向
            base_dsm = dsm_np.reshape(num_init_pose, 7)[:, 0]
            base_dsm_rep = np.repeat(base_dsm, 7)
            # 残差 = 估计高度(Z) - 地面高度(DSM)
            residuals_batch = centers_np[:, 2] - base_dsm_rep

            residuals_batch = residuals_batch.reshape(num_init_pose, 7)
            base_residuals = residuals_batch[:, 0]      # 原始位姿的残差
            valid_mask = ~np.isnan(base_residuals)      # 有效掩码，排除 DSM 无值的点

            # 计算 Jacobian，有限差分公式
            eps_pos = 1e-4  # degrees for lon/lat
            denoms = np.array([eps_pos, eps_pos, eps_alt, eps_ang, eps_ang, eps_ang], dtype=np.float64)
            pert_residuals = residuals_batch[:, 1:7]
            # J_depth: [Num_Pose, 6] -> 每个自由度对残差的影响率
            J_depth_np = (pert_residuals - base_residuals[:, None]) / denoms[None, :]
            # 数值稳定性处理
            J_depth_np = np.nan_to_num(J_depth_np, nan=0.0)
            base_residuals = np.nan_to_num(base_residuals, nan=0.0)
            
            # 归一化残差和 Jacobian 来控制先验强度
            base_residuals = base_residuals / depth_sigma
            J_depth_np = J_depth_np / depth_sigma

            # 转换为统一格式 [1, num_init_pose, 1]
            depth_res = torch.from_numpy(base_residuals).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)
            J_depth_tensor = torch.from_numpy(J_depth_np).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1, 6)
            valid_depth = torch.from_numpy(valid_mask.astype(np.float32)).to(device=J.device, dtype=J.dtype).view(1, num_init_pose, 1)

            # === 角度约束：roll 和 pitch 的 GT 为 0 ===
            # 构建角度扰动位姿（只扰动旋转的3个自由度）
            angle_pose_batches = []
            for i in range(num_init_pose):
                base = pose_data_q[:, i:i + 1, :]
                angle_pose_batches.append(base)
                R_base = base[..., :9].view(3, 3)
                for axis in range(3):
                    delta = torch.zeros(3, device=pose_data_q.device, dtype=pose_data_q.dtype)
                    delta[axis] = eps_rot
                    skew = skew_symmetric(delta).squeeze(0)
                    R_pert = R_base @ (torch.eye(3, device=pose_data_q.device, dtype=pose_data_q.dtype) + skew)
                    pert = base.clone()
                    pert[..., :9] = R_pert.reshape(1, 1, 9)
                    angle_pose_batches.append(pert)

            angle_pose_batch = torch.cat(angle_pose_batches, dim=1)
            # 转换到 WGS84 获取 roll/pitch
            if origin is not None and dd is not None:
                t_c2w_ecef_angle = self._pose_translations_ecef(angle_pose_batch, origin, dd)
                wgs84_angle = self._poses_to_wgs84(angle_pose_batch, t_c2w_ecef_angle)
            else:
                wgs84_angle = self._poses_to_wgs84(angle_pose_batch, render_T_ecef)
            
            wgs84_np = np.asarray(wgs84_angle, dtype=np.float64)
            roll_all = np.deg2rad(wgs84_np[:, 3])   # roll in radians
            pitch_all = np.deg2rad(wgs84_np[:, 4])  # pitch in radians
            roll_all = roll_all.reshape(num_init_pose, 4)
            pitch_all = pitch_all.reshape(num_init_pose, 4)

            roll_base = roll_all[:, 0]
            pitch_base = pitch_all[:, 0]
            roll_pert = roll_all[:, 1:]
            pitch_pert = pitch_all[:, 1:]

            # 计算 Jacobian: d(roll)/d(rotation), d(pitch)/d(rotation)
            J_roll = (roll_pert - roll_base[:, None]) / eps_rot    # [num_init_pose, 3]
            J_pitch = (pitch_pert - pitch_base[:, None]) / eps_rot # [num_init_pose, 3]

            # 残差是 roll 和 pitch（GT 为 0）
            angle_r = np.stack([roll_base, pitch_base], axis=1)  # [num_init_pose, 2]
            J_angle = np.stack([J_roll, J_pitch], axis=1)        # [num_init_pose, 2, 3]

            # 转换为 torch tensor
            angle_res = torch.from_numpy(angle_r).to(device=J.device, dtype=J.dtype)  # [num_init_pose, 2]
            J_angle_rot = torch.from_numpy(J_angle).to(device=J.device, dtype=J.dtype)  # [num_init_pose, 2, 3]
            
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
            # 深度约束的权重：使用 loss 权重 * 有效掩码 * 基础权重
            weights_depth = w_loss_depth * valid_depth * depth_weight  # [1, num_init_pose, 1]

            # 深度约束的梯度: J^T * W * r
            # J_depth_tensor: [1, num_init_pose, 1, 6], depth_res: [1, num_init_pose, 1]
            grad_depth = J_depth_tensor * (depth_res * weights_depth)[..., None]  # [1, num_init_pose, 1, 6]
            grad_depth = grad_depth.sum(-2)  # [1, num_init_pose, 6]

            # 深度约束的 Hessian: J^T * W * J
            # J_depth_tensor: [1, num_init_pose, 1, 6] -> 外积得到 [1, num_init_pose, 1, 6, 6]
            Hess_depth = J_depth_tensor[..., :, None] * J_depth_tensor[..., None, :]  # [1, num_init_pose, 1, 6, 6]
            Hess_depth = weights_depth[..., None, None] * Hess_depth
            Hess_depth = Hess_depth.sum(-3)  # [1, num_init_pose, 6, 6]

            # === 角度约束 ===
            # 角度约束的权重
            weights_angle = w_loss_angle * angle_weight  # [1, num_init_pose, 1]

            # 角度约束的梯度: J^T * W * r
            # J_angle_tensor: [1, num_init_pose, 2, 6], angle_res: [1, num_init_pose, 2]
            grad_angle = torch.einsum("bnij,bni->bnj", J_angle_tensor, angle_res * weights_angle)  # [1, num_init_pose, 6]

            # 角度约束的 Hessian: J^T * W * J
            Hess_angle = torch.einsum("bnri,bnrj->bnij", J_angle_tensor, J_angle_tensor) * weights_angle[..., None]  # [1, num_init_pose, 6, 6]

            # 自适应缩放，避免深度/角度先验主导重投影
            reproj_grad_norm = torch.norm(grad_proj)      # 视觉重投影梯度范数
            depth_grad_norm = torch.norm(grad_depth)      # 深度梯度范数
            angle_grad_norm = torch.norm(grad_angle)      # 角度梯度范数
            
            # 保持深度/角度贡献小于重投影，避免主导
            target_ratio = 0.2  # 让先验梯度的贡献大约是视觉梯度的 20%
            depth_scale = (reproj_grad_norm / (depth_grad_norm + 1e-8)) * target_ratio
            angle_scale = (reproj_grad_norm / (angle_grad_norm + 1e-8)) * target_ratio
            depth_scale = depth_scale.clamp(max=500.0)  # 限制上限，防止视觉纹理极弱时权重爆炸
            angle_scale = angle_scale.clamp(max=10.0)   # 角度约束上限稍小

            # 合并梯度和 Hessian
            grad = grad_proj + grad_depth * depth_scale + grad_angle * angle_scale
            Hess = Hess_proj + Hess_depth * depth_scale + Hess_angle * angle_scale
            w_loss = w_loss_proj  # 返回投影部分的 w_loss
        else:
            grad = grad_proj
            Hess = Hess_proj
            w_loss = w_loss_proj
        
        return -grad, Hess, w_loss, valid_query, p2d_q, cost   # torch.Size([64, 6]) , torch.Size([64, 6, 6]), torch.Size([64, 447]), torch.Size([64, 447])
    
