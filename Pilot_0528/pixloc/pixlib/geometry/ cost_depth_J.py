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
        #  inputs = {
        #     "pose_data_q": pose_data_q,
        #     "f_r": f_r,
        #     "pose_data_r": pose_data_r,
        #     "cam_data_r": cam_data_r,
        #     "f_q": f_q,
        #     "cam_data_q": cam_data_q,
        #     "p3D": p3D,
        #     "c_ref": c_ref,
        #     "c_query": c_query
        # }
        # if cam_data_r[0][0][0] == 128:
        #     torch.save(inputs, "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_128_1.pt")
        # elif cam_data_r[0][0][0] == 256:
        #     torch.save(inputs, "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_256_1.pt")
        # elif cam_data_r[0][0][0] == 512:
        #     torch.save(inputs, "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_512_1.pt")
        # elif cam_data_r[0][0][0] == 64:
        #     torch.save(inputs, "/home/liuxy24/code/FPV_dev_quat_long/FPV-Test/FPV-Test/sample_inputs_64_1.pt")
        num_init_pose = pose_data_q.shape[1]
        # Use the actual camera intrinsics for the center-ray computation.
        cam_K = None
        if cam_data_q is not None and torch.is_tensor(cam_data_q) and cam_data_q.numel() >= 6:
            cam_K = cam_data_q[0, 0, :6].detach().cpu().tolist()
        # === Step 1: 参考帧中投影、采样、可见性判断 ===       
        # print("p2d_r: ", p2d_r.shape)
        # p3d_r = transform_p3d(pose_data_r, p3D)
        # print("p3d_r: ", p3d_r)
        # print("查询  ",cam_data_q[0, 0, :2])
        # p2d_r, visible_r = project_p3d(cam_data_r, p3d_r)
        # print("cam_data_r: ", cam_data_r[0, 0, :2])
        # print("f_r: ", f_r.shape)
        # print("p2d_r: ", p2d_r.shape)
        # print("visible_r: ", visible_r.shape)
        # p2d_r = p2d_r[0]
        # visible_r = visible_r[0]
        # print("p2d_r1: ", p2d_r.shape)
        # print("visible_r1: ", visible_r.shape)
        # print("p2d_r: ", p2d_r)
        # print("visible_r: ", visible_r)
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
        print("res: ", res.shape)
        print("fp_q: ", fp_q.shape)
        print("fp_r: ", fp_r.shape)

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

        # === Step 5: loss 和加权 ===
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn1(cost)

        valid_query = (valid_q & visible_q).float()
        weight_loss = w_loss * valid_query

        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r *  mask.T.view(1, -1, 1)  # [N, C]
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [B, N]
        weights = weight_loss * w_unc

        grad = (J * res[..., :, :, None]).sum(dim=-2)
        grad = weights[:, :, :, None] * grad
        grad = grad.sum(-2)

        Hess = J.permute(0, 1, 2, 4, 3) @ J
        Hess = weights[:, :, :, None, None] * Hess
        Hess = Hess.sum(-3)

        # === Step 6: Soft depth + angle priors (adaptive scaling) ===
        if (
            render_T_ecef is not None
            and gt_depth is not None
            and dsm_path is not None
            and hasattr(self, "last_wgs84_poses")
        ):
            depth_weight = 3.0
            depth_truncate = 5.0
            eps_pos = 1e-4   # degrees for lon/lat
            eps_alt = 0.5    # meters
            eps_ang = 1e-3   # degrees

            grad_depth = torch.zeros_like(grad)
            Hess_depth = torch.zeros_like(Hess)

            # Compute Jacobian in the same local SE(3) parameterization as LM.
            pose_batches = []
            for i in range(num_init_pose):
                base_pose = Pose(pose_data_q[0, i].unsqueeze(0))
                pose_list = [base_pose._data]
                for j in range(6):
                    delta_aa = torch.zeros(1, 3, device=pose_data_q.device, dtype=pose_data_q.dtype)
                    delta_t = torch.zeros(1, 3, device=pose_data_q.device, dtype=pose_data_q.dtype)
                    if j < 3:
                        delta_aa[0, j] = eps_ang
                    else:
                        delta_t[0, j - 3] = eps_alt
                    delta_pose = Pose.from_aa(delta_aa, delta_t)
                    pert_pose = base_pose.compose(delta_pose)
                    pose_list.append(pert_pose._data)
                pose_batches.append(torch.stack(pose_list, dim=1))

            pose_batch = torch.cat(pose_batches, dim=1)
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
            centers_np = np.asarray(centers_batch, dtype=np.float64)
            dsm_np = np.asarray(dsm_heights_batch, dtype=np.float64)
            # Use base-pose DSM height for all perturbations to avoid DSM slope
            # dominating the Jacobian direction.
            base_dsm = dsm_np.reshape(num_init_pose, 7)[:, 0]
            base_dsm_rep = np.repeat(base_dsm, 7)
            residuals_batch = centers_np[:, 2] - base_dsm_rep
            slope_delta = dsm_np - base_dsm_rep
            # if slope_delta.size > 0:
            #     print(
            #         "[depth_prior] dsm slope delta mean/std/max:",
            #         float(np.mean(slope_delta)),
            #         float(np.std(slope_delta)),
            #         float(np.max(np.abs(slope_delta))),
            #     )
            # if len(wgs84_batch) > 0:
            #     base_pose = wgs84_batch[0]
            #     cam_dsm = self._query_dsm_heights([base_pose], dsm_path, use_gpu=self.use_gpu_dsm)[0]
            #     cam_alt = base_pose[2]
            #     vertical_depth = cam_alt - cam_dsm
            #     print(
            #         "[depth_prior] cam_alt/dsm/vertical_depth/gt_depth:",
            #         float(cam_alt),
            #         float(cam_dsm),
            #         float(vertical_depth),
            #         float(gt_depth),
            #     )
            residuals_batch = residuals_batch.reshape(num_init_pose, 7)
            base_residuals = residuals_batch[:, 0]
            valid_mask = ~np.isnan(base_residuals)
            # if num_init_pose > 0:
            #     print(
            #         "[depth_prior] pose0 residuals (base, rx, ry, rz, tx, ty, tz):",
            #         residuals_batch[0].tolist(),
            #     )
            #     if len(centers_batch) >= 7:
            #         print(
            #             "[depth_prior] pose0 center (base/tx/tz):",
            #             centers_batch[0],
            #             centers_batch[4],
            #             centers_batch[6],
            #         )

            denoms = np.array([eps_pos, eps_pos, eps_alt, eps_ang, eps_ang, eps_ang], dtype=np.float64)
            pert_residuals = residuals_batch[:, 1:7]
            J_depth = (pert_residuals - base_residuals[:, None]) / denoms[None, :]
            J_depth = np.nan_to_num(J_depth, nan=0.0)
            # if num_init_pose > 0:
            #     print("[depth_prior] pose0 J_depth:", J_depth[0].tolist())

            base_residuals = np.nan_to_num(base_residuals, nan=0.0)
            valid_res = base_residuals[valid_mask]
            # if valid_res.size > 0:
            #     print(
            #         "[depth_prior] residuals mean/std/min/max:",
            #         float(np.mean(valid_res)),
            #         float(np.std(valid_res)),
            #         float(np.min(valid_res)),
            #         float(np.max(valid_res)),
            #     )
            # else:
            #     print("[depth_prior] residuals: no valid points")
            # print(
            #     "[depth_prior] params weight/truncate/eps_pos/eps_alt/mul:",
            #     float(depth_weight),
            #     float(depth_truncate),
            #     float(eps_pos),
            #     float(eps_alt),
            #     float(mul) if mul is not None else None,
            # )
            # print(
            #     "[depth_prior] valid poses:",
            #     int(np.sum(valid_mask)),
            #     "/",
            #     int(num_init_pose),
            # )
            # Depth scale (meters): normalize residuals/J to control prior strength.
            depth_sigma = 100.0
            base_residuals = base_residuals / depth_sigma
            J_depth = J_depth / depth_sigma
            base_residuals_t = torch.from_numpy(base_residuals).to(device=grad.device, dtype=grad.dtype)
            cost_tensor = base_residuals_t ** 2
            _, w_loss, _ = self.loss_fn1(cost_tensor, alpha=0.0, truncate=depth_truncate)
            w_loss = w_loss * torch.from_numpy(valid_mask.astype(np.float32)).to(device=grad.device, dtype=grad.dtype)

            J_depth_torch = torch.from_numpy(J_depth).to(device=grad.device, dtype=grad.dtype)
            weighted = (base_residuals_t * w_loss * depth_weight).unsqueeze(-1)
            grad_depth = J_depth_torch * weighted

            w_hess = (w_loss * depth_weight).view(-1, 1, 1)
            Hess_depth = J_depth_torch[:, :, None] * J_depth_torch[:, None, :] * w_hess

            # === Step 7: Soft angle prior (roll/pitch -> 0) ===
            angle_weight = 0.0
            eps_rot = 1e-3  # radians
            pose_batches = []
            for i in range(num_init_pose):
                base = pose_data_q[:, i:i + 1, :]
                pose_batches.append(base)
                R_base = base[..., :9].view(3, 3)
                for axis in range(3):
                    delta = torch.zeros(3, device=pose_data_q.device, dtype=pose_data_q.dtype)
                    delta[axis] = eps_rot
                    skew = skew_symmetric(delta).squeeze(0)
                    R_pert = R_base @ (torch.eye(3, device=pose_data_q.device, dtype=pose_data_q.dtype) + skew)
                    pert = base.clone()
                    pert[..., :9] = R_pert.reshape(1, 1, 9)
                    pose_batches.append(pert)

            pose_batch = torch.cat(pose_batches, dim=1)
            if origin is not None and dd is not None:
                t_c2w_ecef_batch = self._pose_translations_ecef(pose_batch, origin, dd)
                wgs84_batch = self._poses_to_wgs84(pose_batch, t_c2w_ecef_batch)
            else:
                wgs84_batch = self._poses_to_wgs84(pose_batch, render_T_ecef)
            wgs84_np = np.asarray(wgs84_batch, dtype=np.float64)
            roll_all = np.deg2rad(wgs84_np[:, 3])
            pitch_all = np.deg2rad(wgs84_np[:, 4])
            roll_all = roll_all.reshape(num_init_pose, 4)
            pitch_all = pitch_all.reshape(num_init_pose, 4)

            roll_base = roll_all[:, 0]
            pitch_base = pitch_all[:, 0]
            roll_pert = roll_all[:, 1:]
            pitch_pert = pitch_all[:, 1:]

            J_roll = (roll_pert - roll_base[:, None]) / eps_rot
            J_pitch = (pitch_pert - pitch_base[:, None]) / eps_rot
            r = np.stack([roll_base, pitch_base], axis=1)
            J = np.stack([J_roll, J_pitch], axis=1)  # (N, 2, 3)

            r_t = torch.from_numpy(r).to(device=grad.device, dtype=grad.dtype)
            J_t = torch.from_numpy(J).to(device=grad.device, dtype=grad.dtype)
            grad_rot = torch.einsum("nij,ni->nj", J_t, r_t) * angle_weight
            Hess_rot = torch.einsum("nri,nrj->nij", J_t, J_t) * angle_weight

            # Prior cost for candidate selection (normalized).
            num_points = max(cost.shape[-1], 1)
            valid_mask_t = torch.from_numpy(valid_mask.astype(np.float32)).to(device=grad.device, dtype=grad.dtype)
            prior_cost = (base_residuals_t ** 2) / float(num_points)
            self.last_prior_cost = prior_cost * valid_mask_t

            grad_angle = torch.zeros_like(grad)
            Hess_angle = torch.zeros_like(Hess)
            grad_angle[:, :, :3] = grad_rot
            Hess_angle[:, :, :3, :3] = Hess_rot

            # Adaptive scaling so priors do not overwhelm reprojection.
            reproj_grad_norm = torch.norm(grad)
            depth_grad_norm = torch.norm(grad_depth)
            angle_grad_norm = torch.norm(grad_angle)
            # Keep depth contribution below reprojection to avoid dominance.
            target_ratio = 0.2
            depth_scale = (reproj_grad_norm / (depth_grad_norm + 1e-8)) * target_ratio
            angle_scale = (reproj_grad_norm / (angle_grad_norm + 1e-8)) * target_ratio
            # Allow depth_scale to reach the target_ratio when reproj is dominant.
            depth_scale = depth_scale.clamp(max=500.0)
            angle_scale = angle_scale.clamp(max=10.0)

            # Gate priors when they conflict with reprojection.
            depth_cos = (
                torch.sum(grad_depth * grad) /
                ((depth_grad_norm + 1e-8) * (reproj_grad_norm + 1e-8))
            ).item()
            # Apply depth prior only when its direction agrees with reprojection.
            depth_gate = 1.0 
            angle_gate = 1.0 if angle_scale.item() > 1e-3 else 0.0
            # print(
            #     "[depth_prior] cos/gate/scale:",
            #     float(depth_cos),
            #     float(depth_gate),
            #     float(depth_scale),
            # )
            # print(
            #     "[depth_prior] grad_norms reproj/depth:",
            #     float(reproj_grad_norm),
            #     float(depth_grad_norm),
            # )
            # eff_depth = depth_grad_norm * depth_scale * depth_gate
            # eff_ratio = eff_depth / (reproj_grad_norm + 1e-8)
            # print("[depth_prior] eff_depth/reproj:", float(eff_ratio))

            grad = grad + grad_depth * depth_scale * depth_gate
            Hess = Hess + Hess_depth * depth_scale * depth_gate
            grad = grad + grad_angle * angle_scale * angle_gate
            Hess = Hess + Hess_angle * angle_scale * angle_gate
        
        return -grad, Hess, w_loss, valid_query, p2d_q, cost   # torch.Size([64, 6]) , torch.Size([64, 6, 6]), torch.Size([64, 447]), torch.Size([64, 447])
    
