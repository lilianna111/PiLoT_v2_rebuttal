import torch
import torch.nn.functional as F
import numpy as np
from osgeo import gdal
import pyproj
from typing import Optional, Tuple
from torch import Tensor
from scipy.spatial.transform import Rotation as R

from . import Pose, Camera
from .optimization import J_normalization, skew_symmetric
from .interpolation import Interpolator
from ...utils.transform import pixloc_to_osg, ECEF_to_WGS84
from ...localization.base_refiner import build_c2w_batch

# ==========================================
# [新增] GPU 加速几何工具类 (PixLocGPU)
# ==========================================
class PixLocGPU:
    @staticmethod
    def ECEF_to_WGS84_tensor(xyz, device='cuda'):
        """
        [GPU版] ECEF -> WGS84 (lon, lat, alt)
        使用迭代法，避免闭解法中的 NaN 风险
        """
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = f * (2 - f)
        
        # 1. 计算经度 Lon
        lon_rad = torch.atan2(y, x)
        
        # 2. 计算纬度 Lat (迭代法，5次迭代足以收敛)
        p = torch.sqrt(x**2 + y**2)
        lat_rad = torch.atan2(z, p * (1 - e2)) # 初始猜测
        h = torch.zeros_like(lat_rad)
        
        for _ in range(5):
            sin_lat = torch.sin(lat_rad)
            N = a / torch.sqrt(1 - e2 * sin_lat**2 + 1e-9)
            h = p / torch.cos(lat_rad) - N
            lat_rad = torch.atan2(z, p * (1 - e2 * N / (N + h + 1e-9)))

        rtod = 180.0 / torch.pi
        return torch.stack([lon_rad * rtod, lat_rad * rtod, h], dim=-1)

    @staticmethod
    def get_rotation_enu_in_ecef_tensor(lon_deg, lat_deg):
        """[GPU版] 计算 ENU 到 ECEF 的旋转矩阵"""
        dtor = torch.pi / 180.0
        lon = lon_deg * dtor
        lat = lat_deg * dtor
        
        sin_lon = torch.sin(lon)
        cos_lon = torch.cos(lon)
        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)
        
        # ENU 向量定义
        east_x, east_y, east_z = -sin_lon, cos_lon, torch.zeros_like(lon)
        up_x, up_y, up_z = cos_lon * cos_lat, sin_lon * cos_lat, sin_lat
        north_x, north_y, north_z = -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat
        
        # 构建 (..., 3, 3) 矩阵
        R = torch.stack([
            torch.stack([east_x, north_x, up_x], dim=-1),
            torch.stack([east_y, north_y, up_y], dim=-1),
            torch.stack([east_z, north_z, up_z], dim=-1)
        ], dim=-2)
        return R

    @staticmethod
    def matrix_to_euler_xyz_tensor(R):
        """[GPU版] 旋转矩阵 -> 欧拉角 (XYZ, Degrees)"""
        sy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
        singular = sy < 1e-6
        
        x = torch.zeros_like(sy)
        y = torch.zeros_like(sy)
        z = torch.zeros_like(sy)
        
        # 非奇异
        x[~singular] = torch.atan2(R[..., 2, 1][~singular], R[..., 2, 2][~singular])
        y[~singular] = torch.atan2(-R[..., 2, 0][~singular], sy[~singular])
        z[~singular] = torch.atan2(R[..., 1, 0][~singular], R[..., 0, 0][~singular])
        
        # 奇异
        x[singular] = torch.atan2(-R[..., 1, 2][singular], R[..., 1, 1][singular])
        y[singular] = torch.atan2(-R[..., 2, 0][singular], sy[singular])
        z[singular] = 0
        
        return torch.stack([x, y, z], dim=-1) * (180.0 / torch.pi)

    @staticmethod
    def pixloc_to_osg_tensor(T_c2w_batch):
        """
        [GPU版] 核心转换函数: Matrix -> [lon, lat, h, r, p, y]
        替代了原本慢速的 CPU pixloc_to_osg
        """
        R_c2w = T_c2w_batch[..., :3, :3]
        t_c2w = T_c2w_batch[..., :3, 3]
        
        # 1. 坐标转换 ECEF -> WGS84
        t_wgs84 = PixLocGPU.ECEF_to_WGS84_tensor(t_c2w)
        lon, lat = t_wgs84[..., 0], t_wgs84[..., 1]
        
        # 2. 姿态转换
        R_ned2ecef = PixLocGPU.get_rotation_enu_in_ecef_tensor(lon, lat)
        R_ecef2enu = R_ned2ecef.transpose(-1, -2) # 转置
        R_pose_enu = torch.matmul(R_ecef2enu, R_c2w)
        
        # 3. 提取欧拉角
        euler_enu = PixLocGPU.matrix_to_euler_xyz_tensor(R_pose_enu)
        
        # 返回拼接后的 6维向量
        return torch.cat([t_wgs84, euler_enu], dim=-1)

# ==========================================
# [新增] 纯 GPU DSM 查询器 (DSM_GPU)
# ==========================================
# ==========================================
# [新增] 纯 GPU DSM 查询器 (DSM_GPU)
# ==========================================
class DSM_GPU:
    def __init__(self, dsm_path, device='cuda'):
        """
        初始化：读取 GeoTIFF，加载到 GPU，并计算 WGS84 到 像素坐标 的映射矩阵。
        """
        self.device = device
        self.dsm_path = dsm_path
        
        # print(f"[DSM_GPU] Loading DSM from: {dsm_path}")
        
        # 1. 使用 GDAL 读取数据
        ds = gdal.Open(dsm_path)
        if ds is None:
            raise FileNotFoundError(f"无法打开 DSM 文件: {dsm_path}")
            
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform() # (top_left_x, w_pixel, 0, top_left_y, 0, h_pixel)
        
        # 读取高度数据
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # 处理无效值 (NoData)
        no_data_val = band.GetNoDataValue()
        if no_data_val is not None:
            data[data == no_data_val] = 0.0
            
        # 转为 Tensor (1, 1, H, W)，并强制转换为 Float32
        self.map_tensor = torch.from_numpy(data).float().to(device).unsqueeze(0).unsqueeze(0)
        
        # 2. 计算 WGS84 (Lon, Lat) -> Normalized UV (-1, 1) 的仿射变换参数
        min_x_proj = gt[0]
        max_x_proj = gt[0] + width * gt[1]
        max_y_proj = gt[3]
        min_y_proj = gt[3] + height * gt[5]
        
        # 建立投影转换器: 4547 (CGCS2000) -> 4326 (WGS84)
        proj_to_wgs84 = pyproj.Transformer.from_crs("EPSG:4547", "EPSG:4326", always_xy=True)
        
        # 计算左下角和右上角的经纬度
        lon_min, lat_min = proj_to_wgs84.transform(min_x_proj, min_y_proj)
        lon_max, lat_max = proj_to_wgs84.transform(max_x_proj, max_y_proj)
        
        # 保存边界用于归一化 (强制转为 Float32 以匹配 map_tensor)
        self.lon_min = torch.tensor(lon_min, device=device, dtype=torch.float32)
        self.lon_scale = torch.tensor(lon_max - lon_min, device=device, dtype=torch.float32)
        self.lat_min = torch.tensor(lat_min, device=device, dtype=torch.float32)
        self.lat_scale = torch.tensor(lat_max - lat_min, device=device, dtype=torch.float32)

    def sample_height(self, lon, lat):
        """
        [GPU] 输入经纬度 Tensor，查询高度。
        """
        # === 核心修复：强制类型对齐 ===
        # 确保输入坐标的类型与 map_tensor 一致 (通常是 Float32)
        # 即使输入的 lon/lat 是 Double，这里也会被转成 Float32
        target_dtype = self.map_tensor.dtype
        if lon.dtype != target_dtype:
            lon = lon.to(target_dtype)
        if lat.dtype != target_dtype:
            lat = lat.to(target_dtype)

        # 1. 坐标归一化到 [0, 1]
        u = (lon - self.lon_min) / self.lon_scale
        v = (lat - self.lat_min) / self.lat_scale
        
        # 2. 映射到 grid_sample 的坐标系 [-1, 1]
        grid_u = u * 2.0 - 1.0
        grid_v = (1.0 - v) * 2.0 - 1.0 # 翻转 Y 轴以匹配图像坐标系
        
        # 3. 构造 Grid
        original_shape = lon.shape
        grid = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1).view(1, -1, 1, 2)
        
        # 4. 极速插值
        sample = F.grid_sample(self.map_tensor, grid, align_corners=False, padding_mode='border')
        
        return sample.view(original_shape)

# 全局单例管理 (Lazy Loading)
_dsm_gpu_instance = None
def get_dsm_gpu(dsm_path, device):
    global _dsm_gpu_instance
    if _dsm_gpu_instance is None:
        _dsm_gpu_instance = DSM_GPU(dsm_path, device)
    return _dsm_gpu_instance


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

    def residual_jacobian_batch_depth(self, pose_data, dsm_path, target_depth, dd, mul, origin):
        """
        [纯 GPU 修正版] 
        修复了逻辑错误：现在使用 Ray Marching (光线步进) 寻找真实的视线落点，
        而不是查询无人机正下方的高度。
        """
        # === 强制输入转 Float32 (防止 Double 报错) ===
        if pose_data.dtype != torch.float32: pose_data = pose_data.float()
            
        B, N, _ = pose_data.shape
        device = pose_data.device
        eps = 1e-4

        # 获取 GPU 地图
        dsm_gpu = get_dsm_gpu(dsm_path, device)

        # === 1. GPU 向量化生成微扰 (保持不变) ===
        perturbations = torch.zeros(7, 6, device=device, dtype=torch.float32)
        for i in range(6): perturbations[i+1, i] = eps 
        
        pose_data_expanded = pose_data.unsqueeze(2) 
        R_curr = pose_data_expanded[..., :9].view(B, N, 1, 3, 3) 
        t_curr = pose_data_expanded[..., 9:].view(B, N, 1, 3, 1)

        d_vecs = perturbations[:, :3]
        z = torch.zeros_like(d_vecs[:, 0])
        skew = torch.stack([z, -d_vecs[:, 2], d_vecs[:, 1], d_vecs[:, 2], z, -d_vecs[:, 0], -d_vecs[:, 1], d_vecs[:, 0], z], dim=-1).reshape(7, 3, 3)
        d_rot_mats = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0) + skew
        d_trans = perturbations[:, 3:].unsqueeze(-1)
        
        R_new = R_curr @ d_rot_mats.view(1, 1, 7, 3, 3)
        t_new = t_curr + (R_curr @ d_trans.view(1, 1, 7, 3, 1))
        
        # poses_flatten: (Total_Count, 12)
        poses_all_flat = torch.cat([R_new.flatten(start_dim=-2), t_new.flatten(start_dim=-2)], dim=-1).view(-1, 12)
        Total_Count = poses_all_flat.shape[0] # B*N*7

        # === 2. 计算光线参数 (Ray Casting Prep) ===
        # 我们需要知道每个 Pose 的：相机中心(Origin) 和 光轴方向(Direction) 在 ECEF 下的表示
        
        # 提取 R_c2w 和 t_c2w (ECEF坐标系)
        # 这里的 Pose 是 pixloc 格式 (R_w2c, t_w2c) 还是 (R_c2w, t_c2w)?
        # 你的 Pose.from_Rt 通常处理的是 w2c。
        # pixloc 内部优化的是 T_w2c (World to Camera)。
        # 所以 t_c2w (相机在世界的位置) = -R^T * t
        # Direction_c2w (光轴在世界方向) = R^T * [0,0,1]
        
        R_w2c = R_new.view(Total_Count, 3, 3)
        t_w2c = t_new.view(Total_Count, 3, 1)
        
        R_c2w = R_w2c.transpose(1, 2)
        center_ecef = -torch.bmm(R_c2w, t_w2c).squeeze(-1) # (Total, 3)
        
        # 假设相机看向 Z 轴正方向 (pixloc/colmap 定义)
        # 如果是 OpenGL 定义则是 -Z。这里假设标准 Z。
        ray_local = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).view(1, 3, 1).expand(Total_Count, -1, -1)
        dir_ecef = torch.bmm(R_c2w, ray_local).squeeze(-1) # (Total, 3)
        
        # === 3. GPU Ray Marching (光线步进) ===
        # 我们沿光轴发射一组点，检查哪个点最先钻入地下
        
        # 配置步进参数
        num_steps = 50       # 采样50个点
        min_dist = 2.0       # 最近 2米
        max_dist = 500.0     # 最远 500米
        
        # 生成采样步长 (Total, Steps)
        steps = torch.linspace(min_dist, max_dist, num_steps, device=device).unsqueeze(0).expand(Total_Count, -1)
        
        # 计算所有采样点的 ECEF 坐标: P = O + t * D
        # (Total, Steps, 3)
        sample_points_ecef = center_ecef.unsqueeze(1) + dir_ecef.unsqueeze(1) * steps.unsqueeze(2)
        
        # 将所有点转为 WGS84 (Total, Steps, 3) -> [lon, lat, alt]
        # PixLocGPU.ECEF_to_WGS84_tensor 支持任意维度
        sample_points_wgs84 = PixLocGPU.ECEF_to_WGS84_tensor(sample_points_ecef)
        
        # 提取经纬度去查 DSM
        sample_lons = sample_points_wgs84[..., 0]
        sample_lats = sample_points_wgs84[..., 1]
        sample_alts = sample_points_wgs84[..., 2] # 射线点的海拔
        
        # 查表得到地面高度 (Total, Steps)
        terrain_alts = dsm_gpu.sample_height(sample_lons, sample_lats)
        
        # === 4. 寻找交点 (Intersection) ===
        # 射线高度 < 地面高度 的点即为地下点
        # (Total, Steps)
        is_underground = sample_alts < terrain_alts
        
        # 找到第一个 True 的索引
        # argmax 会返回第一个最大值(True=1)的索引。如果全是False，返回0。
        first_hit_idx = torch.argmax(is_underground.int(), dim=1) 
        
        # 获取对应的深度
        # gather 需要 index 维度匹配
        pred_depths = torch.gather(steps, 1, first_hit_idx.unsqueeze(1)).squeeze(1)
        
        # 处理无效射线 (如果是索引0且并未真正击中，说明都在天上)
        # 检查索引0是否真的是地下
        hit_at_0 = is_underground[:, 0]
        # 如果 argmax是0 且 第0个点不在地下 -> 说明没击中 -> 设为无效深度
        invalid_mask = (first_hit_idx == 0) & (~hit_at_0)
        
        # 将没击中的设为0或当前最大值，后续会过滤
        pred_depths = torch.where(invalid_mask, torch.zeros_like(pred_depths), pred_depths)
        
        # 恢复形状 (B, N, 7)
        depth_results = pred_depths.view(B, N, 7)

        # === 5. 梯度计算 (同前) ===
        d_center = depth_results[:, :, 0]
        d_perturbed = depth_results[:, :, 1:]
        
        valid_mask = (d_center > 0).unsqueeze(-1) & (d_perturbed > 0)
        
        if torch.is_tensor(target_depth): 
            t_tensor = target_depth.to(device).float()
        else: 
            t_tensor = torch.tensor(target_depth, device=device, dtype=torch.float32)
        
        if t_tensor.numel() == 1: t_val = t_tensor.view(1, 1, 1).expand(B, N, 1)
        elif t_tensor.numel() == B * N: t_val = t_tensor.view(B, N, 1)
        elif t_tensor.numel() == B: t_val = t_tensor.view(B, 1, 1).expand(B, N, 1)
        else: 
             try: t_val = t_tensor.reshape(B, N, 1)
             except: raise RuntimeError(f"Shape mismatch")

        residual = d_center.unsqueeze(-1) - t_val
        J = (d_perturbed - d_center.unsqueeze(-1)) / eps
        
        J = torch.where(valid_mask, J, torch.zeros_like(J))
        residual = torch.where(valid_mask[..., 0:1], residual, torch.zeros_like(residual))
        
        cost = residual.squeeze(-1) ** 2
        _, w_loss, _ = self.loss_fn1(cost)
        w_loss = w_loss.unsqueeze(-1)
        
        J = torch.nan_to_num(J, nan=0.0)
        residual = torch.nan_to_num(residual, nan=0.0)
        w_loss = torch.nan_to_num(w_loss, nan=1.0)
        
        grad = J * residual.view(B, N, 1) * w_loss
        Hess = torch.einsum('bni,bnj->bnij', J, J) * w_loss.unsqueeze(-1)
        
        grad = torch.nan_to_num(grad, nan=0.0).float()
        Hess = torch.nan_to_num(Hess, nan=0.0).float()
        
        return grad, Hess

    def residual_jacobian_batch_quat(
        self, pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D: Tensor,
        c_ref,c_query,dsm_path,gt_depth, dd=None, mul=None, origin=None):
        
        num_init_pose = pose_data_q.shape[1]

        # === Step 1: 参考帧中投影、采样、可见性判断 ===
        p3d_r = transform_p3d(pose_data_r, p3D)
        p2d_r, visible_r = project_p3d(cam_data_r, p3d_r)
        p2d_r = p2d_r[0]
        visible_r = visible_r[0]
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)

        valid_ref_mask = (valid_r & visible_r).to(torch.float32)  # shape: [1, N]
        mask = valid_ref_mask.transpose(1, 0)  # shape: [N, 1]

        fp_r = fp_r[0] * mask  # [N, C]
        p3D = p3D[0] * mask    # [N, 3]

        fp_r = fp_r.repeat(1, num_init_pose, 1, 1)
        p3D = p3D.repeat(1, num_init_pose, 1, 1)
        
        fp_r = torch.nn.functional.normalize(fp_r, dim=-1)
        
        # === Step 2: 查询帧中投影 ===
        p3d_q = transform_p3d(pose_data_q, p3D)
        p2d_q, visible_q = project_p3d(cam_data_q, p3d_q)

        p2d_q = p2d_q * mask.T.view(1, 1, -1, 1)  
        
        # === Step 3: 查询帧特征采样 ===
        fp_q, valid_q, J_f = self.interpolate_feature_map(f_q, p2d_q.reshape(1, -1, 2), return_gradients=True)
        fp_q = fp_q.view(1, num_init_pose, -1, fp_q.shape[-1])
        valid_q = valid_q.view(1, num_init_pose, -1)
        J_f = J_f.view(1, num_init_pose, -1, J_f.shape[-2], J_f.shape[-1])
        visible_q = visible_q.view(1, num_init_pose, -1)
        
        res = fp_q - fp_r

        # === Step 4: Jacobians (Visual) ===
        R = pose_data_q[..., :9].view(1, -1, 3, 3)
        J_p3d_rot = R[:, :, None].matmul(-skew_symmetric(p3D))
        J_p3d_tran = R[:, :, None].expand(-1, -1, J_p3d_rot.shape[2], -1, -1)
        J_p3d_pose = torch.cat((J_p3d_rot, J_p3d_tran), dim=-1)

        fx, fy = cam_data_q[:, :, 2], cam_data_q[:, :, 3]
        zero = torch.zeros_like(fx)
        f_diag_embed = torch.stack([fx, zero, zero, fy], dim=-1).reshape(1, -1, 1, 2, 2)

        J_p2d_p3d = f_diag_embed @ J_project(p3d_q)
        J = J_f @ J_p2d_p3d @ J_p3d_pose 

        # === Step 5: Visual Loss ===
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn1(cost)

        valid_query = (valid_q & visible_q).float()
        weight_loss = w_loss * valid_query

        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r * mask.T.view(1, -1, 1) 
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1) 
        weights = weight_loss * w_unc

        # J^T * r (正梯度)
        grad = (J * res[..., :, :, None]).sum(dim=-2)
        grad = weights[:, :, :, None] * grad
        grad = grad.sum(-2)

        Hess = J.permute(0, 1, 2, 4, 3) @ J
        Hess = weights[:, :, :, None, None] * Hess
        Hess = Hess.sum(-3)

        # === Step 6: Depth Constraints (Pure GPU) ===
        w_depth = 0.5
        
        # 1. 计算深度梯度 (返回正梯度)
        grad_depth, Hess_depth = self.residual_jacobian_batch_depth(
            pose_data_q, dsm_path, gt_depth, dd, mul, origin
        )
        
        # 2. 融合 Hess
        Hess = Hess + (w_depth * Hess_depth)
        
        # 3. 融合 Grad (均为正梯度，直接相加)
        grad = grad + (w_depth * grad_depth)
        
        # 4. 返回负梯度 (-grad) 给优化器
        return -grad, Hess, w_loss, valid_query, p2d_q, cost