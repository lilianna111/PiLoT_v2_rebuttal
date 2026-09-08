import torch
import torch.nn.functional as F
import numpy as np

# ==========================================
# Part 1: 基础坐标转换函数 (PyTorch版)
# ==========================================

def ECEF_to_WGS84_torch(ecef_coords):
    """[GPU版本] ECEF (x,y,z) -> WGS84 (lon, lat, alt)"""
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    
    x, y, z = ecef_coords[..., 0], ecef_coords[..., 1], ecef_coords[..., 2]

    lam = torch.atan2(y, x) 
    p = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(z * a, p * b)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    phi = torch.atan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    N = a / torch.sqrt(1 - e2 * sin_phi**2)
    h = p / cos_phi - N
    
    rad2deg = 180.0 / 3.141592653589793
    return torch.stack([lam * rad2deg, phi * rad2deg, h], dim=-1)

def get_rotation_enu_in_ecef_torch(lon, lat):
    """[GPU版本] 计算 ENU 到 ECEF 的旋转矩阵"""
    deg2rad = 3.141592653589793 / 180.0
    lon_rad, lat_rad = lon * deg2rad, lat * deg2rad
    
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    
    up_x, up_y, up_z = cos_lat * cos_lon, cos_lat * sin_lon, sin_lat
    east_x, east_y, east_z = -sin_lon, cos_lon, torch.zeros_like(lon)
    north_x, north_y, north_z = -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat
    
    row0 = torch.stack([east_x, north_x, up_x], dim=-1)
    row1 = torch.stack([east_y, north_y, up_y], dim=-1)
    row2 = torch.stack([east_z, north_z, up_z], dim=-1)
    
    return torch.stack([row0, row1, row2], dim=-2)

def matrix_to_euler_xyz_torch(R):
    """[GPU版本] 旋转矩阵转欧拉角 (XYZ)"""
    sy = torch.clamp(R[..., 0, 2], -1.0, 1.0)
    beta = torch.asin(sy)
    alpha = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    gamma = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    rad2deg = 180.0 / 3.141592653589793
    return torch.stack([alpha, beta, gamma], dim=-1) * rad2deg

def pixloc_to_osg_torch(T_c2w):
    """
    [GPU版本] PixLoc Pose (ECEF) -> WGS84 & Euler
    T_c2w: Tensor (B, 4, 4)
    """
    if T_c2w.ndim == 2: T_c2w = T_c2w.unsqueeze(0)
    
    R_c2w = T_c2w[..., :3, :3]
    t_c2w = T_c2w[..., :3, 3]
    
    t_wgs84 = ECEF_to_WGS84_torch(t_c2w)
    lon, lat = t_wgs84[..., 0], t_wgs84[..., 1]
    
    R_enu2ecef = get_rotation_enu_in_ecef_torch(lon, lat)
    R_ecef2enu = R_enu2ecef.transpose(-1, -2)
    R_pose_in_enu = torch.matmul(R_ecef2enu, R_c2w)
    euler_in_enu = matrix_to_euler_xyz_torch(R_pose_in_enu)
    
    # 组装 kf_pose [x, y, z, pitch, roll, yaw]
    kf_pose = torch.cat([t_c2w, euler_in_enu], dim=-1)
    return t_wgs84, kf_pose

# ==========================================
# Part 2: 辅助四元数/矩阵运算 (Batch支持)
# ==========================================

def quat_to_rotmat(quat):
    """[w, x, y, z] -> 3x3 Matrix"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2 = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    row0 = torch.stack([1 - 2*(y2 + z2), 2*(xy - wz), 2*(xz + wy)], dim=-1)
    row1 = torch.stack([2*(xy + wz), 1 - 2*(x2 + z2), 2*(yz - wx)], dim=-1)
    row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(x2 + y2)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def pose_vec_to_matrix(pose_vec):
    """[qw, qx, qy, qz, tx, ty, tz] -> 4x4 Matrix"""
    quat = pose_vec[..., :4] 
    trans = pose_vec[..., 4:]
    
    R = quat_to_rotmat(quat)
    T = torch.eye(4, device=pose_vec.device).repeat(pose_vec.shape[:-1] + (1, 1))
    T[..., :3, :3] = R
    T[..., :3, 3] = trans
    return T

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x))"""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    [GPU版本] 旋转矩阵转四元数 [w, x, y, z]
    Args:
        matrix: (..., 3, 3)
    Returns:
        quaternions: (..., 4)
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

# ==========================================
# Part 3: 核心 GPU 采样与优化函数 (稳定版)
# ==========================================

def batch_sample_depth(pose_tensor_vec, dsm_tensor, dsm_meta):
    """GPU批量采样深度"""
    # 1. Pose -> Matrix -> WGS84
    T_c2w = pose_vec_to_matrix(pose_tensor_vec)
    t_wgs84, _ = pixloc_to_osg_torch(T_c2w) # (B, 3) [lon, lat, alt]
    
    lon = t_wgs84[..., 0]
    lat = t_wgs84[..., 1]
    
    # 2. WGS84 -> Pixel Coordinates (Normalized -1 to 1)
    min_x = dsm_meta['min_x']
    max_y = dsm_meta['max_y']
    res_x = dsm_meta['res_x']
    res_y = dsm_meta['res_y'] 
    h_map = dsm_meta['height']
    w_map = dsm_meta['width']
    
    u = (lon - min_x) / res_x
    v = (lat - max_y) / res_y
    
    norm_u = 2.0 * (u / (w_map - 1.0)) - 1.0
    norm_v = 2.0 * (v / (h_map - 1.0)) - 1.0
    
    # 3. Grid Sample
    grid = torch.stack([norm_u, norm_v], dim=-1).view(-1, 1, 1, 2)
    
    # 边界保护：出界填0
    depths = F.grid_sample(dsm_tensor.expand(grid.shape[0], -1, -1, -1), grid, align_corners=True, padding_mode='zeros')
    
    return depths.view(-1)

def residual_jacobian_batch_depth(pose_data_q, dsm_tensor, dsm_meta, target_depth):
    """全 GPU 计算 Jacobian 和 Hessian (带数值稳定保护)"""
    B, N, _ = pose_data_q.shape
    device = pose_data_q.device
    num_poses = B * N
    flat_poses = pose_data_q.view(num_poses, 7)
    
    eps_trans = 0.5
    eps_rot = np.deg2rad(0.1)
    
    # === 构建微扰 ===
    batch_poses = flat_poses.unsqueeze(1).repeat(1, 7, 1) 
    
    # 平移微扰
    batch_poses[:, 1, 4] += eps_trans
    batch_poses[:, 2, 5] += eps_trans
    batch_poses[:, 3, 6] += eps_trans
    
    # 旋转微扰 (四元数更新)
    half_theta = eps_rot / 2.0
    delta_w = np.cos(half_theta)
    delta_xyz = np.sin(half_theta)
    
    q_orig = batch_poses[:, 0, :4] 
    dq = torch.tensor([
        [delta_w, delta_xyz, 0.0, 0.0],
        [delta_w, 0.0, delta_xyz, 0.0],
        [delta_w, 0.0, 0.0, delta_xyz]
    ], device=device, dtype=q_orig.dtype)
    
    for i in range(3):
        idx = 4 + i
        dq_curr = dq[i].unsqueeze(0)
        w1, x1, y1, z1 = q_orig[:, 0], q_orig[:, 1], q_orig[:, 2], q_orig[:, 3]
        w2, x2, y2, z2 = dq_curr[:, 0], dq_curr[:, 1], dq_curr[:, 2], dq_curr[:, 3]
        
        batch_poses[:, idx, 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        batch_poses[:, idx, 1] = w1*x2 + w2*x1 + y1*z2 - z1*y2
        batch_poses[:, idx, 2] = w1*y2 + w2*y1 + z1*x2 - x1*z2
        batch_poses[:, idx, 3] = w1*z2 + w2*z1 + x1*y2 - y1*x2
        
    # === 批量采样 ===
    huge_batch_input = batch_poses.view(-1, 7)
    all_depths = batch_sample_depth(huge_batch_input, dsm_tensor, dsm_meta)
    all_depths = all_depths.view(B, N, 7)
    
    # === 计算梯度 ===
    d_center = all_depths[..., 0]
    d_perturb = all_depths[..., 1:]
    d_center_unsq = d_center.unsqueeze(-1)
    
    J = torch.zeros((B, N, 6), device=device, dtype=pose_data_q.dtype)
    J[..., :3] = (d_perturb[..., :3] - d_center_unsq) / eps_trans
    J[..., 3:] = (d_perturb[..., 3:] - d_center_unsq) / eps_rot
    
    # 🚑 核心修复 1: 梯度清洗与截断
    J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    J = torch.clamp(J, min=-1000.0, max=1000.0)
    
    # === 计算 H, g ===
    if not torch.is_tensor(target_depth):
        target_depth = torch.tensor(target_depth, device=device)
    if target_depth.ndim == 0:
        target_depth = target_depth.view(1, 1).expand(B, N)
    elif target_depth.ndim == 1:
        target_depth = target_depth.view(B, 1).expand(B, N)
        
    residual = d_center - target_depth
    
    # 过滤无效深度
    valid_mask = (d_center > 0.1).float()
    residual = residual * valid_mask
    J = J * valid_mask.unsqueeze(-1)
    
    J_vec = J.unsqueeze(-1)
    H_mat = torch.matmul(J_vec, J_vec.transpose(-1, -2))
    
    # 🚑 核心修复 2: Hessian 阻尼 (防止 Cholesky 崩溃)
    damping = torch.eye(6, device=device).view(1, 1, 6, 6).expand(B, N, -1, -1)
    H_mat = H_mat + (1e-3 * damping)
    
    H_mat = torch.nan_to_num(H_mat, nan=0.0)
    g_vec = J * residual.unsqueeze(-1)
    g_vec = torch.nan_to_num(g_vec, nan=0.0)
    
    return -g_vec, H_mat