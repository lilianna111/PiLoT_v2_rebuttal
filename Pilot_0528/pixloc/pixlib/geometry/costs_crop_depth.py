import torch
from typing import Optional, Tuple
from torch import Tensor
import copy
from . import Pose, Camera
from .optimization import J_normalization, skew_symmetric
from .interpolation import Interpolator
import torch.nn.functional as F
import sys
sys.path.append('/home/amax/Documents/code/PiLoT/PiLoT')
from pixloc.pixlib.geometry.ray_casting import TargetLocation
import pyproj
import numpy as np
import time
from scipy.spatial.transform import Rotation as SciRotation
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
    def residual_jacobian_batch_quat(
        self, pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D: Tensor,
        c_ref,c_query,p2d_r,visible_r, gt_depth=None, dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None):
        # inputs = {
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
        
        # ========== 新增：深度损失部分 ==========
        # 对查询帧 pose_data_q 计算深度损失，并融合到 grad 和 Hess 中
        if gt_depth is not None and dsm_path is not None and render_T_ecef is not None:
            depth_weight = 0.06  # 深度损失权重（提高权重，增强深度约束）
            depth_truncate = 5.0  # 深度鲁棒损失的截断值（单位：米），降低以更快截断大误差
            
            # 【性能优化参数】
            top_k_poses = 100  # 只对重投影误差最小的K个位姿计算深度
            depth_num_sample = 3000  # 深度预测的采样点数（从4000降低）
            compute_gradient = True  # 是否计算梯度（如果False，只对最优位姿计算深度用于监督）
            max_depth_error = 1.0 # 最大允许的深度误差（米），超过此值的位姿将被跳过或降权
            
            # 确保 gt_depth 是标量
            if isinstance(gt_depth, (np.ndarray, torch.Tensor)):
                gt_depth = float(gt_depth.item() if hasattr(gt_depth, 'item') else gt_depth)
            else:
                gt_depth = float(gt_depth)
            
            # 初始化 TargetLocation
            if not hasattr(self, 'target_locator'):
                self.target_locator = TargetLocation({"ray_casting": {}}, use_dsm=False)
            if not hasattr(self, '_ecef_to_wgs84'):
                self._ecef_to_wgs84 = pyproj.Transformer.from_crs(
                    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326", always_xy=True
                )
            transformer_ecef_to_wgs84 = self._ecef_to_wgs84

            # 预计算与位姿无关的量：ECEF -> WGS84、ENU旋转
            t_c2w_ecef = render_T_ecef.to(dtype=torch.float64) if isinstance(render_T_ecef, torch.Tensor) else torch.from_numpy(render_T_ecef).to(device=pose_data_q.device, dtype=torch.float64)
            t_ecef = t_c2w_ecef.detach().cpu().numpy()
            lon, lat, alt = transformer_ecef_to_wgs84.transform(t_ecef[0], t_ecef[1], t_ecef[2], radians=False)
            lat_rad, lon_rad = np.radians(lat), np.radians(lon)
            up = np.array([np.cos(lon_rad) * np.cos(lat_rad), np.sin(lon_rad) * np.cos(lat_rad), np.sin(lat_rad)])
            east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
            north = np.cross(up, east)
            rot_enu_in_ecef = np.column_stack([east, north, up])
            
            # 深度损失的梯度和Hessian
            grad_depth = torch.zeros_like(grad)  # [1, num_init_pose, 6]
            Hess_depth = torch.zeros_like(Hess)  # [1, num_init_pose, 6, 6]
            
            # 数值微分参数
            eps = 1e-4  # 微小扰动
            
            # 【Top-K筛选】根据重投影误差选择最优的K个候选位姿
            # cost的形状: [1, num_init_pose, num_points]
            cost_per_pose = cost.mean(dim=-1)[0]  # [num_init_pose]，每个位姿的平均误差
            
            # 如果候选位姿数量少于top_k，就全部计算
            actual_k = min(top_k_poses, num_init_pose)
            top_k_indices = torch.topk(cost_per_pose, actual_k, largest=False).indices  # 误差最小的K个
            
            print(f"[深度损失] 从 {num_init_pose} 个候选位姿中选择误差最小的 {actual_k} 个进行深度计算")
            print(f"[深度损失] Top-{actual_k} 误差范围: {cost_per_pose[top_k_indices].min().item():.4f} ~ {cost_per_pose[top_k_indices].max().item():.4f}")
            
            # 【批量处理优化】一次性收集所有候选位姿，然后批量预测深度
            t_batch_start = time.time()
            
            # Step 1: 收集所有需要预测的pose（只收集Top-K个）
            all_poses_to_query = []  # 存储所有pose: actual_k * (1 or 4) 
            pose_index_map = {}  # 映射：在all_poses_to_query中的索引 -> 原始位姿索引i
            pose_euler_list = []  # 存储每个base pose的欧拉角，用于后续计算
            
            for idx, i in enumerate(top_k_indices.tolist()):
                try:
                    # 提取当前位姿
                    R_w2c_q = pose_data_q[0, i, :9].view(3, 3).to(dtype=torch.float64)
                    
                    # 转 c2w 并还原 preprocess
                    R_c2w_prep = R_w2c_q.transpose(-1, -2)
                    R_c2w = R_c2w_prep.clone()
                    R_c2w[:, 1] *= -1
                    R_c2w[:, 2] *= -1
                    
                    # 转ECEF -> WGS84 -> 欧拉角（使用预计算的 ENU 旋转基）
                    R_c2w_np = R_c2w.detach().cpu().numpy()
                    R_pose_in_enu = rot_enu_in_ecef.T @ R_c2w_np
                    euler_angles = SciRotation.from_matrix(R_pose_in_enu).as_euler('xyz', degrees=True)
                    pitch, roll, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
                    
                    # 构造批量姿态：基准 + (可选)扰动
                    base_pose = [lon, lat, alt, roll, pitch, yaw]
                    base_idx_in_list = len(all_poses_to_query)
                    all_poses_to_query.append(base_pose)
                    pose_euler_list.append([pitch, roll, yaw])
                    pose_index_map[idx] = (i, base_idx_in_list)  # (原始索引, 在all_poses中的起始索引)
                    
                    # 【可选】添加扰动位姿用于计算梯度
                    if compute_gradient:
                        # 1. 旋转扰动 (Pitch, Roll, Yaw)
                        for j in range(3):
                            euler_perturbed = [pitch, roll, yaw]
                            euler_perturbed[j] += eps
                            perturbed_pose = [lon, lat, alt, euler_perturbed[0], euler_perturbed[1], euler_perturbed[2]]
                            all_poses_to_query.append(perturbed_pose)
                        
                        # 2. 高度扰动 [新增] -> 对应 depths[4]
                        eps_alt = 0.5 # 0.5米
                        all_poses_to_query.append([lon, lat, alt + eps_alt, pitch, roll, yaw])
                        # all_poses_to_query.append(perturbed_pose)
                        
                except Exception as e:
                    # 如果某个位姿处理失败，填充占位符（后续会被跳过）
                    print(f"[深度损失-Pose{i}] 位姿转换失败: {str(e)}")
                    num_poses_per_candidate = 5 if compute_gradient else 1
                    all_poses_to_query.extend([[None]*6] * num_poses_per_candidate)
                    pose_euler_list.append(None)
                    pose_index_map[idx] = (i, None)
            
            # Step 2: 批量预测所有深度
            try:
                valid_poses = [p for p in all_poses_to_query if p[0] is not None]
                if len(valid_poses) > 0:
                    num_poses_per_candidate = 5 if compute_gradient else 1
                    print(f"[深度损失] 批量预测 {len(valid_poses)} 个位姿（Top-{actual_k} × {num_poses_per_candidate}，采样点数={depth_num_sample}）...")
                    all_depths = self.target_locator.predict_center_depth_batch(
                        DSM_path=dsm_path,
                        poses=valid_poses,
                        num_sample=depth_num_sample,  # 使用降低后的采样点数
                        object_pixel_coords=[1915.7, 1075.1]
                    )
                    
                    # 将depths映射回原始索引（考虑跳过的invalid poses）
                    depths_full = []
                    valid_idx = 0
                    for p in all_poses_to_query:
                        if p[0] is not None:
                            depths_full.append(all_depths[valid_idx])
                            valid_idx += 1
                        else:
                            depths_full.append(None)
                    
                    t_batch_end = time.time()
                    print(f"[深度损失] 批量预测完成，耗时: {t_batch_end - t_batch_start:.3f}s")
                    
                    # Step 3: 处理每个候选位姿的深度结果，计算梯度和Hessian
                    for idx in range(actual_k):
                        if idx not in pose_index_map:
                            continue
                        
                        original_i, base_idx_in_list = pose_index_map[idx]
                        if base_idx_in_list is None or pose_euler_list[idx] is None:
                            continue
                        
                        # 提取当前位姿的深度结果（base + 可选的3个扰动）
                    # 提取当前位姿的深度结果（base + 3个旋转 + 1个高度 = 5）
                        num_poses_for_this = 5 if compute_gradient else 1 # <--- 改为5
                        depths = depths_full[base_idx_in_list:base_idx_in_list + num_poses_for_this]
                        
                        if depths[0] is None:
                            continue
                        
                        pred_depth = depths[0]
                        residual = pred_depth - gt_depth
                        
                        # 【过滤大误差】跳过深度误差过大的位姿
                        if abs(residual) > max_depth_error:
                            # print(f"[深度损失-Pose{original_i}] 跳过：误差 {residual:.2f}m 超过阈值 {max_depth_error}m")
                            continue
                        
                        # 应用鲁棒损失函数
                        depth_cost_tensor = torch.tensor([residual ** 2], device=grad.device, dtype=grad.dtype)
                        depth_loss, depth_w_loss, _ = self.loss_fn1(
                            depth_cost_tensor,
                            alpha=0.0,
                            truncate=depth_truncate
                        )
                        
                        # 使用数值微分计算Jacobian（如果compute_gradient=True）
                        J_depth = np.zeros(6)
                        if compute_gradient:
                            # 1. 旋转梯度 (Index 0,1,2)
                            for j in range(3): 
                                if len(depths) > j + 1 and depths[j + 1] is not None:
                                    J_depth[j] = (depths[j + 1] - pred_depth) / eps
                            
                            # 2. 高度梯度 (Index 5) [新增]
                            # depths[4] 对应刚才加的高度扰动位姿
                            eps_alt = 0.5
                            if len(depths) > 4 and depths[4] is not None:
                                J_depth[5] = (depths[4] - pred_depth) / eps_alt
                        
                        # 转为torch tensor
                        J_depth_torch = torch.from_numpy(J_depth).to(device=grad.device, dtype=grad.dtype)
                        
                        # 计算梯度和Hessian（使用原始索引original_i）
                        grad_depth[0, original_i] = J_depth_torch * residual * depth_w_loss.item() * depth_weight
                        Hess_depth[0, original_i] = torch.outer(J_depth_torch, J_depth_torch) * depth_w_loss.item() * depth_weight
                        
                else:
                    print(f"[深度损失] 警告：没有有效的位姿可供预测")
                    
            except Exception as e:
                print(f"[深度损失] 批量预测失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 【添加调试信息】观察两部分损失的数值范围
            grad_reproj_norm = torch.norm(grad).item()
            grad_depth_norm = torch.norm(grad_depth).item()
            Hess_reproj_norm = torch.norm(Hess).item()
            Hess_depth_norm = torch.norm(Hess_depth).item()
            
            print(f"[损失融合] 重投影grad范数: {grad_reproj_norm:.6f}, 深度grad范数: {grad_depth_norm:.6f}, 比例: {grad_depth_norm/(grad_reproj_norm+1e-8):.3f}")
            print(f"[损失融合] 重投影Hess范数: {Hess_reproj_norm:.6f}, 深度Hess范数: {Hess_depth_norm:.6f}, 比例: {Hess_depth_norm/(Hess_reproj_norm+1e-8):.3f}")
            
            # 融合深度损失到总的梯度和Hessian
            grad = grad + grad_depth
            Hess = Hess + Hess_depth
        
        return -grad, Hess, w_loss, valid_query, p2d_q, cost
    
