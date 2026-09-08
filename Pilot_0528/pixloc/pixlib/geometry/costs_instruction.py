import torch
from typing import Optional, Tuple
from torch import Tensor
import copy
from . import Pose, Camera
from .optimization import J_normalization, skew_symmetric
from .interpolation import Interpolator
import torch.nn.functional as F
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
    #将 3D 点从一个坐标系（body）变换到另一个坐标系（view / 相机）
    #输入：
        # p3d：3D 点在 body（无人机机体或渲染参考）坐标系
        # body2view_pose_data：body → view 的位姿（旋转+平移）
    #输出：
        # 3D 点在 view/camera 坐标系下的坐标

    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)
@torch.jit.script
def project_p3d(camera_data, p3d):
    #相机内参投影 + 可见性判断                 可见性判断 → “告诉 LM 哪些点能在图像上看到”
    #LM → 只根据可见点计算 residual、更新位姿
    #输入：
    #camera_data：相机参数，tensor，通常包含 [image_width, image_height, fx, fy, cx, cy]
    # p3d：三维点坐标，shape [N, 3] 或 [B, N, 3]
    # 输出
    # p2d：投影到图像平面的二维像素坐标 [u, v]
    # valid：布尔掩码，表示该 3D 点是否在相机前方且在图像内
    #在相机前方：Z 坐标 > 0
    # 投影在图像范围内：像素坐标 [u, v] 满足
    # 0≤u≤width−1,0≤v≤height−1
    # 0≤u≤width−1,0≤v≤height−1


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
        
        # 插值的目的是从浮点坐标取值，确保投影后的每个 3D 点都能提取特征
        # 取棋盘上的 某个随机位置 [x=1.3, y=0.7] 的值，它不是整数格子，不在任何一个格子的中心，插值就像把四周的格子加权平均，得到这个浮点位置的“连续值”
        # 投影得到的 p2d 是浮点数，不是整数像素
        # LM 优化要求每个 3D 点对应一个特征
        # 直接取最近整数像素会产生误差（特别是亚像素级对齐）
        # 插值可以得到平滑、连续的特征值 → 提高优化精度

        # f_r：参考帧特征图 [C, H, W]
        # p2d_r：投影点 [u, v]（通常是浮点数）
        # 返回：
        # fp_r：插值后的特征 [N, C]
        # valid_r：插值是否有效（在图像内 + 可见）


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
        #核心作用是将原始的误差（通常是平方误差）转换成一个对异常值不敏感的损失值，并计算出对应的权重，供优化器（如 LM 算法）使用

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
        c_ref,c_query):
        # pose位姿； f特征图； cam相机参数； p3D三维点； c特征置信度(影响残差r的权重)
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
        p3d_r = transform_p3d(pose_data_r, p3D)     #把三维点从“body”坐标系变换到“参考帧相机”坐标系
        p2d_r, visible_r = project_p3d(cam_data_r, p3d_r)
        #相机内参投影 + 可见性判断             
        # 输出
        # p2d：投影到图像平面的二维像素坐标 [u, v]
        # valid：布尔掩码，表示该 3D 点是否在相机前方且在图像内
        p2d_r = p2d_r[0]
        visible_r = visible_r[0]
        fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)    #插值

        valid_ref_mask = (valid_r & visible_r).to(torch.float32)  # shape: [1, N]
        #valid_ref_mask 是一个 shape 为 [1, N] 的浮点掩码，只有参考帧中既可见又能插值的点为 1，其余为 0。
        mask = valid_ref_mask.transpose(1, 0)  # shape: [N, 1]，避免 unsqueeze
        #.transpose(1, 0)：将第 0 维和第 1 维交换，得到 [N, 1]

        fp_r = fp_r[0] * mask  # [N, C]
        p3D = p3D[0] * mask    # [N, 3]

        fp_r = fp_r.repeat(1, num_init_pose, 1, 1)             #num_init_pose：批量初始位姿的数量
        #fp_r.repeat(1, num_init_pose, 1, 1) 后 shape 变成 [1, num_init_pose, N, C]
        #“数据复制”，目的是为了 “维度对齐”
        #生成了 num_init_pose（比如 100 个）个，把“唯一的”参考点（Reference Feature Points）复制 100 份，让每一个猜测位姿都拥有一份属于自己的参考点数据，从而一次性算完所有误差。
        p3D = p3D.repeat(1, num_init_pose, 1, 1)               #只保留可见且有效的点，用多个初始位姿（num_init_pose）去优化同一组 3D 点

        # fp_r：参考帧的特征向量 [1, N, C]
        # p3D：三维点 [1, N, 3]
        # mask：可见性/有效性掩码 [N, 1]
        
        fp_r = torch.nn.functional.normalize(fp_r, dim=-1)
        

        # === Step 2: 查询帧中投影 ===
        p3d_q = transform_p3d(pose_data_q, p3D)
        p2d_q, visible_q = project_p3d(cam_data_q, p3d_q)

        p2d_q = p2d_q * mask.T.view(1, 1, -1, 1)  # [B, N, 2] * [1, N, 1]  # [B, N, 2]，广播避免 unsqueeze
        # 只保留参考帧可见的点对应的查询帧投影坐标，view作用变成 [1, 1, N, 1]
        #用掩码筛选查询帧的投影坐标，只保留参考帧有效点对应的查询帧投影，其他点全部置零。
        
        # === Step 3: 查询帧特征采样 ===
        fp_q, valid_q, J_f = self.interpolate_feature_map(f_q, p2d_q.reshape(1, -1, 2), return_gradients=True)
        #对查询帧的特征图 f_q，在所有投影点 p2d_q 处进行双线性插值采样
        #J_f：特征图在投影点处的梯度
        fp_q = fp_q.view(1, num_init_pose, -1, fp_q.shape[-1])
        #将插值采样得到的查询帧特征 fp_q，按照初始位姿数量 num_init_pose 和每个位姿下的点数 N，重新整理成 shape [1, num_init_pose, N, C]
        valid_q = valid_q.view(1, num_init_pose, -1)
        J_f = J_f.view(1, num_init_pose, -1, J_f.shape[-2], J_f.shape[-1])
        visible_q = visible_q.view(1, num_init_pose, -1)
        
        res = fp_q - fp_r

        # === Step 4: Jacobians ===
        R = pose_data_q[..., :9].view(1, -1, 3, 3)
        J_p3d_rot = R[:, :, None].matmul(-skew_symmetric(p3D))    #matmul：矩阵乘法
        J_p3d_tran = R[:, :, None].expand(-1, -1, J_p3d_rot.shape[2], -1, -1)
        J_p3d_pose = torch.cat((J_p3d_rot, J_p3d_tran), dim=-1)

        fx, fy = cam_data_q[:, :, 2], cam_data_q[:, :, 3]
        zero = torch.zeros_like(fx)
        f_diag_embed = torch.stack([fx, zero, zero, fy], dim=-1).reshape(1, -1, 1, 2, 2)      #扩展 fx, fy 到 [B, N, 1, 2, 2]

        J_p2d_p3d = f_diag_embed @ J_project(p3d_q)  # [B, N, 2, 3]
        J = J_f @ J_p2d_p3d @ J_p3d_pose             # [B, N, 2, 6]

        # === Step 5: loss 和加权 ===
        #通过 weights 变量，完美融合了传统几何约束（Outliers剔除）和深度学习特征（Uncertainty预测）
        #5-1 鲁棒核函数(Robust Loss)
        #如果某个点误差特别大，loss_fn1 会给它一个很小的 w_loss，让它不要影响优化结果。
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn1(cost)        

        #5-2 几何有效性(Geometry Validity)
        #如果点在这个位姿下根本看不见（在图像外或深度<0），直接把权重置为 0，剔除掉。
        valid_query = (valid_q & visible_q).float()
        weight_loss = w_loss * valid_query

        #5-3 特征置信度(Uncertainty)
        #利用 confidence。如果网络觉得这个区域纹理不清，w_unc 就会很小。
        ## 插值取出 Query 和 Reference 图上的置信度
        weight_q, _, _ = self.interpolate_feature_map(c_query, p2d_q.reshape(1, -1, 2))
        weight_q = weight_q.view(1, num_init_pose, -1, weight_q.shape[-1])

        weight_r, _, _ = self.interpolate_feature_map(c_ref, p2d_r)
        weight_r = weight_r *  mask.T.view(1, -1, 1)  # [N, C]
        weight_r = weight_r.repeat(1, num_init_pose, 1, 1)

        # 联合置信度：两张图都觉得这个点好，才是真的好
        w_unc = weight_q.squeeze(-1) * weight_r.squeeze(-1)  # [B, N]

        #总权重合成
        weights = weight_loss * w_unc

        # 1. J * res: 把残差投影到切空间
        grad = (J * res[..., :, :, None]).sum(dim=-2)
        #2. 乘以权重
        grad = weights[:, :, :, None] * grad
        # 3. 对所有点求和
        grad = grad.sum(-2)

        # 1. 计算 J.T @ J
        # permute把维度调换，准备矩阵乘法
        Hess = J.permute(0, 1, 2, 4, 3) @ J
        # 2. 乘以权重
        Hess = weights[:, :, :, None, None] * Hess
        # 3. 对所有点求和
        Hess = Hess.sum(-3)
        
        return -grad, Hess, w_loss, valid_query, p2d_q, cost   # torch.Size([64, 6]) , torch.Size([64, 6, 6]), torch.Size([64, 447]), torch.Size([64, 447])
    
