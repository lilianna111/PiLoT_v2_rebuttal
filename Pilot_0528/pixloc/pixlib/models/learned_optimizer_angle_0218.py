import logging
from typing import Tuple, Optional
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Agg')
from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa
import numpy as np
logger = logging.getLogger(__name__)

def transform_p3d(body2view_pose_data, p3d):
    R = body2view_pose_data[..., :9].view(-1, 3, 3)
    t = body2view_pose_data[..., 9:]
    return p3d @ R.transpose(-1, -2) + t.unsqueeze(-2)
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
class DampingNet(nn.Module):
    def __init__(self, conf, num_params=6):
        super().__init__()
        self.conf = conf
        if conf.type == 'constant':
            const = torch.zeros(num_params)
            self.register_parameter('const', torch.nn.Parameter(const))
        else:
            raise ValueError(f'Unsupported type of damping: {conf.type}.')

    def forward(self):
        min_, max_ = self.conf.log_range
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
        return lambda_


class LearnedOptimizer(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        feature_dim=None,

        # deprecated entries
        lambda_=0.,
        learned_damping=True,
    )

    def _init(self, conf):
        self.dampingnet = DampingNet(conf.damping)
        assert conf.learned_damping
        super()._init(conf)
    def compute_overall_loss(self, res, valid, weights, loss_fn=None):
        """
        Compute the overall loss after each iteration of pose optimization.

        Args:
            res (Tensor): Residuals tensor of shape (..., N), where N is the number of residuals.
            valid (Tensor): Boolean mask indicating valid residuals of shape (..., N).
            weights (Tensor): Weights tensor of shape (..., N).
            loss_fn (callable, optional): Robust loss function applied to squared residuals.

        Returns:
            Tensor: Overall weighted loss for the current iteration.
        """
        # Compute squared residuals
        squared_residuals = (res**2).sum(-1)
        # Apply robust loss function if provided
        if loss_fn is not None:
            loss = loss_fn(squared_residuals)
        else:
            loss = squared_residuals

        # Apply valid mask and weights
        loss = loss * valid.float()
        loss = loss * weights

        # Sum the loss over all residuals
        overall_loss = loss.sum(dim=-1)  # Sum over residual dimension

        return overall_loss
    def plot_loss_curve(self, overall_losses, save_path=None):
        """
        Plot the loss curve for the optimization process.

        Args:
            overall_losses (list or array): List of overall loss values for each iteration.
            save_path (str, optional): Path to save the plot image. If None, the plot will not be saved.
        """
        # Generate the iterations corresponding to the losses
        iterations = list(range(len(overall_losses)))

        # # Plot the curve
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, overall_losses, marker='o', linestyle='-', label='Overall Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Overall Loss vs Iteration')
        plt.grid(True)
        plt.legend()

        
        # # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Show the plot
        # plt.show()
    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init, camera: Camera, T_render, ref_camera: Camera,mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None, p2d_r=None,visible_r=None, T_gt = None, batch = True,
             gt_depth=None, gt_roll=None, gt_pitch=None, dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None):

        # ==================== 核心修复区域 START ====================
        # 无论传进来是 (N,2) 还是 (1,N,2) 还是 (1,1,N,2)
        # 统统强制 reshape 成 (1, 总点数, 2)
        # 这能彻底防止维度丢失或错误
        if p2d_r is not None:
            p2d_r = p2d_r.reshape(1, -1, 2)
        
        if visible_r is not None:
            visible_r = visible_r.reshape(1, -1)
        # ==================== 核心修复区域 END ====================
        T = T_init
        T_flat = T_init.to_flat()
        #将 4×4 的变换矩阵，压缩成一个紧凑的向量
        #矩阵不适合直接优化，to_flat() 通常会将位姿转换为 7维向量（四元数+平移，Quaternion + Translation）或者 6维向量（李代数 se(3)）。

        num_init_pose = T_init.shape[0]
        T_render = T_render.to_flat().expand(1, -1)
        qcamera = camera.unsqueeze(0).expand(num_init_pose, -1)
        #把“1 个相机的内参”，复制成“N 个一模一样的相机内参”，以便和那 N 个初始位姿（Initial Poses）一一对应
        #unsqueeze 是 PyTorch 张量的一个方法，用于在指定位置插入一个新的维度（即扩展维度）
        p3D = p3D.unsqueeze(0).expand(1, -1, -1)
        ref_camera = ref_camera.unsqueeze(0).expand(1, -1)
        c_ref, c_query = W_ref_query
        #把这次匹配结果的置信度拆分开，分别拿到‘参考点’的可靠程度 (c_ref) 和‘查询点’的可靠程度 (c_query)
        c_ref = c_ref.unsqueeze(0)
        c_query = c_query.unsqueeze(0)
        F_ref = F_ref.unsqueeze(0)
        F_query = F_query.unsqueeze(0)
        print("p2d_r: ", p2d_r.shape)
        args = (F_ref, 
                T_render.unsqueeze(0),
                ref_camera.unsqueeze(0),
                F_query, 
                qcamera.unsqueeze(0), 
                p3D.unsqueeze(0), 
                c_ref,
                c_query,
                p2d_r,
                visible_r,
                gt_depth,
                gt_roll,
                gt_pitch,
                dd,
                mul,
                origin,
                dsm_path,
                render_T_ecef
                )
        # print("F_ref: ", F_ref.shape)
        # args = tuple(x.double() if torch.is_tensor(x) else x for x in args)
        # T_flat = T_flat.double()
        # T = T.double()
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)
        #创建一个和 T 形状一样的布尔型张量 failed，并全部初始化为 False，放在 T 所在的设备上（比如 GPU）

        lambda_ = self.dampingnet()
        # lambda_ = 0
        overall_losses = []
        ratio = 3.75
        if ref_camera[0][0] == 480/ratio:
            num_iters = 2
        elif ref_camera[0][0] == 960/ratio:
            num_iters = 3
        elif ref_camera[0][0] == 1920/ratio:
            num_iters = 4
        # if T_gt is not None:
        #     p3d_q_gt = transform_p3d(T_gt, p3D)
        #     p2d_q_gt, _ = project_p3d(qcamera, p3d_q_gt)
        #     p2d_q_gt = p2d_q_gt[0]
        # print('优化前：',T_flat[0])
        err_list= []
        start_time = time.time()
        for i in range(num_iters): #self.conf.num_iters
            # 测试时假设训练好的 w_depth、w_angle 均为 1
            g, H, w_loss, valid, p2d_q, cost = self.cost_fn.residual_jacobian_batch_quat(T_flat.unsqueeze(0), *args)
            # g, H, w_loss, valid, p2d_q, cost = self.cost_fn.residual_jacobian_batch_quat(
            #     T_flat.unsqueeze(0), *args, w_depth=0.0, w_angle=1.0
            # )
            g, H, w_loss, valid, p2d_q, cost = [
                x.squeeze(0) if x.shape[0] == 1 else x for x in [g, H, w_loss, valid, p2d_q, cost]   #如果张量第0维是1，就去掉这一维
            ]
            g = g.unsqueeze(-1)
            #在张量 g 的最后一个维度插入一个新的维度
            failed = failed | (valid.long().sum(-1) < 10)  # too few points
            #对于每个样本，如果有效点（valid 为 True 的点）数量少于 10，就把对应的 failed 标记为 True。
            
            # compute the cost and aggregate the weights
            delta = optimizer_step(g, H, lambda_, mask=~failed)
            # print(f"Step 3 - Optimizer step: {time.time() - st*art_time:.6f}s")
            # compute the pose update
            dw, dt = delta.split([3, 3], dim=-1)
            #把张量 delta 按最后一个维度（dim=-1）分成两个部分，每部分长度为3，分别赋值给 dw 和 dt
            T_delta = Pose.from_aa(dw, dt)  
            #from_aa(cls, aa: torch.Tensor, t: torch.Tensor)根据轴角旋转向量（axis-angle，aa）和位移向量（t），构造一个三维空间中的位姿（Pose）
            T = T @ T_delta
            T_flat = T.to_flat()
            #把位姿对象 T（通常是 4x4 变换矩阵或 Pose 类）转换成一个“扁平化”的向量（比如 12维或6维）
            # p2d_q_gt1 = p2d_q_gt.repeat(T_flat.shape[0], 1, 1)
            # error = p2d_q_gt1 - p2d_q  

            # # 计算每个点的欧氏误差，再对有效点求平均
            # error_norm = torch.norm(error, dim=-1)  # [B, N]
            # error_mean = (error_norm * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)  # [B]
            # err_list.append(error_mean)
            # 关键：用于记录/选择的 cost 必须基于“更新后（含后处理）”的姿态。
            # 这里重新前向一次，拿到 post-update 的 cost，避免使用 update 前的旧 cost。
            with torch.no_grad():
                _, _, _, _, _, cost_post = self.cost_fn.residual_jacobian_batch_quat(
                    T_flat.unsqueeze(0), *args
                )
                cost_post = cost_post.squeeze(0) if cost_post.shape[0] == 1 else cost_post
            overall_losses.append(cost_post)
            # if self.early_stop(i=i, T_delta=T_delta, grad=g.squeeze(-1), cost=w_loss):
            #     break
        
        # errors_all = torch.stack(err_list, dim=0).T  # [b, 10]
        # 展示每个样本的误差变化 
        # for i in range(len(errors_all)):
        #     print(f"样本 {i} 的误差变化: {errors_all[i]}")

        # print('优化后：',T_flat[0])
        # 最终用于选 pose 的总损失：使用最后一次“更新后”的 cost。
        final_cost = overall_losses[-1] if len(overall_losses) > 0 else cost
        total_cost_loss = final_cost.sum(dim=1)
        # 低有效点候选在上游已被标记 failed，这里直接加大惩罚，避免被误选
        if failed.dtype != torch.bool:
            failed = failed.bool()
        total_cost_loss = torch.where(
            failed,
            torch.full_like(total_cost_loss, 1e12),
            total_cost_loss,
        )
        # best_idx = torch.argmin(total_loss)  # 最佳收敛位姿索引
        # print('argmax weight loss: ', torch.argmax(total_loss))
        # print('argmin cost: ', torch.argmin(total_cost_loss))
        
        end_time = time.time()
        # print('LM time: ', end_time - start_time)
        return T, failed, total_cost_loss # T[initial pose]

