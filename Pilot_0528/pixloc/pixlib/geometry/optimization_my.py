from ast import Lambda
from packaging import version
import torch
import logging
import time
logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky

def optimizer_step(B: torch.Tensor, A: torch.Tensor, lambda_=0, mask = None):

        tikhonov_matrix = torch.diag(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)).cuda()
        #第一步：吉洪诺夫正则化 (Tikhonov Regularization) / 阻尼
        #制让矩阵变成 正定矩阵 (Positive Definite)，保证它是可逆的
        tikhonov_matrix = tikhonov_matrix.to(A) #!?self.conf.trainable=False
        A = A + tikhonov_matrix
        #第二步：掩码保护 (Masking Protection)
        if mask is not None:
            # make sure that masked elements are not singular
            A = torch.where(mask[:, None, None], A, torch.ones_like(A).to(A))
            # set g to 0 to delta is 0 for masked elements
            B = B.masked_fill(~mask[:,None, None], 0.)
            
        #第三步：CPU 求解 (CPU Offloading)
        A_ = A.cpu()
        B_ = B.cpu()
        # start_time = time.time()
        
        #第四步：多重保险求解策略 (The Solver Hierarchy)
        try:
            ##方案 A：Cholesky 分解
            U = torch.linalg.cholesky(A_) # 或者 cholesky(A_)
        except RuntimeError as e:
            # === 修改这里：同时捕获两种错误信息 ===
            error_msg = str(e)
            if 'singular U' in error_msg or 'positive-definite' in error_msg:
                # logger.debug('Cholesky decomposition failed, fallback to LU.')
                try:
                    # start_time = time.time()
                    delta = torch.solve(A_, B_)[..., 0]
                    # print(f"LU solve: {time.time() - start_time:.6f}s")
                except RuntimeError:
                    delta = torch.zeros_like(B_)[..., 0]
                    # logger.debug('A is not invertible')
            else:
                raise # 如果是其他奇怪的错误，再报错
        else:
            delta = torch.cholesky_solve(B_, U)[..., 0]

        return delta.to(A.device)

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


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def J_normalization(x):
    """Jacobian of the L2 normalization, assuming that we normalize
       along the last dimension.
    """
    x_normed = torch.nn.functional.normalize(x, dim=-1)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    Id = torch.diag_embed(torch.ones_like(x_normed))
    J = (Id - x_normed.unsqueeze(-1) @ x_normed.unsqueeze(-2))
    J = J / norm.unsqueeze(-1)
    return J
