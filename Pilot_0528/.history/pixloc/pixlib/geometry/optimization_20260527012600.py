from packaging import version
import torch
import logging
logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky

def optimizer_step(B: torch.Tensor,
                   A: torch.Tensor,
                   lambda_=0,
                   mask=None,
                   eps: float = 1e-6,
                   max_tries: int = 8) -> torch.Tensor:
    """Solve A * delta = B with a robust Cholesky-based solver.

    [原理/背景：GN/LM 为什么会出现这个线性系统？]
    非线性最小二乘里，我们在当前位姿/参数处做一次线性化：
      minimize ||r(x)||^2
    得到 Gauss-Newton 近似的正规方程（或 LM 形式）：
      (H + λD) Δ = -g
    其中 H≈JᵀWJ（理论上半正定/对称），g≈JᵀWr。
    在本代码里，A 就对应 (H + λD)（或其某种近似），B 对应右端项。

    [原理：为什么会报 Cholesky not positive-definite？]
    Cholesky 分解只适用于对称正定（SPD）矩阵。
    但在实际中，A 可能因为以下原因变得“非正定/病态”：
    - 有效点过少、几何退化（Hessian 退化/接近奇异）
    - 权重极端或出现 NaN/Inf（数值污染）
    - 浮点误差导致轻微非对称，从而触发分解失败

    [工程策略（本函数做了什么）]
    - 对称化：把 A 拉回对称矩阵（减小数值误差影响）
    - LM 对角阻尼：在对角线上加正量，让 A 更接近 SPD（步长更保守、更稳定）
    - mask 保护：对无效 batch 元素直接用单位阵 I（保证可解），并把 B 置 0（不更新）
    - cholesky_ex + jitter 重试：失败就逐步加大对角 jitter，避免偶发非正定导致进程崩溃
    - 最后兜底：退回通用 solve；再不行就返回 0 step（宁可不更新，也不崩）
    """
    n = A.shape[-1]

    # [改动点] 兼容 B 的两种形状：
    # - (..., n)   : 常见的梯度向量
    # - (..., n, 1): 线性方程右端列向量
    # 统一成 (..., n, 1)，便于后续 solve / cholesky_solve。
    if B.ndim == A.ndim - 1:
        B = B[..., None]

    # [改动点/原理] 数值稳定性：对称化 Hessian/近似 Hessian。
    # 理论上 GN/LM 的 H 应该是对称的，但浮点误差/实现细节可能导致轻微非对称，
    # 非对称会破坏“可 Cholesky 分解”的前提，从而更容易失败。
    A = 0.5 * (A + A.transpose(-1, -2))

    # [改动点/原理] Levenberg(-Marquardt) 阻尼：在对角线上加正值，使矩阵更接近正定。
    # 你遇到的报错 "not positive-definite" 就是因为 A 不是正定矩阵，Cholesky 无法分解；
    # 直觉上：λ 越大，系统越“像梯度下降”，步长更保守，因此更稳定（但收敛可能更慢）。
    if torch.is_tensor(lambda_):
        lam = lambda_.to(device=A.device, dtype=A.dtype)
    else:
        lam = torch.tensor(lambda_, device=A.device, dtype=A.dtype)

    if lam.ndim == 0:
        # 标量阻尼：diag(A) * lambda
        diag_damp = A.diagonal(dim1=-2, dim2=-1) * lam
    else:
        # Allow per-parameter damping (e.g. shape (n,) or (..., n)).
        if lam.shape[-1] != n:
            lam = lam[..., None].expand(A.shape[:-2] + (n,))
        diag_damp = A.diagonal(dim1=-2, dim2=-1) * lam

    # [原理] clamp(min=eps)：避免某些维度 diag 太小/为 0 时阻尼不起作用（退化/低纹理时更明显）
    A = A + diag_damp.clamp(min=eps).diag_embed()

    if mask is not None:
        mask = mask.to(device=A.device)
        # [改动点/原理] mask 保护：对无效 batch 元素用单位阵 I 替换，确保系统可解且不会引入奇异矩阵。
        # 旧写法如果用 ones_like(A)（全 1 矩阵）替代，会是奇异矩阵，反而更不稳定。
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
        eye = eye.reshape((1,) * (A.ndim - 2) + (n, n)).expand_as(A)
        A = torch.where(mask[..., None, None], A, eye)
        # [改动点/原理] 同时把无效元素的 B 置 0 -> 解出来 delta = 0，表示“不更新”（避免无效数据影响优化）。
        B = B.masked_fill(~mask[..., None, None], 0.)

    # [保持原策略] 在 CPU 上求解（沿用原实现的 CPU offload）。
    # 如果你希望全程 GPU 求解，可以再讨论，但要注意数值稳定性/性能权衡。
    A_ = A.cpu()
    B_ = B.cpu()

    # [改动点/原理] Cholesky 重试机制（核心修复）：
    # - 直接 torch.linalg.cholesky(A_) 一旦遇到非正定就会抛异常并导致进程挂掉；
    # - 这里用 cholesky_ex 先拿到 info（是否成功），失败就给对角线加更大的 jitter 再试。
    # 原理直觉：对角线加 δI 会把所有特征值整体上移 δ，从而把“略微为负/接近 0”的特征值推回正区间，
    # 使矩阵更接近 SPD。代价是步长更小（更保守）。
    jitter = eps
    U = None
    for _ in range(max_tries):
        try:
            if hasattr(torch.linalg, 'cholesky_ex'):
                U, info = torch.linalg.cholesky_ex(A_)
                if torch.all(info == 0):
                    break
                U = None
            else:
                U = cholesky(A_)
                break
        except Exception:
            U = None

        # 失败后：对角线上加 jitter*I 并放大 jitter，直到成功或达到最大尝试次数。
        A_ = A_ + (jitter * torch.eye(n, dtype=A_.dtype, device=A_.device)).reshape(
            (1,) * (A_.ndim - 2) + (n, n)
        )
        jitter *= 10.0

    if U is not None:
        # Cholesky 成功：用 cholesky_solve 解线性方程（更快、更稳定）
        delta = torch.cholesky_solve(B_, U)[..., 0]
        return delta.to(A.device)

    # [改动点/原理] 最后兜底：如果始终无法把矩阵修到可 Cholesky 分解，
    # 就退回到通用线性求解（indefinite 但可逆时也能解），避免进程崩溃。
    try:
        delta = torch.linalg.solve(A_, B_)[..., 0]
    except Exception:
        # 极端情况下（仍不可解/数值异常）返回 0 step：等价于“本次迭代不更新”。
        logger.debug('Linear solve failed; returning zero step.')
        delta = torch.zeros_like(B_)[..., 0]
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
