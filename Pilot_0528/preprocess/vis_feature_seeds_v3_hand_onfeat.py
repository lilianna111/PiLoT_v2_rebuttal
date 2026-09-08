import os, cv2, torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
from typing import Optional
def _rot_tri(center, angle_rad, size_px=8.0, aspect=0.9):
    """
    在起点画一个朝向 angle_rad 的小三角
    - center: (x,y)
    - size_px: 控制大小
    - aspect: 控制胖瘦
    """
    s = float(size_px)
    # 定义局部坐标的三角形（质心在原点，tip朝x正方向）
    tip  = np.array([+0.6*s, 0.0])
    base = 0.6*s
    hw   = aspect * 0.55*s
    rear_top    = np.array([-base, +hw])
    rear_bottom = np.array([-base, -hw])
    tri = np.stack([tip, rear_top, rear_bottom], axis=0)

    # 旋转
    c, s_ = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c,-s_],[s_,c]])
    tri = tri @ R.T

    # 平移
    tri[:,0] += center[0]
    tri[:,1] += center[1]
    return tri
def pick_indices_from_qr_last(query_img_path, target_size, p2d_qr_last, k=5, tol_px=12, dedup=True):
    """
    仅允许从 p2d_qr_last 这组点中选 k 个点。
    - 点击时会“吸附”到最近的候选点；若最近距离 > tol_px 则忽略该次点击。
    - 返回：所选点在 p2d_qr_last 中的索引（可直接用于 p2d_r/p2d_q_init/p2d_q_refined 的同索引子集化）
    """
    Wt, Ht = target_size
    # 读图并缩放到与可视化一致的尺寸
    qry_bgr = cv2.imread(query_img_path); assert qry_bgr is not None, "读取 query 失败"
    qry = cv2.cvtColor(qry_bgr, cv2.COLOR_BGR2RGB)
    qry = cv2.resize(qry, (Wt, Ht), interpolation=cv2.INTER_AREA)

    P = np.asarray(p2d_qr_last, dtype=np.float32)
    if P.ndim == 3:
        P = P[0]  # (K,2)

    picked = []
    used = set()

    fig, ax = plt.subplots()
    ax.imshow(qry, extent=(0, Wt, Ht, 0))
    ax.set_xlim(0, Wt); ax.set_ylim(Ht, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"仅从紫色候选点中点击选择 {k} 个（左键）。ESC/关闭窗口结束。")

    # 画候选集（只能选这些）
    cand = ax.scatter(P[:,0], P[:,1], s=18, c='#9b5de5', edgecolors='white', linewidths=0.6, alpha=0.95, label='candidates')
    sel  = ax.scatter([], [], s=44, c='yellow', edgecolors='black', linewidths=1.2, label='selected')
    ax.legend(loc='upper right', fontsize=8)

    def on_click(ev):
        nonlocal picked
        if ev.inaxes != ax or ev.button != 1:
            return
        if len(picked) >= k:
            return
        if ev.xdata is None or ev.ydata is None:
            return

        # 最近邻到候选集
        c = np.array([ev.xdata, ev.ydata], dtype=np.float32)
        d2 = np.sum((P - c[None, :])**2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] > (tol_px * tol_px):
            # 点击离任何候选点都太远，忽略
            return
        if dedup and j in used:
            # 已被选过，忽略
            return

        picked.append(j); used.add(j)
        sel.set_offsets(P[np.array(picked, dtype=int)])
        fig.canvas.draw_idle()

        if len(picked) == k:
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if len(picked) == 0:
        print("⚠️ 未选中任何点（或全在阈值外），返回空索引。")
        return np.array([], dtype=np.int64)

    return np.array(picked, dtype=np.int64)
def crop_16_9(img):
    H, W = img.shape[:2]
    th = int(round(W*9/16))
    if th <= H:
        return img[:th, :]
    tw = int(round(H*16/9))
    x0 = max(0, (W - tw)//2)
    return img[:, x0:x0+tw]
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def _rot_tri(center, angle_rad, size_px=12.0, aspect=0.9):
    s = float(size_px)
    tip  = np.array([+0.60*s, 0.0])
    base = 0.60*s
    hw   = aspect * 0.55*s
    rear_top    = np.array([-base, +hw])
    rear_bottom = np.array([-base, -hw])
    tri = np.stack([tip, rear_top, rear_bottom], axis=0)
    c, s_ = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s_],[s_, c]], dtype=np.float32)
    tri = (tri @ R.T)
    tri[:,0] += center[0]; tri[:,1] += center[1]
    return tri
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors

def _blend_to_white(rgb, w):
    # w∈[0,1] 越大越亮
    return tuple((1.0 - w) * np.array(rgb) + w * 1.0)
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors
# 可选：用色谱的话启用
# from matplotlib import cm

def _draw_tri_arrows(ax, starts, disps, q90_len=18.0, color="#00BFFF",
                     alpha=0.98, width_scale=(0.7,1.4), big_thresh=390,
                     zorder=7, 
                     topk_indices=None,   # ← 新增：置信度高→低索引
                     loss=None,           # ← 新增：与 topk_indices 对应的分数
                     loss_min_is_better=True,   # loss 越小越好
                     alpha_range=(0.35, 0.95),  # 透明度范围
                     size_gain=0.5,             # 高权重→更大
                     use_colormap=False, cmap_name='turbo', edgecolor='black', lw=0.9):
    """
    starts, disps: 形状均为 (N,2)
    topk_indices: 长度 N 的索引数组，如 [7,3,0,5,...] 表示 starts[7] 置信度最高
    loss: 与 topk_indices 对应的分数数组，长度 N 或 <=N（会按提供的索引位置填入）
    """

    if len(starts) == 0:
        return 0.0

    starts = np.asarray(starts, dtype=np.float32)
    disps  = np.asarray(disps,  dtype=np.float32)
    n = len(starts)

    # === 位移幅度 → 基础尺寸（保留你原逻辑）===
    mag = np.linalg.norm(disps, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    q90 = np.quantile(mag[mag > 0], 0.90) if np.any(mag > 0) else 1.0
    rel = np.clip(mag / (q90 + 1e-6), 0.2, 2.0)
    base_sizes = q90_len * np.interp(rel, [0.2, 2.0], list(width_scale))

    # === 1) 排名权重（置信度高→低）===
    if topk_indices is not None and len(topk_indices) == n:
        topk_indices = np.asarray(topk_indices, dtype=int)
        ranks = np.empty(n, dtype=float)
        ranks[topk_indices] = np.arange(n, dtype=float)  # 0=最高, n-1=最低
        w_rank = 1.0 - ranks / (n - 1 + 1e-6)
    else:
        # 如果没给，就按当前顺序线性衰减
        w_rank = np.linspace(1.0, 0.0, n, endpoint=True).astype(np.float32)

    # === 2) loss 权重（越小越好或越大越好）===
    w_loss = None
    if (loss is not None) and (topk_indices is not None) and (len(topk_indices) > 0):
        # 构造长度 n 的 loss 向量，按 topk_indices 放置
        loss = np.asarray(loss, dtype=float)
        # 若 loss 数量不足 n，则仅填入前 len(loss) 个；其余置为均值
        full_loss = np.full(n, np.nan, dtype=float)
        put_len = min(len(loss), len(topk_indices))
        full_loss[topk_indices[:put_len]] = loss[:put_len]
        # 用均值填补 NaN（或用分位数）
        if np.any(np.isnan(full_loss)):
            fill_val = np.nanmean(full_loss) if np.any(~np.isnan(full_loss)) else 1.0
            full_loss = np.where(np.isnan(full_loss), fill_val, full_loss)

        # 用分位数做稳健归一化
        lo = np.percentile(full_loss, 5)
        hi = np.percentile(full_loss, 95)
        if hi - lo < 1e-9:
            w_loss = np.ones(n, dtype=float) * 0.5
        else:
            norm = np.clip((full_loss - lo) / (hi - lo), 0.0, 1.0)
            w_loss = 1.0 - norm if loss_min_is_better else norm  # 小更好→权重大

    # === 融合权重 ===
    if w_loss is None:
        w = w_rank
    else:
        w = 0.5 * w_rank + 0.5 * w_loss  # 简洁平均（可按需改成加权）

    # === 把权重映射到可视化 ===
    amin, amax = alpha_range
    alphas = amin + (amax - amin) * w
    sizes  = base_sizes * (1.0 + size_gain * w)

    # 颜色：默认固定主色，避免被浅白遮罩吃掉；可选色谱方案在下面
    if use_colormap:
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)
        face_rgbs = cmap(w)[:, :3]
    else:
        base_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
        face_rgbs = np.tile(base_rgb, (n, 1))

    # 画图顺序：先低权重，再高权重（高的盖在上层）
    order = np.argsort(w)  # 低→高
    for idx in order:
        (x, y)   = starts[idx]
        (dx, dy) = disps[idx]
        sz, a    = sizes[idx], alphas[idx]
        rgb      = face_rgbs[idx]
        ang = np.arctan2(dy, dx)
        tri = _rot_tri((x, y), ang, size_px=sz, aspect=0.55)
        ax.add_patch(Polygon(
            tri, closed=True,
            facecolor=rgb, edgecolor='white', linewidth=lw,
            alpha=a, antialiased=True, zorder=zorder
        ))

    big_ratio = float(((mag > float(big_thresh)).mean() * 100.0)) if len(mag) else 0.0
    return big_ratio

def _draw_tri_arrows_topk(ax, starts, disps, q90_len=18.0, color="#00BFFF",
                     alpha=0.98, width_scale=(0.7,1.4), big_thresh=390,
                     zorder=7, topk_indices=None,
                     alpha_range=(0.35, 0.95), size_gain=0.5):
    """
    topk_indices: 长度与 starts 相同，给出按置信度从大到小的索引顺序。
                  例如 [7, 3, 0, 5, ...]，表示 starts[7] 置信度最高。
    alpha_range : 置信度→透明度范围 (min, max)
    size_gain   : 置信度→尺寸增益比例 (0~1)
    """
    if len(starts) == 0:
        return 0.0

    starts = np.asarray(starts, dtype=np.float32)
    disps  = np.asarray(disps,  dtype=np.float32)

    # 位移幅度 → 基础尺寸（保持你的逻辑）
    mag = np.linalg.norm(disps, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    q90 = np.quantile(mag[mag > 0], 0.90) if np.any(mag > 0) else 1.0
    rel = np.clip(mag / (q90 + 1e-6), 0.2, 2.0)
    base_sizes = q90_len * np.interp(rel, [0.2, 2.0], list(width_scale))

    n = len(starts)

    # === 用 topk_indices 生成排名权重 w ∈ [0,1]（高→低）===
    if topk_indices is not None and len(topk_indices) == n:
        topk_indices = np.asarray(topk_indices, dtype=int)
        # ranks[i] = 0 表示最高；n-1 表示最低
        ranks = np.empty(n, dtype=float)
        ranks[topk_indices] = np.arange(n, dtype=float)
        w = 1.0 - ranks / (n - 1 + 1e-6)
    else:
        # 没提供就按当前顺序线性递减
        w = np.linspace(1.0, 0.0, n, endpoint=True).astype(np.float32)

    # 由置信度权重控制大小/透明度（颜色保持固定）
    amin, amax = alpha_range
    alphas = amin + (amax - amin) * w
    sizes  = base_sizes * (1.0 + size_gain * w)

    base_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
    face_rgbs = np.tile(base_rgb, (n, 1))

    # 画图顺序：先低置信度，后高置信度（高的盖在上面）
    order = np.argsort(w)  # 升序：低→高
    for idx in order:
        (x, y) = starts[idx]
        (dx, dy) = disps[idx]
        sz = sizes[idx]
        a  = alphas[idx]
        rgb = face_rgbs[idx]

        ang = np.arctan2(dy, dx)
        tri = _rot_tri((x, y), ang, size_px=sz, aspect=0.55)
        ax.add_patch(Polygon(
            tri, closed=True,
            facecolor=rgb, edgecolor='white', linewidth=0.8,  # 建议加一道深色描边，穿透浅白遮罩
            alpha=a, antialiased=True, zorder=zorder
        ))

    big_ratio = float(((mag > float(big_thresh)).mean() * 100.0)) if len(mag) else 0.0
    return big_ratio

def _draw_tri_arrows_bak(ax, starts, disps, q90_len=18.0, color=(0,0,0),
                     alpha=0.98, width_scale=(0.7,1.4), big_thresh=390,
                     zorder=7):
    if len(starts) == 0:
        return 0.0
    starts = np.asarray(starts, dtype=np.float32)
    disps  = np.asarray(disps,  dtype=np.float32)
    mag = np.linalg.norm(disps, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    q90 = np.quantile(mag[mag>0], 0.90) if np.any(mag>0) else 1.0
    rel = np.clip(mag / (q90 + 1e-6), 0.2, 2.0)
    sizes = q90_len * np.interp(rel, [0.2, 2.0], list(width_scale))
    for (x,y),(dx,dy),size in zip(starts, disps, sizes):
        ang = np.arctan2(dy, dx)
        tri = _rot_tri((x,y), ang, size_px=size, aspect=0.55)
        ax.add_patch(Polygon(tri, closed=True, facecolor=color,
                             edgecolor='white', linewidth=0.0,
                             alpha=alpha, antialiased=True, zorder=zorder))
    
    big_ratio = float(((mag > float(big_thresh)).mean() * 100.0)) if len(mag) else 0.0
    return big_ratio
def get_first_existing(d, keys):
    for k in keys:
        if k in d: return d[k]
    raise KeyError(f"找不到任何可用的特征键：{keys}")
# def pca_rgb(feat: np.ndarray, pca_model: Optional[PCA] = None,
#             ref_sign: Optional[np.ndarray] = None):
#     """
#     (C,H,W)->(H,W,3) RGB。若提供 pca_model，则仅 transform；
#     ref_sign 是长度为3的向量，指定每个主成分的符号(±1)，用于锁定颜色方向。
#     返回：rgb(uint8), pca_model, sign(3,)
#     """
#     C, H, W = feat.shape
#     X = feat.reshape(C, -1).T.astype(np.float32)
#     X = X - X.mean(axis=0, keepdims=True)
#     from sklearn.decomposition import PCA
#     if pca_model is None:
#         # pca_model = PCA(n_components=min(3, C), svd_solver='auto', whiten=False)
#         pca_model = PCA(n_components=3, svd_solver='auto', whiten=False, random_state=0)

#         Y = pca_model.fit_transform(X)
#     else:
#         Y = pca_model.transform(X)

#     # 计算/应用符号：以 ref_sign 为准；否则按当前数据均值正向
#     if ref_sign is None:
#         sign = np.ones(Y.shape[1], dtype=np.int8)
#         for i in range(Y.shape[1]):
#             if Y[:, i].mean() < 0: sign[i] = -1
#     else:
#         sign = np.asarray(ref_sign, dtype=np.int8)
#     Y = Y * sign  # 锁定方向

#     # 稳健归一(1–99分位) -> [0,1]
#     rgb = np.zeros((Y.shape[0], 3), dtype=np.float32)
#     for i in range(Y.shape[1]):
#         lo, hi = np.percentile(Y[:, i], [1, 99])
#         if hi - lo < 1e-6: rgb[:, i] = 0.5
#         else:              rgb[:, i] = np.clip((Y[:, i] - lo) / (hi - lo), 0, 1)

#     rgb = (rgb.reshape(H, W, 3) * 255).astype(np.uint8)
#     return rgb, pca_model, sign

def pca_rgb(feat: np.ndarray, pca_model: Optional[PCA] = None,
            ref_sign: Optional[np.ndarray] = None):
    """
    PCA 特征映射到固定伪彩色（偏绿）：
    - 保留彩虹细节，但整体偏绿，高值更亮
    - 返回 rgb(uint8), pca_model, sign(3,)
    """
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)

    if pca_model is None:
        pca_model = PCA(n_components=3, svd_solver='auto', whiten=False, random_state=0)
        Y = pca_model.fit_transform(X)
    else:
        Y = pca_model.transform(X)

    # 固定符号
    if ref_sign is None:
        sign = np.ones(Y.shape[1], dtype=np.int8)
        for i in range(Y.shape[1]):
            if Y[:, i].mean() < 0: sign[i] = -1
    else:
        sign = np.asarray(ref_sign, dtype=np.int8)
    Y = Y * sign

    # 归一化到 [0,1]
    norm_rgb = np.zeros_like(Y, dtype=np.float32)
    for i in range(3):
        lo, hi = np.percentile(Y[:, i], [1, 99])
        if hi - lo < 1e-6:
            norm_rgb[:, i] = 0.5
        else:
            norm_rgb[:, i] = np.clip((Y[:, i] - lo) / (hi - lo), 0, 1)

    # 固定彩虹偏绿配色
    rgb = np.zeros_like(norm_rgb)
    rgb[:,0] = norm_rgb[:,0]*0.6 + 0.3  # R，稍亮
    rgb[:,1] = norm_rgb[:,1]*0.7 + 0.3  # G，主导
    rgb[:,2] = norm_rgb[:,2]*0.5 + 0.2  # B，辅助高光

    rgb = (rgb.reshape(H,W,3) * 255).astype(np.uint8)
    return rgb, pca_model, sign




def visualize_query_trend_like_fig(
    data,
    save_path,
    ref_img_path,
    query_img_path,
    p2d_r,                  # (K,2) ref 像素（ref 原始尺寸）
    p2d_q_init,             # (B,K,2) 已按 target_size 缩放
    p2d_q_refined,          # (B,K,2) 已按 target_size 缩放
    w_r,
    p2d_q_refined_last=None,# (K,2) 或 (B,K,2)；仅用于在 Ref 侧“对应点”高亮
    target_size=(512, 288),
    # —— 抽样控制 —— #
    picked_idx=None,        # np.ndarray[Idx]，显式指定要展示的索引
    sample_k=None,          # 若不传 picked_idx，则随机抽样 K' 个点
    seed=0,                 # 抽样随机种子（可复现）
    # —— 外观控制 —— #
    arrow_q90_len=10.0,
    tri_alpha=0.98,
    # tri_color="#e64980", #pink
    tri_color="#00BFFF",  #yellow
    
    basin_alpha_out=0.4,
    basin_edge_width=1.6,
    show_basin=True,
    big_disp_thresh_px=390
):
    # 1) 读图
    ref_bgr = cv2.imread(ref_img_path); qry_bgr = cv2.imread(query_img_path)
    assert ref_bgr is not None and qry_bgr is not None, "读取图像失败"
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    qry = cv2.cvtColor(qry_bgr, cv2.COLOR_BGR2RGB)

    # 2) 尺度
    rH0, rW0 = ref.shape[:2]
    Wt, Ht = target_size
    ref = cv2.resize(ref, (Wt, Ht), interpolation=cv2.INTER_AREA)
    qry = cv2.resize(qry, (Wt, Ht), interpolation=cv2.INTER_AREA)
    w_r = cv2.resize(w_r[0], (Wt, Ht), interpolation=cv2.INTER_AREA) 
    
    f_q = get_first_existing(data, ["f_q","feat_q","q_feat","query_feat"]).numpy()
    f_r = get_first_existing(data, ["f_r","f_ref","f_render","feat_r","r_feat","ref_feat","render_feat"]).numpy()

    f_r = f_r / (np.linalg.norm(f_r, axis=0, keepdims=True) + 1e-6)
    
    # r_rgb, pca_ref, ref_sign = pca_rgb(f_r, pca_model=None, ref_sign=None)
    # q_rgb, _, _ = pca_rgb(f_q, pca_model=pca_ref, ref_sign=ref_sign)

    r_rgb, pca_model, ref_sign = pca_rgb(f_r)
    q_rgb, _, _ = pca_rgb(f_q, pca_model=pca_model, ref_sign=ref_sign)

    r_rgb = crop_16_9(r_rgb); q_rgb = crop_16_9(q_rgb)
    
    ref = cv2.resize(r_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
    qry = cv2.resize(q_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)

    # 3) 转 numpy & 形状
    p2d_r          = np.asarray(p2d_r,          dtype=np.float32)
    p2d_q_init     = np.asarray(p2d_q_init,     dtype=np.float32)
    p2d_q_refined  = np.asarray(p2d_q_refined,  dtype=np.float32)
    if p2d_q_init.ndim == 2:    p2d_q_init    = p2d_q_init[None, ...]
    if p2d_q_refined.ndim == 2: p2d_q_refined = p2d_q_refined[None, ...]
    B, K, _ = p2d_q_init.shape
    assert p2d_q_refined.shape[:2] == (B, K), "p2d_q_init 与 p2d_q_refined 形状不一致"
    # >>>>>>>>>>>>>>>>>>>>> 新增：若提供了 picked_idx，则直接按索引子集化 <<<<<<<<<<<<<<<<<<<<
    if picked_idx is not None:
        picked_idx = np.atleast_1d(np.asarray(picked_idx, dtype=np.int64))
        p2d_r = p2d_r[picked_idx]
        p2d_q_init    = p2d_q_init[:, picked_idx, :]
        p2d_q_refined = p2d_q_refined[:, picked_idx, :]
        if p2d_q_refined_last is not None:
            p2d_q_refined_last = np.asarray(p2d_q_refined_last, dtype=np.float32)
            if p2d_q_refined_last.ndim == 3:
                p2d_q_refined_last = p2d_q_refined_last[0]
            p2d_q_refined_last = p2d_q_refined_last[picked_idx]
    # 4) 统一抽样索引（关键！保证三者一致，支持多个）
    if picked_idx is None:
        idx_all = np.arange(K, dtype=np.int64)
    use_uncertain = True
    if use_uncertain:
        if picked_idx is None:
            idx_all = np.arange(K, dtype=np.int64)

            if sample_k is not None and sample_k < K:
                # ---- 使用 p2d_r 的坐标找到对应的 w_r 权重值 ----
                w_r = np.asarray(w_r, dtype=np.float32)  # 确保 w_r 是 numpy 数组
                w_r_flat = w_r.flatten()  # 将 w_r 扁平化为一维数组

                # 获取 p2d_r 的每个坐标，按照 (x, y) 索引对应到 w_r 中
                coords = p2d_r.astype(np.int32)  # 将坐标转为整数索引
                idx = coords[:, 1] * w_r.shape[1] + coords[:, 0]  # 通过 (y, x) 坐标索引到 w_r 扁平化数组

                # 根据 w_r 的权重升序排序，找到最小的 5 个点的索引
                sorted_idx = np.argsort(w_r_flat[idx])  # 排序后，选择权重最小的点
                picked_idx = sorted_idx[:sample_k]  # 选择前 sample_k 个最小的权重点

                # 通过这些索引，筛选出对应的 p2d_r 和其他相关数组
                p2d_r = p2d_r[picked_idx]
                p2d_q_init    = p2d_q_init[:, picked_idx, :]
                p2d_q_refined = p2d_q_refined[:, picked_idx, :]
                if p2d_q_refined_last is not None:
                    p2d_q_refined_last = p2d_q_refined_last[0][picked_idx]

            else:
                picked_idx = idx_all
    else:
        if sample_k is not None and sample_k < K:
            rng = np.random.default_rng(seed)

            # --- 定义边界阈值（像素/归一化都行，这里假设 p2d_r 在像素坐标系里）---
            x_min, y_min = np.min(p2d_r, axis=0)
            x_max, y_max = np.max(p2d_r, axis=0)
            margin_x = 0.2 * (x_max - x_min)   # 留5%边界
            margin_y = 0.2 * (y_max - y_min)

            # --- 筛掉边缘点 ---
            mask = (
                (p2d_r[:,0] > x_min + margin_x) &
                (p2d_r[:,0] < x_max - margin_x) &
                (p2d_r[:,1] > y_min + margin_y) &
                (p2d_r[:,1] < y_max - margin_y)
            )
            valid_idx = idx_all[mask]
            if len(valid_idx) < sample_k:   # 如果剩余太少，就退回所有点
                valid_idx = idx_all

            # --- Farthest Point Sampling (FPS) ---
            pts = p2d_r[valid_idx]
            first = rng.integers(0, len(valid_idx))
            picked = [valid_idx[first]]
            dists = np.full(len(valid_idx), np.inf, dtype=np.float32)

            for _ in range(1, sample_k):
                last_pt = pts[np.where(valid_idx==picked[-1])[0][0]]
                dist2 = np.sum((pts - last_pt) ** 2, axis=1)
                dists = np.minimum(dists, dist2)
                next_idx = int(np.argmax(dists))
                picked.append(valid_idx[next_idx])

            picked_idx = np.array(picked, dtype=np.int64)

        else:
            picked_idx = idx_all
        picked_idx = np.atleast_1d(np.asarray(picked_idx, dtype=np.int64))
        p2d_r = p2d_r[picked_idx]                         # (K',2)
        p2d_q_init    = p2d_q_init[:, picked_idx, :]       # (B,K',2)
        p2d_q_refined = p2d_q_refined[:, picked_idx, :]    # (B,K',2)
        if p2d_q_refined_last is not None:
            p2d_q_refined_last = np.asarray(p2d_q_refined_last, dtype=np.float32)
            if p2d_q_refined_last.ndim == 3:
                p2d_q_refined_last = p2d_q_refined_last[0]
            p2d_q_refined_last = p2d_q_refined_last[picked_idx]  # (K',2)
    # 允许 int / list / ndarray

    # 5) 坐标缩放（Ref->可视化尺寸）
    p2d_r_v = p2d_r.copy()

    # 6) 取第 0 帧
    pq0 = p2d_q_init.reshape(-1, 2)       # -> (200,2)    # (K',2)
    pq1 = p2d_q_refined.reshape(-1, 2)  # (K',2)
    disp = pq1 - pq0

    # 7) 过滤非法/越界/NaN（统一过滤，三者一起）
    def _valid_xy(a):
        return np.isfinite(a).all(axis=1)
    def _inb(a):
        x,y = a[:,0], a[:,1]
        return (x>=0)&(x<Wt)&(y>=0)&(y<Ht)
    m = _valid_xy(pq0) & _valid_xy(pq1) & _inb(pq0) & _inb(pq1)
    pq0, pq1, disp = pq0[m], pq1[m], disp[m]
    p2d_r_v_vis = p2d_r_v
    refined_last_vis = p2d_q_refined_last if p2d_q_refined_last is not None else None
    # —— 只保留“更接近 refined_last_vis”的位移 —— #
    # if refined_last_vis is not None:
    #     R = np.asarray(refined_last_vis, dtype=np.float32).reshape(-1, 2)  # (M,2)

    #     # 对每个向量，计算其起点/终点到“最近的 refined 点”的距离
    #     # d0: 起点 pq0 到最近 refined 的距离；d1: 终点 pq1 到最近 refined 的距离
    #     diff0 = pq0[:, None, :] - R[None, :, :]         # (N,M,2)
    #     diff1 = pq1[:, None, :] - R[None, :, :]         # (N,M,2)
    #     d0 = np.sqrt((diff0**2).sum(axis=2)).min(axis=1)  # (N,)
    #     d1 = np.sqrt((diff1**2).sum(axis=2)).min(axis=1)  # (N,)

    #     # 仅保留“确实更靠近”的那些位移（可选提高严苛度：& ((d0 - d1) >= 1.0)）
    #     closer_mask = d1 < d0

    #     pq0, pq1, disp = pq0[closer_mask], pq1[closer_mask], disp[closer_mask]

    # 若全被滤掉，给出可读提示
    if len(pq0) == 0:
        fig = plt.figure(figsize=((Wt*2)/100.0, Ht/100.0), dpi=120)
        ax = fig.add_subplot(1,1,1); ax.axis('off')
        ax.imshow(qry); ax.text(20,40,"No valid sampled matches after masking.",
                                color='white', fontsize=14, weight='bold',
                                bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=6))
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"⚠️ 采样后有效点为 0，已输出提示图：{save_path}")
        return

    # 8) 收敛盆地（convex hull of refined points）
    

    # 9) 画布
    fig = plt.figure(figsize=((Wt*2)/100.0, Ht/100.0), dpi=120)

    # 左：Ref
    axL = fig.add_subplot(1,2,1)
    axL.imshow(ref, extent=(0,Wt,Ht,0), interpolation='nearest')
    axL.set_xlim(0,Wt); axL.set_ylim(Ht,0)
    axL.set_xticks([]); axL.set_yticks([]); axL.set_frame_on(False)
    for s in axL.spines.values(): s.set_visible(False)
    # axL.text(12,22,"Reference",color='white',fontsize=11,weight='bold',
    #          bbox=dict(facecolor=(0,0,0,0.38), edgecolor='none', pad=3), zorder=9)
    num_pts = len(p2d_r_v_vis)  # 或 len(refined_last_vis)

    # 生成 num_pts 个互异颜色（tab20/glasbey都可以；tab20自带）
    def vivid_colors(n, seed=0):
        import matplotlib.colors as mcolors
        # 黄金分割遍历色相，固定高饱和&高明度 → 颜色更“鲜”
        phi = 0.61803398875
        h = (np.arange(n) * phi + 0.15) % 1.0
        s = np.full(n, 0.90)   # 0.85~1.00 越大越饱和
        v = np.full(n, 0.98)   # 0.95~1.00 越大越亮
        hsv = np.stack([h, s, v], axis=1)
        return mcolors.hsv_to_rgb(hsv)       # (n,3) in [0,1]

    # —— 在两处 scatter 之前（替换你原来的 tab20 方案）——
    num_pts = len(p2d_r_v_vis)
    point_colors = vivid_colors(num_pts)
    
    # 在 Ref 面板高亮“与（抽样后）对应”的 ref 点
    axL.scatter(p2d_r_v_vis[:, 0], p2d_r_v_vis[:, 1],
            s=200, c=point_colors, edgecolors='white', linewidths=5,
            alpha=0.98, zorder=8)

    # 右：Query
    axR = fig.add_subplot(1,2,2)
    axR.imshow(qry, extent=(0,Wt,Ht,0), interpolation='nearest')
    axR.set_xlim(0,Wt); axR.set_ylim(Ht,0)
    axR.set_xticks([]); axR.set_yticks([]); axR.set_frame_on(False)
    for s in axR.spines.values(): s.set_visible(False)
    # Query 侧
    if refined_last_vis is not None:
        axR.scatter(refined_last_vis[:, 0], refined_last_vis[:, 1],
                    s=200, c=point_colors, edgecolors='white', linewidths=5,
                    alpha=0.98, zorder=10)
    for i in range(int(len(m)/100)):
        hull = cv2.convexHull(p2d_q_init[:,i][m[i*100:(i+1)*100]]).astype(np.float32)  # (M,1,2)
        basin_poly = hull[:,0,:]
        axR.add_patch(Polygon(basin_poly, closed=True,
                            facecolor=(1, 1, 1, basin_alpha_out),  # ← 直接给收敛域填充半透明白
                            edgecolor='white', linewidth=0.0,
                            antialiased=False, zorder=3))
    # 只在“起点”画小三角，指向 refined
    if refined_last_vis is not None:
        R = np.asarray(refined_last_vis, dtype=np.float32).reshape(-1, 2)  # (M,2)
        diff0 = pq0[:, None, :] - R[None, :, :]
        diff1 = pq1[:, None, :] - R[None, :, :]
        d0 = np.sqrt((diff0**2).sum(axis=2)).min(axis=1)  # 起点到最近 refined 距离
        d1 = np.sqrt((diff1**2).sum(axis=2)).min(axis=1)  # 终点到最近 refined 距离
        closer_mask = d1 < d0  # True=朝 refined，更近；False=离开

        # 正确趋势箭头：大、颜色深
        # _draw_tri_arrows_(axR, pq0, disp,
        #          q90_len=arrow_q90_len*1.5,
        #          color="#FFFF00",
        #          alpha=1,
        #          big_thresh=big_disp_thresh_px,
        #          zorder=9,
        #          topk_indices=topk_indices )
        # topk_indices: pq0 置信度从高到低的索引顺序
# loss         : 与 topk_indices 对应的分数（长度可等于 N 或小于 N）
        # _draw_tri_arrows(
        #     axR, pq0, disp,
        #     q90_len=arrow_q90_len*2.0,
        #     color="#FFFF00",              # 建议深色主体 + 明显描边穿透浅白遮罩
        #     alpha=0.98,
        #     big_thresh=big_disp_thresh_px,
        #     zorder=9,
        #     topk_indices=topk_indices,
        #     loss=loss,                    # ← 新增
        #     loss_min_is_better=True,      # 你的 loss 越小越好就保持 True
        #     use_colormap=False,           # 若想用色谱区分改为 True
        #     edgecolor='black', lw=1.2
        # )


        _draw_tri_arrows_bak(axR, pq0, disp,
                         q90_len=arrow_q90_len*2.0,   # 放大
                         color="#0000FF", alpha=tri_alpha,
                         big_thresh=big_disp_thresh_px,
                         zorder=9)

        # # 错误趋势箭头：小、颜色浅
        # _draw_tri_arrows(axR, pq0[~closer_mask], disp[~closer_mask],
        #                  q90_len=arrow_q90_len*1.6,   # 缩小
        #                  color="#FFFF00", alpha=1, # 变浅
        #                  big_thresh=big_disp_thresh_px,
        #                  zorder=7)
    else:
        big_ratio = _draw_tri_arrows(axR, pq0, disp,
                                     q90_len=arrow_q90_len,
                                     color="#0000FF", alpha=tri_alpha,
                                     big_thresh=big_disp_thresh_px,
                                     zorder=8)

    # axR.text(0.02, 0.08, f"Convergence basin in query\n{big_ratio:.0f}% at >{int(big_disp_thresh_px)}px",
    #      transform=axR.transAxes, clip_on=False,  # ← 相对坐标、防裁切
    #      color='white', fontsize=11, weight='bold',
    #      bbox=dict(facecolor=(0,0,0,0.38), edgecolor='none', pad=3), zorder=9)

    fig.patch.set_alpha(0)
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0)
    fig.savefig(save_path, dpi=120, pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"✅ saved: {save_path} | 可视点数: {len(pq0)}")


seq_name = "6444"
pt_files = [
    os.path.join("/home/amax/Documents/datasets/fea",seq_name ,"0_feature.pt"),
    os.path.join("/home/amax/Documents/datasets/fea",seq_name ,"1_feature.pt"),
    os.path.join("/home/amax/Documents/datasets/fea",seq_name ,"2_feature.pt")
    
]
def crop_16_9(img):
        H, W = img.shape[:2]
        th = int(round(W*9/16))
        if th <= H: return img[:th,:]
        tw = int(round(H*16/9))
        x0 = max(0,(W-tw)//2); return img[:,x0:x0+tw]
save_dir = os.path.join("/home/amax/Documents/datasets/fea", seq_name)
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"pyramid_level_fine.png")
ref_img_path =  os.path.join("/home/amax/Documents/datasets/fea",seq_name ,"0_ref.png")
query_img_path = os.path.join("/home/amax/Documents/datasets/fea",seq_name ,"0_query.png")
print("ref_img_path", ref_img_path)
data = torch.load(pt_files[0], map_location="cpu")
p2d_q_init = data["p2d_query"][:].numpy()     
scale_pt_q =  1920 / 240   
p2d_q_init =  scale_pt_q*p2d_q_init

# --- img
ref = cv2.cvtColor(cv2.imread(ref_img_path),  cv2.COLOR_BGR2RGB)
qry = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
ref = crop_16_9(ref); qry = crop_16_9(qry)

data = torch.load(pt_files[-1], map_location="cpu")
scale_pt_q =  1920 / 960   
p2d_r  = data["p2d_render"].numpy()   *scale_pt_q       
p2d_qr = data["p2d_query_refined"].numpy()    *scale_pt_q   
p2d_qr_last = data["p2d_query_refined_last"].numpy()    *scale_pt_q   
w_r = data["w_r"].numpy() 

target_size = (1920, 1080)
# 先手工在 Query 上点 5 个点，得到索引
picked_idx = pick_indices_from_qr_last(
    query_img_path=query_img_path,
    target_size=target_size,
    p2d_qr_last=p2d_qr_last,  # 注意：这里就是 data["p2d_query_refined_last"] * scale_pt_q
    k=5,
    tol_px=12,   # 可调
    dedup=True
)

print("手选索引:", picked_idx.tolist())

# 用手选索引出图；此时 sample_k 设为 None（忽略）
visualize_query_trend_like_fig(data,
    save_path, ref_img_path, query_img_path,
    p2d_r, p2d_q_init, p2d_qr, w_r,
    p2d_q_refined_last=p2d_qr_last,
    target_size=target_size,
    picked_idx=picked_idx,   # ✅ 关键：传入你手选的索引
    sample_k=None,           # ✅ 不再内部采样
    seed=24
)