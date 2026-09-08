# costs.py 模块说明

`pixloc/pixlib/geometry/costs.py` 实现基于特征重投影的绝对位姿代价，并支持深度（DSM）与姿态角（roll/pitch）先验约束，用于高斯-牛顿优化。

---

## 1. 模块结构概览

| 类型 | 名称 | 说明 |
|------|------|------|
| JIT 函数 | `J_project` | 3D→2D 投影雅可比 |
| JIT 函数 | `transform_p3d` | 位姿变换 3D 点（体→相机） |
| JIT 函数 | `project_p3d` | 相机系 3D 点投影到 2D + 有效性 |
| JIT 函数 | `skew_symmetric` | 向量→反对称矩阵 |
| 类 | `DirectAbsoluteCost2` | 主代价类：重投影 + 深度 + 角度 |

---

## 2. JIT 工具函数

### 2.1 `J_project(p3d)`

- **作用**：计算透视投影 \( (x,y,z) \to (x/z, y/z) \) 对 3D 点的雅可比。
- **输入**：`p3d`，形状 `(..., 3)`。
- **输出**：形状 `(..., 2, 3)`，对应 \(\partial (u,v) / \partial (x,y,z)\)。
- **常数**：`z.clamp(min=1e-3)` 防止除零。

### 2.2 `transform_p3d(body2view_pose_data, p3d)`

- **作用**：用位姿把 3D 点从“体坐标系”变到“相机坐标系”。
- **位姿格式**：前 9 维为 3×3 旋转（行展平），后 3 维为平移；即 `p3d @ R^T + t`。

### 2.3 `project_p3d(camera_data, p3d)`

- **作用**：相机系 3D 点透视投影到 2D，并判断是否在图像内、深度为正。
- **相机格式**：`[w, h, fx, fy, cx, cy]`（前 2 为尺寸，后 4 为内参）。
- **返回**：`(p2d, valid)`，`valid` 为深度 > eps 且在图像范围内的掩码。
- **常数**：`eps=1e-4` 用于深度正性与 clamp。

### 2.4 `skew_symmetric(v)`

- **作用**：由 3 维向量得到反对称矩阵 \([v]_\times\)，用于李代数/旋转求导。

---

## 3. 类 `DirectAbsoluteCost2`

### 3.1 初始化与属性

```python
def __init__(self):
    self.iter = 2           # 迭代计数，用于诊断打印（如 iter<=3 时打融合信息）
    self._ecef_to_wgs84     # pyproj ECEF→WGS84（懒初始化）
    self._wgs84_to_ecef     # pyproj WGS84→ECEF（懒初始化）
    self._dsm_cache         # DSM 栅格与 geotransform 缓存
    self.use_gpu_dsm        # 是否用 GPU 做 DSM 插值（根据 torch.cuda）
```

### 3.2 坐标与辅助方法

| 方法 | 功能 |
|------|------|
| `_poses_to_wgs84(pose_data_q, render_T_ecef)` | 候选位姿 + ECEF 原点 → 每条位姿的 WGS84 (lon, lat, alt, roll, pitch, yaw) |
| `_pose_translations_ecef(pose_data_q, origin, dd)` | 位姿平移转到 ECEF：\(C_{local} + origin + dd\) |
| `_center_points_from_wgs84_poses(wgs84_poses, depth, K, mul)` | 由 WGS84 姿态 + 中心深度 → 像主射线落点 WGS84 坐标 |
| `_center_points_from_pose_ecef(pose_data_q, depth, K, origin, dd, mul)` | 由 ECEF 下位姿 + 深度直接算中心射线落点 WGS84（避免欧拉往返） |
| `_query_dsm_heights(wgs84_points, dsm_path, proj_crs, use_gpu)` | 按 WGS84 经纬度在 DSM 上双线性插值得到高程；支持 CPU/GPU |

### 3.3 特征插值与鲁棒核

- **`bilinear_grid_sample(im, grid, align_corners)`**  
  自定义双线性插值（可替换 `F.grid_sample`），便于控制边界与梯度。

- **`interpolate_feature_map(feature, p2d, return_gradients)`**  
  在特征图 `feature` 上对 2D 点 `p2d` 插值；可选返回图像梯度。  
  - 有效区域：`interpolation_pad=4`，即 `p2d` 在 `[pad, size-pad-1]` 内有效。  
  - 格网坐标：`pts.clamp(-2, 2)`，`align_corners=True`。

- **`loss_fn1(cost, alpha=0.0, truncate=0.1, eps=1e-7, derivatives=True)`**  
  广义鲁棒核：输入标量/逐元素 cost，输出  
  - loss（标度由 `truncate**2` 保持量纲）、  
  - 一阶导 `loss_d1`（用于鲁棒权重）、  
  - 二阶导占位 `loss_d2`（当前为 0）。  
  - **alpha**：0 → log 核，2 → 平方；默认 **alpha=0**，**truncate=0.1**。

---

## 4. 主接口：`residual_jacobian_batch_quat`

对**多候选查询位姿**在一个 batch 内计算：

1. **重投影残差** + 雅可比；  
2. 若提供 ECEF、gt_depth、dsm_path，则增加**深度先验**（中心射线落点高度 − DSM）与**角度先验**（roll/pitch − gt）；  
3. 统一过鲁棒核得到权重，再合并梯度与 Hessian，供高斯-牛顿使用。

### 4.1 签名

```python
def residual_jacobian_batch_quat(
    self, pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D,
    c_ref, c_query, p2d_r, visible_r,
    gt_depth=None, gt_roll=None, gt_pitch=None,
    dd=None, mul=None, origin=None, dsm_path=None, render_T_ecef=None,
    w_depth=None, w_angle=None
)
```

### 4.2 输入说明

| 参数 | 含义 |
|------|------|
| `pose_data_q` | 查询帧候选位姿，形状含 `[1, num_init_pose, 12]`（9 旋转 + 3 平移） |
| `f_r`, `f_q` | 参考帧 / 查询帧特征图 |
| `pose_data_r`, `cam_data_r` | 参考帧位姿与相机（本接口内主要用相机） |
| `cam_data_q` | 查询帧相机 `[w, h, fx, fy, cx, cy]` |
| `p3D` | 参考帧下 3D 点（同一批点投影到查询帧） |
| `c_ref`, `c_query` | 参考/查询帧上用于置信度插值的图（如不确定性图） |
| `p2d_r`, `visible_r` | 参考帧 2D 点与可见性 |
| `gt_depth` | 查询帧中心深度（米），用于深度约束与中心射线 |
| `gt_roll`, `gt_pitch` | 姿态先验（度），转弧度后做残差 |
| `origin`, `dd` | 局部到 ECEF 的平移；若为 None 则用 `render_T_ecef` |
| `mul` | 深度单位缩放（如米→与位姿一致） |
| `dsm_path` | DSM GeoTIFF 路径，用于地面高度查询 |
| `render_T_ecef` | 渲染/世界原点在 ECEF 下的坐标 |
| `w_depth`, `w_angle` | 保留兼容，**当前未用**；融合权重改为**自动平衡** |

**深度/角度约束启用条件**：`render_T_ecef`、`gt_depth`、`dsm_path` 均非空。

### 4.3 计算流程简述

1. **参考帧**：在 `p2d_r` 上插值 `f_r`，得到参考特征与有效掩码；`p3D` 与参考特征按掩码处理并复制到 `num_init_pose` 份。
2. **查询帧**：用每个候选位姿把 `p3D` 变到查询相机并投影得到 `p2d_q`；在 `p2d_q` 上插值 `f_q` 及特征梯度。
3. **重投影残差**：`res = fp_q - fp_r`（特征差），再得到 per-point cost：`cost_proj = (res^2).sum(dim=C)`。
4. **重投影雅可比**：  
   \(J = J_{feat} \cdot J_{p2d/p3d} \cdot J_{p3d/pose}\)  
   其中 \(J_{p3d/pose}\) 含旋转（李代数）与平移，与 `J_project`、相机内参组合得到 6DoF 雅可比。
5. **深度约束**（在启用时）：  
   - 残差：\(r = (z_{center} - h_{dsm}) / \texttt{DEPTH\_NORM}\)（米→无量纲），  
   - 雅可比：解析 \([(u\times v)^T,\ -u^T]\)（u：ECEF 天向，v：中心射线方向），同除 `DEPTH_NORM`。  
   - `DEPTH_NORM = 10.0`（米），固定常数。
6. **角度约束**（在启用时）：  
   - 残差：`(roll - gt_roll)`, `(pitch - gt_pitch)` 弧度，并做 \([-π,π]\) 规整。  
   - 雅可比：由 `R_pose_in_enu` 的 roll/pitch 解析式对相机系旋转 ω 求导，平移部分为 0。
7. **鲁棒核与权重**：  
   - 将 `cost_proj`、`cost_depth`、`cost_angle` 拼成 `cost_all`，一次 `loss_fn1` 得到统一的 `w_loss_proj`、`w_loss_depth`、`w_loss_angle`。
8. **投影权重**：  
   - 有效 + 可见 + 鲁棒权重 + 不确定性权重（在 `c_ref`/`c_query` 上插值）得到 `weights_proj`。
9. **梯度与 Hessian**：  
   - 投影：`grad_proj`、`Hess_proj`；  
   - 深度/角度：`grad_depth`、`grad_angle`，`Hess_depth`、`Hess_angle`。  
   - **自动权重平衡**：  
     - \(w_{d\_auto} = \|grad_{proj}\| / (\|grad_{depth}\| + \epsilon)\)，  
     - \(w_{a\_auto} = \|grad_{proj}\| / (\|grad_{angle}\| + \epsilon)\)，  
     - 并 clamp 到 `[0.01, 100]`。  
   - 合并：  
     - `grad = grad_proj + w_d_auto*grad_depth + w_a_auto*grad_angle`，  
     - `Hess` 同理。
10. **返回**：  
    - `-grad`（最小化 loss 的上升方向）、`Hess`、`w_loss`（投影鲁棒权重）、`valid_query`、`p2d_q`、`cost`（仅投影部分）。

### 4.4 返回值

| 返回值 | 含义 |
|--------|------|
| 第 1 个 | `-grad`：负梯度，形状 `[1, num_init_pose, 6]` |
| 第 2 个 | `Hess`：Hessian，形状 `[1, num_init_pose, 6, 6]` |
| 第 3 个 | `w_loss`：投影鲁棒权重 |
| 第 4 个 | `valid_query`：查询帧有效+可见掩码 |
| 第 5 个 | `p2d_q`：查询帧 2D 投影点 |
| 第 6 个 | `cost`：投影 cost（不含深度/角度） |

---

## 5. 固定常数汇总

（不参与训练，仅作实现与调参参考。）

| 位置 | 常数 | 说明 |
|------|------|------|
| `J_project` | `z.clamp(min=1e-3)` | 投影雅可比防除零 |
| `project_p3d` | `eps=1e-4` | 深度正性、clamp |
| `loss_fn1` | `alpha=0.0` | 鲁棒核为 log 型 |
| `loss_fn1` | `truncate=0.1` | 鲁棒核尺度 |
| `loss_fn1` | `eps=1e-7` | 防除零 |
| `interpolate_feature_map` | `interpolation_pad=4` | 有效区边界 |
| `interpolate_feature_map` | `pts.clamp(-2, 2)` | 格网坐标范围 |
| `residual_jacobian_batch_quat` | `DEPTH_NORM=10.0` | 深度残差/雅可比归一化（米） |
| 角度 roll/pitch 求导 | `clamp(±(1-1e-7))`, `+1e-8` | asin/根号防除零 |
| `DirectAbsoluteCost2` | `self.iter=2` | 迭代/诊断 |
| `_center_points_*` 缺 K | `[1920,1080,1350,1350,960,540]` | 默认内参 |

---

## 6. 依赖

- **PyTorch**：张量、JIT、CUDA（可选 DSM）
- **pyproj**：ECEF ↔ WGS84
- **GDAL (osgeo.gdal)**：读 DSM GeoTIFF
- **scipy**：`map_coordinates`（CPU DSM 插值）、`Rotation`
- **numpy**
- 本包：`Pose`, `Camera`；`optimization.J_normalization`, `skew_symmetric`；`interpolation.Interpolator`；`transform.ECEF_to_WGS84`, `WGS84_to_ECEF`, `get_rotation_enu_in_ecef`

---

## 7. 使用注意

1. **坐标系**：位姿为世界/体到相机（w2c）；代码中会转成 c2w 做中心射线、ECEF 等。ENU 与相机朝向的约定见注释（R_c2w 的 1、2 列取反后对应 ENU）。
2. **深度单位**：`gt_depth` 为米；若与位姿尺度不一致，用 `mul` 换算；深度约束的雅可比与残差已用 `DEPTH_NORM` 归一化。
3. **诊断**：当 `self.iter <= 3` 时会打印查询帧 WGS84、中心点、DSM 高度以及约束融合权重、梯度占比，便于调试。
4. **DSM**：首次读取会缓存为 `.npy` 与 geotransform `.txt`；投影坐标系默认 `EPSG:4547`，可在 `_query_dsm_heights` 中改 `proj_crs`。
