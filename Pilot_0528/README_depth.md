# 深度约束（DSM高度误差）实现详解

## 1. 概述

深度约束是一种基于DSM（数字表面模型）地图的相机位姿优化约束。其核心思想是：**根据当前相机位姿和深度值反投影得到的3D点高度，应该与DSM地图上对应位置的高度一致**。如果不一致，说明相机位姿估计有误。

### 1.1 基本原理

对于图像中心点，给定：
- **相机位姿** (`pose_data_q`): 包含旋转矩阵R和平移向量t
- **深度值** (`gt_depth`): 图像中心点的深度（米）
- **相机内参** (`cam_data_q`): 包含焦距fx, fy和主点cx, cy

我们可以：
1. 反投影得到3D点的世界坐标
2. 转换到WGS84坐标系获取经纬度
3. 转换到EPSG:4547坐标系（DSM地图坐标系）
4. 从DSM地图读取该位置的高度值
5. 比较3D点的高度和DSM高度，如果不相等，说明pose有问题

### 1.2 数学表达

深度约束的残差定义为：

```
res_depth = height_3d_wgs84 - height_dsm
```

其中：
- `height_3d_wgs84`: 根据pose和depth反投影得到的3D点的WGS84高度
- `height_dsm`: 从DSM地图读取的对应位置的高度

优化目标是最小化 `res_depth²`，从而约束相机位姿。

## 2. 完整流程

### 2.1 流程图

```
图像中心点 (cx, cy) + 深度值 depth
    ↓
反投影到相机坐标系 (p3d_cam)
    ↓
变换到世界坐标系 (p3d_world, ENU局部坐标系)
    ↓
坐标转换链：
ENU → ECEF → WGS84 (获取经纬度) → EPSG:4547
    ↓
从DSM地图读取高度 (双线性插值)
    ↓
计算高度误差: res_depth = height_3d - height_dsm
    ↓
计算Jacobian: J_depth = ∂height/∂pose
    ↓
计算Gradient和Hessian
    ↓
与投影约束合并
```

### 2.2 详细步骤

#### Step 1: 从图像中心点反投影到3D点（相机坐标系）

**代码位置**: `costs.py` 第586-611行

```python
# 获取图像中心点坐标（主点）
cx_all = cam_data_q[batch_idx, :, 4]  # [num_init_pose]
cy_all = cam_data_q[batch_idx, :, 5]  # [num_init_pose]
fx_all = cam_data_q[batch_idx, :, 2]  # [num_init_pose]
fy_all = cam_data_q[batch_idx, :, 3]  # [num_init_pose]

# 构建图像中心点的齐次坐标
p2d_center = torch.stack([cx_all, cy_all, torch.ones_like(cx_all)], dim=-1)

# 反投影: p3d_cam = depth * K_inv @ p2d_center
# 其中 K_inv = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]
p3d_cam = depth * K_inv @ p2d_center
```

**数学公式**:
```
p3d_cam = depth * [ (cx - cx)/fx, (cy - cy)/fy, 1 ]^T
        = depth * [ 0, 0, 1 ]^T
        = [0, 0, depth]^T
```

实际上，图像中心点反投影后，在相机坐标系下的归一化坐标为 `[0, 0, 1]`，乘以深度后得到 `[0, 0, depth]`。

#### Step 2: 变换到世界坐标系（ENU局部坐标系）

**代码位置**: `costs.py` 第613-621行

```python
# 提取pose的旋转和平移
R_w2c = pose_data_q[batch_idx, :, :9].view(num_init_pose, 3, 3)
t_w2c = pose_data_q[batch_idx, :, 9:].view(num_init_pose, 3)

# 计算Camera-to-World变换
R_c2w = R_w2c.transpose(-1, -2)  # R_c2w = R_w2c^T
t_c2w = -R_c2w @ t_w2c           # t_c2w = -R_c2w @ t_w2c

# 变换到世界坐标系
p3d_world = R_c2w @ p3d_cam + t_c2w
```

**数学公式**:
```
p3d_world = R_c2w @ p3d_cam + t_c2w
```

其中 `p3d_world` 是在ENU（东-北-天）局部坐标系下的坐标。

#### Step 3: 坐标转换链

**代码位置**: `costs.py` 第623-667行

##### 3.1 ENU → ECEF

```python
# 构建ENU坐标系的旋转基
up = [cos(lon)*cos(lat), sin(lon)*cos(lat), sin(lat)]
east = [-sin(lon), cos(lon), 0]
north = cross(up, east)
rot_enu_in_ecef = [east, north, up]

# 转换到ECEF
p3d_ecef = rot_enu_in_ecef @ p3d_world + t_ecef_ref
```

##### 3.2 ECEF → WGS84

```python
lon, lat, alt = transformer_ecef_to_wgs84.transform(
    p3d_ecef[0], p3d_ecef[1], p3d_ecef[2]
)
```

##### 3.3 WGS84 → EPSG:4547

```python
x_4547, y_4547 = transformer_wgs84_to_4547.transform(lon, lat)
```

EPSG:4547是DSM地图使用的投影坐标系（通常是高斯-克吕格投影）。

#### Step 4: 从DSM读取高度（双线性插值）

**代码位置**: `costs.py` 第669-713行

```python
# 初始化DSM插值器（使用双线性插值）
interpolator = RegularGridInterpolator(
    (y_coords, x_coords), dsm_data_flipped,
    method='linear',  # 双线性插值
    bounds_error=False,
    fill_value=nodata
)

# 批量插值所有点
sample_coords = np.column_stack([y_4547_all, x_4547_all])
dsm_heights_all = interpolator(sample_coords)
```

**双线性插值原理**:
对于坐标 `(x, y)`，找到其周围的4个像素点：
- `(x0, y0)`, `(x1, y0)`, `(x0, y1)`, `(x1, y1)`

插值公式：
```
h = h00 * (1-dx) * (1-dy) + h10 * dx * (1-dy) + h01 * (1-dx) * dy + h11 * dx * dy
```

其中 `dx = (x - x0) / (x1 - x0)`, `dy = (y - y0) / (y1 - y0)`

#### Step 5: 计算高度误差

**代码位置**: `costs.py` 第722-727行

```python
# ENU坐标系下的相对高度
height_3d_enu = p3d_world[:, 2]

# 转换为WGS84绝对高度
height_3d_wgs84 = height_3d_enu + alt_ref_depth

# 高度残差
res_depth = height_3d_wgs84 - dsm_height_tensor
```

**注意**: 
- `p3d_world[2]` 是ENU坐标系下的Z坐标（相对高度）
- 需要加上参考原点的高度 `alt_ref_depth` 得到WGS84绝对高度
- DSM高度也是WGS84绝对高度，可以直接比较

## 3. Jacobian链式计算

### 3.1 高度对pose的Jacobian

**代码位置**: `costs.py` 第729-743行

我们需要计算 `∂height/∂pose`，其中 `height = p3d_world[2]`。

#### 3.1.1 对旋转的Jacobian

```
height = (R_c2w @ p3d_cam + t_c2w)[2]
       = R_c2w[2, :] @ p3d_cam + t_c2w[2]
```

对旋转的导数：
```
∂height/∂R = ∂(R_c2w @ p3d_cam)/∂R
           = R_c2w @ skew_symmetric(p3d_cam)
```

其中 `skew_symmetric(p3d_cam)` 是反对称矩阵：
```
[  0    -z    y ]
[  z     0   -x ]
[ -y     x    0 ]
```

**代码实现**:
```python
R_c2w_row2 = R_c2w[:, 2:3, :]  # R_c2w的第3行
skew_p3d_cam = skew_symmetric(p3d_cam)
J_height_rot = R_c2w_row2 @ skew_p3d_cam  # [num_init_pose, 3]
```

#### 3.1.2 对平移的Jacobian

```
∂height/∂t = ∂t_c2w[2]/∂t_w2c
            = ∂(-R_c2w @ t_w2c)[2]/∂t_w2c
            = -R_c2w[2, :]
```

**代码实现**:
```python
J_height_tran = -R_c2w_row2.squeeze(1)  # [num_init_pose, 3]
```

#### 3.1.3 完整Jacobian

```python
J_depth = [J_height_rot, J_height_tran]  # [num_init_pose, 6]
```

形状为 `[num_init_pose, 6]`，其中6表示6个pose参数（3个旋转 + 3个平移）。

## 4. Cost、Gradient和Hessian计算

### 4.1 Cost计算

**代码位置**: `costs.py` 第746-747行

```python
cost_depth = res_depth ** 2  # [num_init_pose]
cost_depth, w_loss_depth, _ = self.loss_fn1(cost_depth)
```

- 初始cost：`cost_depth = res_depth²`
- 经过鲁棒损失函数处理（如Huber损失），得到加权cost `w_loss_depth`

### 4.2 Gradient计算

**代码位置**: `costs.py` 第749行

```python
grad_depth = w_loss_depth.unsqueeze(-1) * res_depth.unsqueeze(-1) * J_depth
```

**数学公式**:
```
grad_depth = w_loss_depth * res_depth * J_depth^T
```

形状：`[num_init_pose, 6]`

### 4.3 Hessian计算

**代码位置**: `costs.py` 第751-755行

```python
Hess_depth = (J_depth.unsqueeze(-1) @ J_depth.unsqueeze(-2))  # [num_init_pose, 6, 6]
Hess_depth = w_loss_depth.unsqueeze(-1).unsqueeze(-1) * Hess_depth
```

**数学公式**:
```
Hess_depth = w_loss_depth * J_depth^T @ J_depth
```

形状：`[num_init_pose, 6, 6]`

## 5. 与投影约束合并

### 5.1 投影约束回顾

投影约束基于特征匹配：
- **残差**: `res = fp_q - fp_r`（查询帧特征 - 参考帧特征）
- **Cost**: `cost_proj = Σ(res²)` 经过鲁棒损失函数处理
- **Gradient**: `grad_proj = Σ(weights * J^T @ res)`
- **Hessian**: `Hess_proj = Σ(weights * J^T @ J)`

其中 `J` 是特征残差对pose的Jacobian链：
```
J = J_f @ J_p2d_p3d @ J_p3d_pose
```

### 5.2 约束合并

**代码位置**: `costs.py` 第770-787行

```python
if grad_depth is not None and Hess_depth is not None and cost_depth is not None:
    # 合并cost（需要广播到每个点）
    cost_depth_expanded = cost_depth.unsqueeze(-1).expand(B, num_poses, N)
    cost = cost + depth_weight * cost_depth_expanded
    
    # 合并gradient
    grad = grad + depth_weight * grad_depth
    
    # 合并Hessian
    Hess = Hess + depth_weight * Hess_depth
```

**形状说明**:
- `cost`: `[B, num_init_pose, N]` - 每个pose的每个点的cost
- `cost_depth`: `[1, num_init_pose]` - 每个pose的深度cost
- `cost_depth_expanded`: `[B, num_init_pose, N]` - 广播到每个点
- `grad`: `[B, num_init_pose, 6]` - 每个pose的gradient
- `grad_depth`: `[1, num_init_pose, 6]` - 每个pose的深度gradient
- `Hess`: `[B, num_init_pose, 6, 6]` - 每个pose的Hessian
- `Hess_depth`: `[1, num_init_pose, 6, 6]` - 每个pose的深度Hessian

**深度约束权重**: `depth_weight = 0.1`（可调整）

## 6. 关键代码位置

### 6.1 主要函数

- **`residual_jacobian_batch_quat`**: `costs.py` 第260行
  - 主函数，计算投影约束和深度约束

### 6.2 深度约束相关代码段

- **Step 6.1-6.2**: 反投影和坐标变换 (`costs.py` 第586-621行)
- **Step 6.3**: 坐标转换和DSM读取 (`costs.py` 第623-720行)
- **Step 6.4**: 高度误差计算 (`costs.py` 第722-727行)
- **Step 6.5**: Jacobian计算 (`costs.py` 第729-743行)
- **Step 6.6**: Cost/Gradient/Hessian计算 (`costs.py` 第745-760行)
- **Step 6**: 约束合并 (`costs.py` 第770-787行)

### 6.3 辅助函数

- **`get_world_point`**: `costs.py` 第161行
  - 根据pose、K和depth计算3D点的WGS84坐标（用于验证）

## 7. 坐标系说明

### 7.1 坐标系转换链

```
相机坐标系 (Camera)
    ↓ R_c2w, t_c2w
ENU局部坐标系 (East-North-Up, 相对于参考原点)
    ↓ rot_enu_in_ecef
ECEF坐标系 (Earth-Centered Earth-Fixed)
    ↓ transformer_ecef_to_wgs84
WGS84坐标系 (经纬度)
    ↓ transformer_wgs84_to_4547
EPSG:4547坐标系 (DSM地图投影坐标系)
```

### 7.2 各坐标系说明

- **相机坐标系**: 原点在相机光心，Z轴沿光轴方向
- **ENU局部坐标系**: 以参考原点为原点，东-北-天方向为坐标轴
- **ECEF坐标系**: 地心地固坐标系，原点在地球质心
- **WGS84坐标系**: 经纬度坐标系，用于GPS定位
- **EPSG:4547坐标系**: 高斯-克吕格投影坐标系，用于DSM地图

## 8. 性能优化

### 8.1 批量处理

- 所有pose的坐标转换批量进行
- DSM高度读取使用批量双线性插值
- Jacobian计算向量化

### 8.2 缓存机制

- 坐标转换器缓存：`_transformer_wgs84_to_4547_depth`, `_transformer_ecef_to_wgs84_depth`
- DSM插值器缓存：`_dsm_interpolator_depth`
- DSM边界和无效值缓存：`_dsm_bounds_depth`, `_dsm_nodata_depth`

### 8.3 双线性插值

使用 `scipy.interpolate.RegularGridInterpolator` 进行批量双线性插值，比逐个像素查询快得多。

## 9. 参数说明

### 9.1 输入参数

- `gt_depth`: 深度值（米），图像中心点的深度
- `dsm_path`: DSM地图文件路径（.tif格式）
- `render_T_ecef`: 参考原点的ECEF坐标（用于构建ENU坐标系）

### 9.2 可调参数

- `depth_weight`: 深度约束权重，默认0.1
  - 增大权重：深度约束影响更大，但可能影响投影约束
  - 减小权重：深度约束影响较小，主要依赖投影约束

### 9.3 输出

- `cost`: 合并后的cost `[B, num_init_pose, N]`
- `grad`: 合并后的gradient `[B, num_init_pose, 6]`
- `Hess`: 合并后的Hessian `[B, num_init_pose, 6, 6]`

## 10. 使用示例

深度约束会自动在优化过程中生效，只要提供以下参数：

```python
ret = self.cost_fn.residual_jacobian_batch_quat(
    pose_data_q, f_r, pose_data_r, cam_data_r, f_q, cam_data_q, p3D,
    c_ref, c_query, p2d_r, visible_r,
    gt_depth=depth_value,           # 深度值
    dsm_path=dsm_file_path,         # DSM文件路径
    render_T_ecef=reference_ecef    # 参考原点ECEF坐标
)
```

## 11. 注意事项

1. **坐标系一致性**: 确保所有坐标转换正确，特别是ENU到ECEF的转换
2. **DSM有效性**: 检查DSM高度值的有效性（非NaN、非NoData、在边界内）
3. **深度值单位**: 确保深度值单位是米，与DSM高度单位一致
4. **性能**: DSM插值器会在第一次使用时初始化，后续调用会复用
5. **错误处理**: 如果坐标转换或DSM读取失败，会回退到使用p3d_world的Z坐标

## 12. 数学公式总结

### 12.1 完整Jacobian链

```
J_depth = [J_height_rot, J_height_tran]

其中：
J_height_rot = R_c2w[2, :] @ skew_symmetric(p3d_cam)
J_height_tran = -R_c2w[2, :]
```

### 12.2 优化目标

```
minimize: cost_total = cost_proj + depth_weight * cost_depth

其中：
cost_proj = Σ(weights * loss_fn(res²))
cost_depth = loss_fn(res_depth²)
res_depth = height_3d_wgs84 - height_dsm
```

### 12.3 梯度下降更新

```
pose_new = pose_old - learning_rate * (Hess_total)^(-1) @ grad_total

其中：
grad_total = grad_proj + depth_weight * grad_depth
Hess_total = Hess_proj + depth_weight * Hess_depth
```

## 13. 参考文献

- 坐标系转换：pyproj库文档
- 双线性插值：scipy.interpolate.RegularGridInterpolator
- DSM数据格式：rasterio库文档
- 优化算法：Gauss-Newton / Levenberg-Marquardt

