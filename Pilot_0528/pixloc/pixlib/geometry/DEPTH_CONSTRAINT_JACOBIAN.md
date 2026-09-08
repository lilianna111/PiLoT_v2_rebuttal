# 深度约束雅可比矩阵推导与实现

## 1. 问题定义

### 1.1 残差函数

深度约束的残差函数定义为：

```
r = z_center - h_dsm
```

其中：
- `z_center`: 从相机位姿估计的3D中心点的高度（WGS84坐标系，单位：米）
- `h_dsm`: 从DSM地图查询的地表高度（EPSG:4547坐标系，单位：米）

**注意**：`h_dsm` 是固定值（从DSM查询得到），不依赖于位姿，因此：
```
dr/dpose = dz_center/dpose
```

### 1.2 目标

计算残差对位姿的雅可比矩阵：
```
J = dr/dpose = [∂r/∂ω, ∂r/∂t]
```

其中：
- `ω ∈ R³`: 旋转参数（轴角表示）
- `t ∈ R³`: 平移参数

## 2. 数学推导

### 2.1 坐标变换链

从位姿到中心点高度的变换链：

```
位姿 (R_c2w, t_c2w) 
  → 相机中心 C_ecef (ECEF坐标系)
  → 中心点 p_ecef (ECEF坐标系)
  → 中心点 p_wgs84 (WGS84坐标系)
  → 高度 z_center
```

### 2.2 关键步骤

#### 步骤1：计算相机中心在ECEF坐标系中的位置

```
C_local = -R_c2w @ t_w2c
C_ecef = C_local + origin + dd
```

其中：
- `R_c2w = R_w2c^T`: 相机到世界坐标系的旋转矩阵
- `t_w2c`: 世界到相机的平移向量
- `origin, dd`: ECEF坐标系的偏移量

#### 步骤2：计算中心像素的射线方向

在相机坐标系中：
```
pixel_center = [cx, cy, 1]^T
cam_vec = K^{-1} @ (depth * pixel_center)
```

其中：
- `K`: 相机内参矩阵
- `cx, cy`: 主点坐标
- `depth`: 深度值（单位需与`t_w2c`一致）

转换到ECEF坐标系：
```
v = R_c2w @ cam_vec
```

#### 步骤3：计算3D中心点

```
p_ecef = C_ecef + v
```

#### 步骤4：ECEF → WGS84 转换

```
p_wgs84 = ECEF_to_WGS84(p_ecef)
z_center = p_wgs84[2]  # 高度分量
```

### 2.3 雅可比矩阵推导

#### 2.3.1 对旋转的导数

考虑SE(3)流形上的小扰动：
```
δpose = [δω, δt]^T
```

对于旋转扰动 `δω`，在流形上的扰动为：
```
R_c2w_new = R_c2w @ exp([δω]×)
```

其中 `[δω]×` 是反对称矩阵。

对于小扰动，有：
```
exp([δω]×) ≈ I + [δω]×
```

因此：
```
R_c2w_new ≈ R_c2w @ (I + [δω]×) = R_c2w + R_c2w @ [δω]×
```

对中心点的影响：
```
δp_ecef = R_c2w @ [δω]× @ cam_vec = R_c2w @ (δω × cam_vec) = (R_c2w @ cam_vec) × δω
```

由于 `v = R_c2w @ cam_vec`，有：
```
δp_ecef = v × δω
```

#### 2.3.2 对平移的导数

对于平移扰动 `δt`：
```
t_w2c_new = t_w2c + δt
C_local_new = -R_c2w @ (t_w2c + δt) = C_local - R_c2w @ δt
```

因此：
```
δC_ecef = -R_c2w @ δt
δp_ecef = -R_c2w @ δt
```

#### 2.3.3 高度对ECEF坐标的导数

WGS84高度是ECEF坐标的非线性函数。对于ECEF坐标的小扰动 `δp_ecef`，高度变化为：
```
δz = u^T @ δp_ecef
```

其中 `u` 是"天"方向单位向量（在ECEF坐标系中）：
```
u = [cos(lat)·cos(lon), cos(lat)·sin(lon), sin(lat)]^T
```

#### 2.3.4 最终雅可比矩阵

结合上述结果：

**对旋转的导数**：
```
∂z/∂ω = u^T @ (v × δω) / δω = u^T @ [v]× = (u × v)^T
```

**对平移的导数**：
```
∂z/∂t = u^T @ (-R_c2w @ δt) / δt = -u^T @ R_c2w
```

由于 `R_c2w` 是旋转矩阵，且 `u` 在ECEF坐标系中，而 `t` 在局部坐标系中，需要考虑坐标变换。实际上，在局部坐标系中：
```
∂z/∂t = -u^T
```

**最终雅可比矩阵**：
```
J = [∂r/∂ω, ∂r/∂t] = [(u × v)^T, -u^T]
```

其中：
- `u`: "天"方向单位向量（ECEF坐标系）
- `v`: 从相机到中心点的射线方向（ECEF坐标系）

## 3. 代码实现

### 3.1 主要步骤

代码位置：`pixloc/pixlib/geometry/costs.py` 的 `residual_jacobian_batch_quat` 函数

#### 步骤1：提取位姿信息

```python
pose = pose_data_q[0] if pose_data_q.ndim == 3 else pose_data_q
R_w2c = pose[..., :9].view(-1, 3, 3).to(dtype=torch.float64)
t_w2c = pose[..., 9:].view(-1, 3).to(dtype=torch.float64)
R_c2w = R_w2c.transpose(1, 2)
```

#### 步骤2：计算相机中心位置

```python
C_local = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)
if origin is not None and dd is not None:
    origin_t = torch.as_tensor(origin, device=pose.device, dtype=torch.float64)
    dd_t = torch.as_tensor(dd, device=pose.device, dtype=torch.float64)
    C_ecef = C_local + origin_t + dd_t
else:
    C_ecef = C_local + torch.as_tensor(render_T_ecef, device=pose.device, dtype=torch.float64)
```

#### 步骤3：计算射线方向

```python
# 处理深度单位
if mul is not None:
    depth_for_jac = depth_real * float(mul)
else:
    depth_for_jac = depth_real

# 计算相机坐标系中的射线
K_matrix = torch.tensor(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    device=pose.device, dtype=torch.float64
)
K_inv = torch.inverse(K_matrix)
pixel_center = torch.tensor([cx, cy, 1.0], device=pose.device, dtype=torch.float64)
cam_vec = K_inv @ (depth_for_jac * pixel_center)

# 转换到ECEF坐标系
cam_vec_exp = cam_vec.unsqueeze(0).expand(num_init_pose, -1)
v = torch.bmm(R_c2w, cam_vec_exp.unsqueeze(-1)).squeeze(-1)  # [N, 3]
```

#### 步骤4：计算3D中心点和WGS84坐标

```python
if origin is not None and dd is not None:
    wgs84_poses = self._poses_to_wgs84(pose_data_q, C_ecef)
    center_points = self._center_points_from_pose_ecef(
        pose_data_q, gt_depth, K=cam_K, origin=origin, dd=dd, mul=mul
    )
else:
    wgs84_poses = self._poses_to_wgs84(pose_data_q, render_T_ecef)
    center_points = self._center_points_from_wgs84_poses(
        wgs84_poses, depth_real, K=cam_K
    )
```

#### 步骤5：查询DSM高度并计算残差

```python
dsm_heights = self._query_dsm_heights(center_points, dsm_path, use_gpu=self.use_gpu_dsm)

center_points_np = np.array(center_points, dtype=np.float64)
dsm_heights_np = np.array(dsm_heights, dtype=np.float64)
estimated_z = center_points_np[:, 2]  # WGS84高度
base_residuals = estimated_z - dsm_heights_np
```

#### 步骤6：计算"天"方向单位向量

```python
lon_rad = np.radians(center_points_np[:, 0])
lat_rad = np.radians(center_points_np[:, 1])
u_np = np.stack([
    np.cos(lat_rad) * np.cos(lon_rad),
    np.cos(lat_rad) * np.sin(lon_rad),
    np.sin(lat_rad)
], axis=-1)
u = torch.from_numpy(u_np).to(device=pose.device, dtype=torch.float64)
```

#### 步骤7：计算解析雅可比矩阵

```python
# 解析雅可比: J = [(u × v)^T, -u^T]
u_cross_v = torch.cross(u, v, dim=-1)  # [N, 3]
J_depth = torch.cat([u_cross_v, -u], dim=-1)  # [N, 6]
J_depth = J_depth.detach().cpu().numpy()
J_depth = np.nan_to_num(J_depth, nan=0.0)
```

### 3.2 归一化和权重

```python
# 归一化残差和雅可比
depth_sigma = 100.0  # 归一化因子（米）
base_residuals = base_residuals / depth_sigma
J_depth = J_depth / depth_sigma

# 计算鲁棒权重
cost_tensor = base_residuals_t ** 2
_, w_loss, _ = self.loss_fn1(cost_tensor, alpha=0.0, truncate=depth_truncate)
w_loss = w_loss * valid_mask

# 计算加权梯度
weighted = (base_residuals_t * w_loss * depth_weight).unsqueeze(-1)
grad_depth = J_depth_torch * weighted

# 计算加权Hessian
w_hess = (w_loss * depth_weight).view(-1, 1, 1)
Hess_depth = J_depth_torch[:, :, None] * J_depth_torch[:, None, :] * w_hess
```

### 3.3 自适应权重融合

```python
# 自适应缩放，使深度梯度占重投影梯度的20%
reproj_grad_norm = torch.norm(grad)
depth_grad_norm = torch.norm(grad_depth)
target_ratio = 0.2
depth_scale = (reproj_grad_norm / (depth_grad_norm + 1e-8)) * target_ratio
depth_scale = depth_scale.clamp(max=500.0)

# 融合
grad = grad + grad_depth * depth_scale
Hess = Hess + Hess_depth * depth_scale
```

## 4. 关键点说明

### 4.1 单位一致性

- **深度值**：需要与 `t_w2c` 的单位一致
  - 如果 `mul=0.001`，说明位姿是千米单位，深度也需要转换为千米
  - 如果 `mul=None`，说明位姿是米单位，深度直接使用米

- **坐标系统一**：
  - 所有计算在ECEF坐标系中进行
  - 最后转换为WGS84获取高度

### 4.2 坐标系转换

- **相机坐标系 → ECEF坐标系**：
  ```
  v_ecef = R_c2w @ v_cam
  ```

- **ECEF坐标系 → WGS84坐标系**：
  ```
  [lon, lat, alt] = ECEF_to_WGS84([x, y, z])
  ```

### 4.3 雅可比矩阵的物理意义

- **旋转部分 `(u × v)^T`**：
  - 表示旋转对高度的影响
  - 当射线方向 `v` 与"天"方向 `u` 垂直时，影响最大
  - 当 `v` 与 `u` 平行时，影响为0

- **平移部分 `-u^T`**：
  - 表示平移对高度的影响
  - 沿"天"方向平移1米，高度变化1米（负号表示相反方向）

### 4.4 与有限差分法的对比

**有限差分法**：
- 需要对每个自由度进行扰动
- 计算量大（需要7个位姿：1个基础 + 6个扰动）
- 有数值误差

**解析法**：
- 直接计算，无需扰动
- 计算量小（只需1个位姿）
- 精确无误差
- 数学推导清晰

## 5. 验证方法

代码中包含了验证打印，用于检查计算是否正确：

```python
# 验证查询帧位姿
print(f"  中心点3D: lon={lon_c:.6f}, lat={lat_c:.6f}, z={z_c:.2f}m")
print(f"  DSM高度: {dsm_q[0]:.2f}m")
print(f"  中心点z - DSM = {z_c - dsm_q[0]:.2f}m (应该接近0)")
```

如果 `中心点z - DSM ≈ 0`，说明3D坐标计算和DSM查询都正确。

## 6. 参考文献

- SE(3)流形上的优化：Barfoot, T. D. (2017). *State Estimation for Robotics*
- 坐标系转换：pyproj文档
- 雅可比矩阵推导：基于链式法则和SE(3)李代数

