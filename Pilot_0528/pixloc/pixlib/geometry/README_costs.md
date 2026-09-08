# costs.py 说明文档

本模块实现**直接法绝对位姿优化**所需的代价、残差与雅可比计算，面向“参考图 + 多候选查询位姿”的批量优化（如 PixLoc 式流程），并融合**重投影约束**、**DSM 深度约束**与**姿态角先验（roll/pitch）**。

---

## 一、整体做什么

- **输入**：参考帧特征与 2D 点、查询帧特征与相机内参、多个候选位姿 `pose_data_q`、参考帧下的 3D 点 `p3D`，以及可选的 DSM 路径、GT 深度、GT roll/pitch、坐标系参数等。  
- **输出**：对每个候选位姿的 **负梯度 `-grad`**、**Hessian 近似 `Hess`**、鲁棒权重 `w_loss`、有效掩码 `valid_query`、投影点 `p2d_q`、以及投影代价 `cost`，供上层优化器（如 `learned_optimizer.py`）做高斯-牛顿/拟牛顿步。

核心入口是类 **`DirectAbsoluteCost2`** 的 **`residual_jacobian_batch_quat`**。

---

## 二、符号与坐标系

- **位姿**：`pose_data_q` 为 `[B, num_init_pose, 12]`，前 9 维为 `R_w2c`（行优先），后 3 维为 `t_w2c`；即世界到相机的旋转和平移。  
- **坐标系**：  
  - 局部世界/相机：优化在“局部”进行，再通过 `origin`、`dd` 或 `render_T_ecef` 与 **ECEF** 关联。  
  - **ECEF**：地心直角坐标。  
  - **WGS84**：经度、纬度、高程 (lon, lat, alt) + 姿态角 (roll, pitch, yaw)，用于和 DSM、先验角对接。  
  - **ENU**：由当前经纬度定义的东-北-天，用于把相机旋转表达为 roll/pitch/yaw（如 `xyz` 欧拉角）。

---

## 三、主要函数与类

### 3.1 工具函数（JIT）

| 函数 | 作用 |
|------|------|
| `J_project(p3d)` | 透视投影的雅可比：\( \partial (x/z,\, y/z) / \partial (x,y,z) \)，形状 `(..., 2, 3)`。 |
| `transform_p3d(pose_data, p3d)` | 用 pose 的 R,t 将 3D 点从世界变到相机：`p3d @ R.T + t`。 |
| `project_p3d(camera_data, p3d)` | 将相机系 3D 点投影到 2D（含 fx,fy,cx,cy 与图像边界），返回 `p2d` 与 `valid`。 |
| `skew_symmetric(v)` | 由 3 维向量构造反对称矩阵 `[v]×`，用于旋转李代数相关雅可比。 |

### 3.2 类 `DirectAbsoluteCost2`

#### 初始化与缓存

- `iter`：当前迭代次数（用于控制调试打印）。  
- `_ecef_to_wgs84` / `_wgs84_to_ecef`：pyproj 转换器，懒加载。  
- `_dsm_cache`：按 DSM 文件路径缓存栅格与仿射，支持 CPU（`map_coordinates`）与 GPU（`grid_sample`）。  
- `use_gpu_dsm`：是否用 GPU 查 DSM。

#### 坐标与中心射线

- **`_poses_to_wgs84(pose_data_q, render_T_ecef)`**  
  将候选位姿的“相机中心”从局部/ECEF 转到 WGS84，得到列表 `[lon, lat, alt, roll, pitch, yaw]`（roll/pitch/yaw 由 ENU 下 xyz 欧拉角得到）。

- **`_pose_translations_ecef(pose_data_q, origin, dd)`**  
  由 `R_w2c`、`t_w2c` 得到相机中心在局部下的位置，再加 `origin` 和 `dd`，得到 ECEF 下的相机中心。

- **`_center_points_from_wgs84_poses(wgs84_poses, depth, K, mul)`**  
  对每组 (lon, lat, alt, roll, pitch, yaw)，用内参 K 和给定深度（及可选 `mul`）计算图像中心射线在 WGS84 下的 3D 点（先 ECEF 再转 WGS84）。

- **`_center_points_from_pose_ecef(pose_data_q, depth, K, origin, dd, mul)`**  
  同上，但直接从 ECEF 位姿和深度算中心射线与地面交点（不经过 WGS84 欧拉角转换），数值更一致。

#### DSM

- **`_query_dsm_heights(wgs84_points, dsm_path, proj_crs, use_gpu)`**  
  用经纬度在 DSM 上双线性插值得到高程；支持投影系（默认 EPSG:4547）与缓存；GPU 分支用 `grid_sample`。

#### 特征与插值

- **`bilinear_grid_sample(im, grid, align_corners)`**  
  可微双线性采样，用于特征图在非整数像素的取值（与梯度计算一致）。

- **`interpolate_feature_map(feature, p2d, return_gradients)`**  
  在 `p2d` 处对特征图采样，返回特征向量、有效掩码；若 `return_gradients=True`，用中心差分估计 \(\partial f/\partial (u,v)\)，供链式求雅可比。

#### 鲁棒损失

- **`loss_fn1(cost, alpha, truncate, eps, derivatives)`**  
  对标量/逐元素 cost 做鲁棒化：  
  - 内部变量 \(x = \text{cost}/\text{truncate}^2\)。  
  - \(\alpha=0\)：近似 log 型；\(\alpha=2\)：二次；否则为广义 Charbonnier 型。  
  - 返回：加权后的 loss、一阶权重 `loss_d1`、二阶占位 `loss_d2`（当前未用），用于对残差和雅可比加权，抑制外点。

---

## 四、核心流程：`residual_jacobian_batch_quat`

该函数实现“多候选位姿 × 多 3D 点”的批量残差与雅可比，并融合三类约束。

### 4.1 输入概览

- `pose_data_q`：`[1, num_init_pose, 12]` 候选位姿。  
- `f_r`, `f_q`：参考/查询特征图。  
- `cam_data_r`, `cam_data_q`：参考/查询相机（含尺寸与内参）。  
- `p3D`：参考帧下的 3D 点（世界/局部）。  
- `c_ref`, c_query`：参考/查询上的置信或权重图（用于加权残差）。  
- `p2d_r`, visible_r`：参考帧 2D 观测与可见性。  
- 可选：`gt_depth`, `gt_roll`, `gt_pitch`, `origin`, `dd`, `mul`, `dsm_path`, `render_T_ecef`。

### 4.2 步骤 1–2：参考帧特征与掩码

- 在参考帧 2D 点 `p2d_r` 处插值得到 `fp_r`，并与 `visible_r` 得到有效掩码 `valid_ref_mask`。  
- 对 `fp_r`、`p3D` 按掩码置零并复制到 `num_init_pose` 份，再对特征做 L2 归一化。

### 4.3 步骤 3：查询帧投影与特征

- 用 `transform_p3d` + `project_p3d` 将每个候选位姿下的 `p3D` 投影到查询像平面，得到 `p2d_q` 与 `visible_q`。  
- 在 `p2d_q` 处插值查询帧特征 `fp_q` 及（若需要）特征对像素的雅可比 `J_f`。

### 4.4 步骤 4：重投影残差与雅可比

- **残差**：`res = fp_q - fp_r`（同维特征差）。  
- **雅可比链**：  
  - `J_p3d_pose`：\( \partial(\text{p3d\_q})/\partial\text{pose} \)（旋转部分用 `-R @ [p3D]×`，平移部分为 R）。  
  - `J_p2d_p3d`：由 `J_project` 与相机内参得到 \( \partial\text{p2d}/\partial\text{p3d\_q} \)。  
  - `J = J_f @ J_p2d_p3d @ J_p3d_pose`，得到每个点对 6-DoF 位姿的雅可比。

### 4.5 步骤 5–6：深度约束与角度约束（可选）

当提供 `render_T_ecef`/origin/dd`、`gt_depth`、`dsm_path` 时：

- **深度约束**  
  - 对每个候选位姿，用 `_center_points_from_pose_ecef` / `_center_points_from_wgs84_poses` 和 `gt_depth`（及 `mul`）得到“中心射线与地面对应的 3D 点”的 WGS84 高程。  
  - 在对应 (lon, lat) 上用 `_query_dsm_heights` 取 DSM 高程。  
  - **残差**：`depth_res = 估计高程 - DSM 高程`（并除以 `depth_sigma` 做尺度归一化）。  
  - **雅可比**（解析）：\( \mathrm{d}(\text{中心点高程})/\mathrm{d}\text{pose} \) 写为 `[(u×v)ᵀ, -uᵀ]`，其中 `u` 为当地“天”方向单位向量，`v` 为 ECEF 下中心射线方向。

- **角度约束**  
  - 由当前 `R_w2c` 与相机中心经纬度得到 ENU 下的 `R_pose_in_enu`，再取 xyz 欧拉角得到 `roll_base`、`pitch_base`。  
  - **残差**：`angle_res = [roll_base - gt_roll, pitch_base - gt_pitch]`（弧度）。  
  - **雅可比**：通过 `d(roll)/d(R_pose_in_enu)`、`d(pitch)/d(R_pose_in_enu)` 以及 `d(R_pose_in_enu)/d(ω_c)`（相机系旋转向量）的链式法则解析得到，再补零扩展到 6-DoF 的 `J_angle_tensor`。

### 4.6 步骤 7：统一鲁棒加权与梯度/Hessian

- 投影 cost：`cost_proj = sum(res²)`（按特征维度）。  
- 若有深度/角度：  
  - `cost_depth = depth_res²`，`cost_angle = sum(angle_res²)`；  
  - 将 `cost_proj`、`cost_depth`、`cost_angle` 沿最后一维拼成 `cost_all`，送入 `loss_fn1` 得到加权 loss 与权重 `w_loss_proj`、`w_loss_depth`、`w_loss_angle`。  
- 用 `c_ref`、c_query` 在参考/查询 2D 位置采样得到不确定性权重，与 `w_loss_proj`、`valid_query` 组合成 `weights_proj`。  
- **梯度**：  
  - 投影：`grad_proj = sum(weights_proj * J^T * res)`（对点维度求和）。  
  - 深度：`grad_depth`、角度：`grad_angle` 由各自残差与雅可比及权重得到。  
  - 对深度/角度梯度做**自适应缩放**（使相对投影梯度的范数约为一固定比例，如 20%），再相加得到总梯度 `grad`。  
- **Hessian**：  
  - 投影：`Hess_proj = sum(weights_proj * J^T J)`；深度与角度同理，缩放后相加得 `Hess`。

### 4.7 返回值

- `-grad`：负梯度（便于优化器做梯度上升式步进或与现有接口一致）。  
- `Hess`：近似 Hessian。  
- `w_loss`：投影部分的鲁棒权重。  
- `valid_query`：查询帧有效观测掩码。  
- `p2d_q`：投影 2D 点。  
- `cost`：投影部分代价（未含深度/角度的加权 loss）。

---

## 五、与上层优化器的配合

- **learned_optimizer**（或其它 base）在每步将当前候选位姿 `T_flat` 与固定参考数据、查询特征等传入 `residual_jacobian_batch_quat`。  
- 用返回的 `-grad`、`Hess` 做高斯-牛顿步（如 `Hess @ dx = -grad`）或加权最小二乘步，更新位姿。  
- `valid_query`、`p2d_q`、`cost` 可用于可视化、早停或置信度估计。

---

## 六、依赖与配置要点

- **坐标系**：若使用 DSM 与角度先验，须保证 `origin`/`dd` 或 render_T_ecef` 与场景一致，且 DSM 的 CRS（默认 EPSG:4547）与 `_query_dsm_heights` 的 `proj_crs` 一致。  
- **深度单位**：`gt_depth` 与 `t_w2c` 一致；`mul` 用于在“米↔其它单位”之间转换时乘到深度上。  
- **迭代计数**：通过设置 `cost_fn.iter` 可控制前若干迭代的调试打印（如深度/角度权重的诊断信息）。

以上即为 `costs.py` 的“怎么做”说明，便于维护和扩展（例如新增约束或其它鲁棒核）。
