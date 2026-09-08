# `main.py` 完整流程与数据处理

> 对应入口：`/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528/main.py`  
> 对应配置：`configs/feicuiwan_m4t.yaml`  
> 本文只描述当前实际执行路径，便于后续分阶段统计和降低耗时。

## 1. 总体流程

```text
run_feicuiwan.sh
  -> main.py 读取 YAML
  -> DualProcessTask 初始化数据、相机、GT 和队列
  -> 启动裁剪进程 rendering_worker
  -> 启动定位进程 localization_worker

裁剪进程：上一时刻位姿
  -> DSM/DOM 地图裁剪
  -> RDOM 图像 + 每像素投影坐标点云
  -> 随机采样 500 个 2D-3D 对应点
  -> task_q

定位进程：query 图像 + task_q 数据
  -> 投影坐标转 ECEF，并生成多组初始位姿
  -> query/RDOM 多尺度特征提取
  -> 多尺度 LM 位姿优化
  -> 选择最优位姿并转回 WGS84/欧拉角
  -> 新位姿写入 pose_q，驱动后续地图裁剪
  -> 全部完成后保存结果并评估
```

## 2. 启动与初始化

1. `run_feicuiwan.sh` 按实验名调用 `main.py`，实际使用 `configs/feicuiwan_m4t.yaml`。
2. `main.py` 读取配置，创建 `DualProcessTask`。
3. 根据实验名读取：
   - query 图像：`<dataset_path>/images/<name>/`
   - GT pose：`<dataset_path>/poses/<name>.txt`
   - 深度先验：`/media/amax/AE0E2AFD0E2ABE69/datasets/depth_1/<name>.txt`
   - roll/pitch 先验：`/media/amax/AE0E2AFD0E2ABE69/datasets/angle/<name>.txt`
4. Query 图像从 `3840x2160` 缩放到约 `512x288`，并一次性读入内存。
5. 相机内参同步缩放到 `512x288`。
6. GT 第一帧提供初始位姿和 ECEF 原点；完整 GT 文件被解析为逐帧字典。
7. 创建两个队列：
   - `pose_q`：定位进程向裁剪进程发送位姿。
   - `task_q`：裁剪进程向定位进程发送 RDOM、3D 点和辅助数据。
8. `pose_q` 初始放入两份相同的第一帧 GT 位姿，用于启动流水线。

## 3. 裁剪进程

### 3.1 地图只加载一次

当前地图路径在 `rendering_worker()` 内固定为 0.3 m DSM/DOM：

- DOM：`DSM0.3_ortho_merge.tif`
- DSM：`DSM0.3_DSM_merge.tif`
- DSM 缓存：`DSM0.3.npy`

`read_DSM_config()` 一次性得到 DOM、DSM、地理变换、DSM 有效高程中位数等数据，后续所有帧复用。

### 3.2 位姿转为裁剪几何

输入位姿格式为：

```text
[longitude, latitude, altitude, roll, pitch, yaw]
```

处理步骤：

1. WGS84 经纬度转为 `EPSG:4547` 投影坐标。
2. 欧拉角转为裁剪所需的相机外参。
3. 使用固定的裁剪相机内参，得到图像四角射线。

### 3.3 DSM 射线与地图裁剪

对四个图像角点分别执行：

1. 从相机中心沿像素射线采样 4000 个点。
2. 比较射线高度与 DSM 高程。
3. 取高度差最小的位置作为射线和地面的交点。
4. 四个交点形成 DSM/DOM 上的四边形区域。
5. 先取局部包围框，再通过透视变换拉成规则矩形。

裁剪输出：

- `dom_crop`：RDOM 彩色参考图，`H x W x 3`。
- `point_cloud_crop`：与 RDOM 像素一一对应的 `EPSG:4547` 三维坐标，`H x W x 3`。

### 3.4 尺寸对齐与点采样

1. RDOM 和点云统一缩放到 `512x288`。
2. 有效点条件为坐标有限且 `Z > 0`。
3. 必要时补齐到 16 的倍数；当前 `512x288` 本身已经满足。
4. 从有效像素中随机选择最多 500 个点。
5. 同步保存：
   - `points_sampled`：500 个投影坐标三维点。
   - `p2d_r`：这些点在 RDOM 上的二维像素位置。
6. 数据通过 `task_q` 发给定位进程。

## 4. 定位进程

### 4.1 模型初始化

`RenderLocalizer` 根据配置和 checkpoint 构建：

```text
TwoViewRefiner
  -> Unet_fusion 特征网络
  -> 3 个 LearnedOptimizer
  -> DirectAbsoluteCost2
```

配置使用三层特征尺度 `[1, 2, 4]`，每层优化器独立。

### 4.2 3D 点与候选位姿处理

`back_project()` 调用 `get_3D_samples_v3()`：

1. 将 500 个点从 `EPSG:4547` 转为 ECEF。
2. 乘以 `mul=0.001`，减小数值尺度。
3. 先减全局 ECEF 原点，再减局部点云中心 `dd`，改善优化数值稳定性。
4. 根据上一定位结果生成候选 query 位姿：
   - yaw：12 个固定偏移，范围为 `+/-1, +/-3, ..., +/-11` 度。
   - pitch、roll：当前不扩展。
   - ECEF X/Y：各 11 个位置，范围 `-10 m` 到 `+10 m`，步长 `2 m`。
   - ECEF Z：当前不扩展。
5. 笛卡尔积得到 `12 x 11 x 11 = 1452` 个初始位姿。
6. 每个候选再附加最多 1 m 的随机平移扰动；第一个候选不加该随机扰动。

因此，配置中的 `num_init_pose: 64` 并不决定当前实际候选数量。

### 4.3 Query 与 RDOM 预处理

1. Query 和 RDOM 都从 `512x288` 补零到 `512x512`。
2. 分别输入同一个 `Unet_fusion` 提取三层特征。
3. 网络结构为：
   - `mobileone_s0` 编码器；
   - U-Net 解码器；
   - 三层 32 维特征头；
   - 三层置信度头。
4. 优化前将置信度通道和描述子通道分开。
5. 当前前端最多只采样 500 点，而置信度筛选也是 `top_k=500`，所以通常不会继续减少点数。

## 5. 多尺度 LM 优化

优化顺序为从粗到细：`1/4 -> 1/2 -> 1`。

### 5.1 每层输入

- 500 个中心化 ECEF 三维点。
- RDOM 点对应的二维坐标和参考特征。
- Query 多尺度特征图。
- 当前批量候选位姿。
- Query/RDOM 相机内参。
- 深度、roll、pitch 和 DSM 路径。

### 5.2 每次 LM 迭代

1. 将三维点投影到每个候选位姿的 Query 图像。
2. 在 Query 特征图上双线性采样特征及特征梯度。
3. 计算 Query 特征与 RDOM 特征的残差。
4. 通过投影链式法则得到对 6DoF 位姿的 Jacobian。
5. 使用 Query/RDOM 置信度和鲁棒核加权。
6. 融合深度先验：中心射线点高度与 DSM 高度的差。
7. 融合角度先验：候选 roll/pitch 与外部 roll/pitch 的差。
8. 累积梯度和 Hessian，加入学习到的 LM damping。
9. 解线性方程，限制单步旋转不超过 2 度、平移不超过 2 m。
10. 更新全部候选位姿。

三层实际迭代次数由特征分辨率硬编码为约 `2、3、4`，不是直接使用 YAML 中的 `num_iters: 3`。

### 5.3 候选缩减与选择

1. 最粗层优化全部 1452 个候选。
2. 最粗层结束后，仅保留损失最小的 128 个候选。
3. 中层和细层继续优化这 128 个候选。
4. 最后剔除优化失败，以及相对参考位姿平移或旋转变化过大的候选。
5. 在有效候选中选择总损失最小者。
6. 将中心化坐标还原到 ECEF，再转为 WGS84 和 `[pitch, roll, yaw]`。

## 6. 位姿反馈与输出

1. 每帧定位完成后，下一次地图裁剪始终使用当前预测位姿。
2. 代码不再计算 GT 阈值，也不会使用 GT 位姿重置后续帧。
3. 新位姿通过 `pose_q` 返回裁剪进程，形成闭环。

当前 `pose_q` 预先放入两份初始位姿，因此流水线中裁剪位姿相对定位结果存在约两帧的反馈延迟：前两次裁剪均使用初始 GT，后续裁剪依次使用较早的预测结果。

全部帧结束后才统一写出：

- 预测 pose：`/media/amax/PS2000/ral/crop_0629/<name>.txt`
- 裁剪图像：`/media/amax/PS2000/ral/crop_0629/<name>/`
随后 `evaluate()` 计算 ECEF 平移误差、整体旋转误差和 yaw 误差，并输出中位数、标准差和阈值召回率。

## 7. 后续耗时分析的阶段边界

建议按下列边界分别统计，不要只看单帧总时间：

| 阶段 | 当前计时字段 | 主要操作 |
|---|---|---|
| 地图裁剪 | `crop_ms` | 四角射线、DOM/DSM 裁剪、透视变换、resize、采样 |
| 3D/候选生成 | `back_project_ms`、`bp_samples_ms` | CRS 转换、中心化、1452 个候选生成 |
| 特征提取 | `feature_ms` | Query 和 RDOM 各执行一次 Unet_fusion |
| LM 总耗时 | `optimizer_ms` | 三个尺度、全部候选的迭代优化 |
| LM 残差 | `res_detail` | 投影、特征采样、深度和角度先验、系统构建 |
| 线性求解 | `solve` | 6x6 LM 方程批量求解 |
| 后处理 | `post_ms` | 候选过滤、最优选择、ECEF/WGS84 还原 |

当前最值得优先拆分观察的是：最粗层 1452 候选的残差计算、深度 DSM 查询、Query/RDOM 双次特征提取，以及裁剪进程的 4000 点四射线采样。
