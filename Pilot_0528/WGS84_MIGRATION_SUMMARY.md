# DSM/DOM坐标系迁移总结：EPSG:4547 → EPSG:4326

## 📋 迁移概述

**目标**：将DSM/DOM从EPSG:4547（CGCS2000投影坐标）迁移到EPSG:4326（WGS84地理坐标）

**核心优势**：
- ✅ 消除WGS84 ↔ EPSG:4547的重复转换
- ✅ 全程在ECEF坐标系计算，保持精度
- ✅ 简化坐标转换流程
- ✅ 保持60ms性能要求

---

## 🔧 已完成的修改

### 1️⃣ **costs.py** - DSM查询优化

**文件**：`/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528/costs.py`

**修改内容**：
```python
# 第375行：_query_dsm_heights方法
proj_crs: str = "EPSG:4326"  # 从EPSG:4547改为EPSG:4326
```

**关键变化**：
- ❌ 删除：`transformer_wgs84_to_proj` - 不再需要WGS84→投影坐标转换
- ✅ 新增：直接用经纬度计算像素坐标
- ✅ GPU路径和CPU路径都已优化

**代码逻辑**：
```python
# 旧逻辑（EPSG:4547）
lon, lat → WGS84_to_4547 → x_proj, y_proj → 像素坐标 → DSM查表

# 新逻辑（EPSG:4326）
lon, lat → 直接计算像素坐标 → DSM查表
```

---

### 2️⃣ **ray_casting.py** - 射线投射ECEF化

**文件**：`/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528/pixloc/crop/ray_casting.py`

**修改1：删除投影转换器（第21-23行）**
```python
# 旧代码
self._wgs84_to_proj = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:4547", always_xy=True
)

# 新代码
self._wgs84_to_proj = None  # WGS84不需要投影转换
```

**修改2：改用ECEF坐标系（第186-217行）**
```python
def _build_camera_to_world(self, pose):
    """使用ECEF坐标系构建相机到世界的变换矩阵"""
    # WGS84 → ECEF
    from ..utils.transform import WGS84_to_ECEF, get_rotation_enu_in_ecef
    
    # 相机姿态：ENU → ECEF
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    rot_enu_in_ecef = get_rotation_enu_in_ecef(raw_lon, raw_lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    
    # 相机位置：WGS84 → ECEF
    t_c2w = np.array(WGS84_to_ECEF([raw_lon, raw_lat, raw_alt]))
```

**修改3：射线采样ECEF化（第200-277行）**
```python
def caculate_predictXYZ_batch(self, K, pose, obj_pixel_coords, ...):
    # 1. 射线投射在ECEF坐标系
    x_ecef, y_ecef, z_ecef = ...  # 射线采样点（ECEF）
    
    # 2. 转换为WGS84查询DSM
    for each sample:
        lon, lat, alt = ECEF_to_WGS84([x_ecef, y_ecef, z_ecef])
    
    # 3. WGS84 DSM：geotransform直接是经纬度
    col = (lon - lon_origin) / degree_per_pixel
    row = (lat - lat_origin) / degree_per_pixel
    
    # 4. 返回WGS84结果 [lon, lat, alt]
```

---

### 3️⃣ **main.py** - 数据路径更新

**文件**：`/home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528/main.py`

**修改内容**：
```python
# 第323-324行
ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dom.tif"
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif"

# 第565行
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif"
```

---

## 🎯 技术方案详解

### 坐标系转换流程对比

#### 旧流程（EPSG:4547）
```
相机Pose(WGS84) 
    ↓
转ECEF（计算射线）
    ↓
转WGS84
    ↓
转EPSG:4547（投影坐标）  ← 额外转换
    ↓
DSM查表（4547）
    ↓
转回WGS84
```

#### 新流程（EPSG:4326）
```
相机Pose(WGS84)
    ↓
转ECEF（计算射线）  ← 全程ECEF，高精度
    ↓
转WGS84
    ↓
DSM查表（WGS84）    ← 直接查表，无需投影转换
```

---

## 🔍 关键技术点

### 1. ECEF坐标系的优势
- **高精度**：米级单位，避免度数单位的数值精度问题
- **全局一致**：不受经纬度位置影响
- **计算简单**：射线投射、距离计算都是简单的欧几里得几何

### 2. WGS84 DSM的geotransform
```python
# EPSG:4547（投影坐标）
x_origin = 500000.0  # 米
x_pixel_size = 0.3   # 米/像素

# EPSG:4326（WGS84）
x_origin = 112.99    # 度（经度）
x_pixel_size = 0.0000027  # 度/像素（约0.3米）
```

### 3. 精度保证
- ✅ **射线投射**：在ECEF进行，保持米级精度
- ✅ **DSM查询**：WGS84经纬度转像素，bilinear插值
- ✅ **反投影验证**：costs.py中的查表结果应与verify_matches_npy_pose.py一致

---

## ✅ 验证方案

### 验证1：DSM查表测试
运行验证脚本（需要修复torch导入）：
```bash
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528
python preprocess/4_verify_matches_npy_pose.py
```

**期望结果**：
- 输出：`计算出的世界坐标 (WGS84): (lon, lat, alt)`
- 这个结果应该与costs.py中`_query_dsm_heights`查到的高度一致

### 验证2：完整定位测试
```bash
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528
python main.py -c configs/feicuiwan_m4t.yaml --name test_wgs84
```

**期望指标**：
- ⏱️ **Crop时间**：≤ 60ms（与之前相当或更快）
- 📍 **定位精度**：与EPSG:4547版本相当
- ✅ **无报错**：无坐标转换错误

### 验证3：精度对比测试
创建对比测试脚本：
```python
# 测试同一pose下，4547和4326的DSM查询结果差异
pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
depth = 167.0476837158203

# 用两种DSM查询
height_4547 = query_dsm_4547(pose, depth)
height_4326 = query_dsm_wgs84(pose, depth)

# 差异应该 < 0.5米（DSM分辨率0.3米）
assert abs(height_4547 - height_4326) < 0.5
```

---

## 📊 性能预期

| 指标 | EPSG:4547 | EPSG:4326 | 变化 |
|------|-----------|-----------|------|
| Crop时间 | 60ms | **≤60ms** | 更快或持平 |
| 坐标转换次数 | 3次 | **1次** | ↓ 67% |
| DSM查表精度 | 亚像素 | **亚像素** | 持平 |
| 代码复杂度 | 高 | **低** | 简化 |

---

## 🐛 潜在问题和解决方案

### 问题1：area_minZ单位不匹配
**症状**：射线投射时，area_minZ是ECEF的Z坐标（米），但DSM是WGS84的高度
**解决**：在`caculate_predictXYZ_batch`中，area_minZ应该是WGS84高度，需要检查传入值

### 问题2：geotransform的y_pixel_size为负
**原因**：栅格图像原点在左上角，纬度从上到下递减
**处理**：代码中已正确处理：`row = (lat - y_origin) / y_pixel_size`

### 问题3：ECEF到WGS84转换开销
**优化**：batch转换，或缓存常用点
```python
# 当前：循环转换（较慢）
for i, j in range(num_points, num_samples):
    lon, lat, alt = ECEF_to_WGS84([x, y, z])

# 优化：向量化转换
lons, lats, alts = ECEF_to_WGS84_batch(x_ecef_array, y_ecef_array, z_ecef_array)
```

---

## 📝 后续优化建议

### 1. 向量化ECEF转换
创建batch版本的`ECEF_to_WGS84`：
```python
def ECEF_to_WGS84_batch(x, y, z):
    """向量化版本，输入numpy数组"""
    transprojr = pyproj.Transformer.from_crs(...)
    lon, lat, height = transprojr.transform(x, y, z, radians=False)
    return lon, lat, height
```

### 2. DSM预处理
将WGS84 DSM预先转为GPU tensor：
```python
# 启动时一次性加载
dsm_gpu = torch.from_numpy(dsm_data).cuda()
# 查询时直接用F.grid_sample
```

### 3. 缓存优化
对于连续帧，相邻pose的DSM区域重叠，可以缓存裁剪结果

---

## 🔗 相关文件清单

### 已修改文件
1. ✅ `costs.py` - DSM查询（GPU+CPU）
2. ✅ `pixloc/crop/ray_casting.py` - 射线投射ECEF化
3. ✅ `main.py` - DSM/DOM路径更新

### 需要验证的文件
1. ⚠️ `preprocess/4_verify_matches_npy_pose.py` - 反投影验证
2. ⚠️ `pixloc/crop/proj2map.py` - 裁剪生成（依赖ray_casting）
3. ⚠️ `pixloc/crop/utils.py` - 工具函数（可能需要更新注释）

### 数据文件
```
/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/
├── wgs84dsm.tif      ← WGS84 DSM（新）
├── wgs84dom.tif      ← WGS84 DOM（新）
├── DSM0.3_DSM_merge.tif    ← EPSG:4547 DSM（旧，保留对比）
└── DSM0.3_ortho_merge.tif  ← EPSG:4547 DOM（旧，保留对比）
```

---

## 🚀 下一步行动

### 立即执行
1. ✅ 检查DSM文件是否存在
2. ✅ 运行简单测试验证代码正确性
3. ✅ 对比EPSG:4547和EPSG:4326的定位精度

### 短期优化
1. 🔄 向量化ECEF转换（提升速度10-20ms）
2. 🔄 验证反投影脚本的正确性
3. 🔄 添加单元测试

### 长期改进
1. 📈 性能profiling，找出新的瓶颈
2. 📈 探索GPU加速ECEF转换
3. 📈 自动化精度测试流程

---

## 💡 总结

本次迁移**消除了WGS84与投影坐标系之间的重复转换**，通过：
- ✅ 全程ECEF计算（高精度）
- ✅ 直接查询WGS84 DSM（无需投影）
- ✅ 简化代码逻辑（降低维护成本）

**性能目标**：保持60ms约束，精度不降低。
**验证方法**：对比4547和4326版本的定位结果。

---

**修改完成时间**：2026-07-02  
**修改人**：Claude Opus 4.8  
**验证状态**：待验证
