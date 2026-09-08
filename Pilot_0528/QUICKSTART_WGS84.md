# WGS84迁移 - 快速开始指南

## 📋 已完成的修改

### ✅ 1. costs.py - DSM查询优化
- 修改 `_query_dsm_heights` 默认参数：`proj_crs="EPSG:4326"`
- 删除 WGS84→EPSG:4547 的坐标转换
- 直接用经纬度计算DSM像素坐标

### ✅ 2. ray_casting.py - ECEF坐标系
- 删除 `_wgs84_to_proj` 投影转换器
- `_build_camera_to_world` 改用ECEF坐标系
- `caculate_predictXYZ_batch` 射线采样在ECEF，查表用WGS84

### ✅ 3. main.py - 数据路径
- DSM路径：`/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif`
- DOM路径：`/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dom.tif`

---

## 🚀 使用FPVLoc环境验证

### 方法1：快速验证（推荐）

```bash
# 1. 激活FPVLoc环境
conda activate FPVLoc

# 2. 进入项目目录
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528

# 3. 运行验证脚本
chmod +x verify_in_fpvloc.sh
bash verify_in_fpvloc.sh
```

### 方法2：手动验证

```bash
# 1. 激活环境
conda activate FPVLoc
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528

# 2. 检查DSM文件
ls -lh /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84*.tif

# 3. 运行Python验证
python verify_wgs84_migration.py

# 4. 运行完整定位测试
python main.py -c configs/feicuiwan_m4t.yaml --name test_wgs84
```

---

## 🔍 验证检查清单

### ✅ 文件检查
- [ ] wgs84dsm.tif 存在且可读
- [ ] wgs84dom.tif 存在且可读
- [ ] 文件坐标系是EPSG:4326（用gdalinfo检查）

### ✅ 代码验证
- [ ] costs.py 的 proj_crs="EPSG:4326"
- [ ] ray_casting.py 使用ECEF坐标系
- [ ] main.py 路径指向wgs84*.tif

### ✅ 功能测试
- [ ] 反投影精度：计算高度与DSM高度差异 < 1m
- [ ] Crop时间：≤ 60ms
- [ ] 定位精度：与EPSG:4547版本相当

---

## 📊 预期结果

### 验证脚本输出（成功）
```
🔍 验证WGS84迁移 - 反投影测试
================================================================================
📍 测试Pose: [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803]
📏 测试深度: 167.05m

🔄 步骤1：反投影计算世界坐标...
   计算结果: Lon=112.99905400, Lat=28.29049700, Alt=324.55m

🗺️  步骤2：查询WGS84 DSM高度...
   DSM高度: 324.50m

📊 步骤3：精度验证...
   计算高度: 324.55m
   DSM高度:  324.50m
   高度差异: 0.05m
   ✅ 精度验证通过！误差 0.05m < 1.0m

✅ WGS84迁移验证通过！
```

### 性能对比

| 指标 | EPSG:4547 | EPSG:4326 | 目标 |
|------|-----------|-----------|------|
| Crop时间 | 60ms | ? | ≤60ms |
| 坐标转换 | 3次 | 1次 | 减少 |
| 定位精度 | 基线 | ? | 持平 |

---

## ⚠️ 常见问题

### Q1: DSM文件不存在怎么办？
**A**: 确保已经准备了WGS84坐标系的DSM/DOM文件：
```bash
ls /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84*.tif
```

### Q2: 精度验证失败怎么办？
**A**: 检查：
1. DSM文件坐标系是否正确（用gdalinfo检查）
2. geotransform是否正确读取
3. ECEF转换函数是否正确

### Q3: Crop时间变慢怎么办？
**A**: 优化建议：
1. 向量化ECEF转换（当前是循环）
2. 预加载DSM到GPU
3. 缓存常用转换结果

### Q4: 与EPSG:4547结果不一致？
**A**: 可能原因：
1. DSM重采样方法不同（双线性 vs 最近邻）
2. 坐标转换精度差异（可接受范围：< 0.5m）
3. 检查两个DSM是否来自同一源数据

---

## 🔄 回滚方案

如果WGS84版本有问题，可以快速回滚：

```bash
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528

# 回滚costs.py
git checkout costs.py

# 回滚ray_casting.py
git checkout pixloc/crop/ray_casting.py

# 回滚main.py
git checkout main.py
```

或者手动修改：
1. `costs.py:379` - 改回 `proj_crs="EPSG:4547"`
2. `ray_casting.py:21` - 恢复 `_wgs84_to_proj` 转换器
3. `main.py:323-324` - 改回旧DSM路径

---

## 📞 技术支持

### 修改文件清单
```
modified:   costs.py
modified:   pixloc/crop/ray_casting.py
modified:   main.py
created:    WGS84_MIGRATION_SUMMARY.md
created:    verify_wgs84_migration.py
created:    verify_in_fpvloc.sh
created:    QUICKSTART.md
```

### 关键函数
- `costs.py:375` - `_query_dsm_heights()`
- `ray_casting.py:186` - `_build_camera_to_world()`
- `ray_casting.py:200` - `caculate_predictXYZ_batch()`

### 调试命令
```bash
# 查看DSM信息
gdalinfo /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif

# 测试ECEF转换
python -c "from transform import WGS84_to_ECEF, ECEF_to_WGS84; \
    print(ECEF_to_WGS84(WGS84_to_ECEF([112.999054, 28.290497, 157.50])))"

# 查看代码修改
git diff costs.py
git diff pixloc/crop/ray_casting.py
```

---

## ✅ 完成后的行动

验证通过后：
1. ✅ 运行完整数据集测试
2. ✅ 记录性能指标（时间、精度）
3. ✅ 对比EPSG:4547和EPSG:4326的结果
4. ✅ 如果性能/精度满足要求，可以删除旧DSM文件

---

**最后更新**: 2026-07-02  
**验证环境**: FPVLoc conda环境  
**状态**: ✅ 代码修改完成，待验证
