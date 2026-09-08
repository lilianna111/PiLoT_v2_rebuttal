# Bug修复: crop部分4326迁移错误

## 问题描述
错误信息:
```
OpenCV(4.10.0) error: (-215:Assertion failed) _src.total() > 0 in function 'warpPerspective'
```

**原因**: `ray_casting.py` 输出ECEF坐标，但 `proj2map.py` 的 `geo_coords_to_dsm_index()` 期望投影坐标（4547）。

---

## 修复内容

### 1. proj2map.py 第58-77行
**添加新函数**: `ecef_to_dsm_index_4326()`
```python
def ecef_to_dsm_index_4326(ecef_point, transform):
    """
    将ECEF坐标转换为4326 DSM的像素索引
    """
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, alt = transformer.transform(ecef_point[0], ecef_point[1], ecef_point[2])
    col, row = ~transform * (lon, lat)
    return int(row), int(col)
```

### 2. proj2map.py 第169-172行
**修改**: 使用新函数转换ECEF→DSM索引
```python
# 旧代码:
dsm_indices = [
    geo_coords_to_dsm_index(float(xyz[0]), float(xyz[1]), dsm_transform)
    for xyz in world_points
]

# 新代码:
dsm_indices = [
    ecef_to_dsm_index_4326(xyz, dsm_transform)
    for xyz in world_points
]
```

### 3. proj2map.py 第229-248行
**修改**: 将WGS84(lon,lat,alt)转换为ECEF点云
```python
# 将 WGS84(lon, lat, alt) 转换为 ECEF(X, Y, Z)
import pyproj
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

# 展平数组进行批量转换
lon_flat = lon_warp.flatten()
lat_flat = lat_warp.flatten()
alt_flat = dsm_warp.flatten()

x_ecef, y_ecef, z_ecef = transformer.transform(lon_flat, lat_flat, alt_flat)

# 重塑回原始形状
x_ecef = np.array(x_ecef).reshape(out_h, out_w)
y_ecef = np.array(y_ecef).reshape(out_h, out_w)
z_ecef = np.array(z_ecef).reshape(out_h, out_w)

# 组成 ECEF xyz 点云
xyz = np.stack([x_ecef, y_ecef, z_ecef], axis=-1)
```

---

## 数据流修正

**修正前（错误）**:
```
ray_casting → ECEF点云
  ↓
crop → 期望4547坐标 → ❌ 索引错误
```

**修正后（正确）**:
```
ray_casting → ECEF点云
  ↓
crop → ECEF转WGS84 → 4326 DSM索引 → WGS84裁剪 → 转ECEF输出
  ↓
定位 → 接收ECEF点云 ✓
```

---

## 修复验证

需要重新运行crop流程确认:
1. `warpPerspective` 不再报错
2. 裁剪图像和点云正常生成
3. 点云坐标为ECEF格式

---

## 状态
- [x] 添加 `ecef_to_dsm_index_4326()` 函数
- [x] 修改DSM索引计算
- [x] 修改点云输出为ECEF
- [ ] 实际运行验证
