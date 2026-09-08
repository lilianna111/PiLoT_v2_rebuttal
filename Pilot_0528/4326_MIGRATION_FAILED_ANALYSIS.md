# 4326迁移失败分析与建议

## 当前问题

**错误**: `warpPerspective` 断言失败，源图像为空

**根本原因**: 
- 4个角点的射线投射全部返回相同的ECEF坐标
- 导致所有点映射到同一个DSM像素 `(1811, 3438)`
- 裁剪区域为 `(0, 0)` - 空数组

**调试输出**:
```
[DEBUG] world_points shape: (4, 3)
[DEBUG] world_points sample: [-2195404.43, 5173897.23, 3005272.82]
[DEBUG] dsm_indices: [(1811, 3438), (1811, 3438), (1811, 3438), (1811, 3438)]
[DEBUG] dsm_crop shape: (0, 0), non-zero: 0
```

## 问题分析

### 为什么4个角点返回相同的ECEF坐标？

在 `caculate_predictXYZ_batch()` 中：
1. 4个不同的像素坐标 → 4条不同的射线（ECEF空间）
2. 所有射线与 `area_minZ` 高度平面求交
3. **问题**: `area_minZ = 25m` 是全局最小高程，不是局部地面高程
4. 在ECEF空间中，高程变化对应的坐标变化很小
5. 4条射线与同一个高度平面的交点几乎重合

### 为什么4547坐标系没有这个问题？

- 4547是投影坐标系（平面坐标，单位：米）
- 射线在平面空间采样，横向距离大
- 即使与同一高度平面相交，4个角点的x, y坐标差异明显

### 为什么ECEF坐标系有问题？

- ECEF是球面坐标系（单位：米，但在地球尺度）
- 相机距离地面只有几十到几百米
- 在这么小的尺度下，ECEF坐标变化极小
- 4个角点的射线交点差异被数值精度淹没

## 解决方案

### 方案1: 回退到4547坐标系（推荐）

**优点**:
- 已验证可行
- 射线投射在平面空间，数值稳定
- 不需要修改crop逻辑

**缺点**:
- 仍需要4547→ECEF转换
- 无法使用4326 DSM

**实施**:
```bash
git checkout HEAD -- pixloc/crop/ray_casting.py
git checkout HEAD -- pixloc/crop/proj2map.py
git checkout HEAD -- main.py
```

### 方案2: 改进ECEF射线投射（复杂）

**思路**:
- 不使用全局 `area_minZ`
- 对每个像素独立查询DSM高程
- 在WGS84空间进行射线-DSM求交

**优点**:
- 可以使用4326 DSM
- 理论上更准确

**缺点**:
- 需要重写射线投射逻辑
- 性能可能下降（每个像素都要查DSM）
- 实现复杂度高

### 方案3: 混合方案

**思路**:
- crop部分保持4547坐标系
- 只在costs.py改用4326 DSM
- 在costs.py内部做4547→WGS84转换查4326 DSM

**优点**:
- crop部分稳定不变
- costs.py可以使用4326 DSM
- 修改范围小

**缺点**:
- 仍需要坐标转换
- 没有完全消除4547依赖

## 建议

**立即行动**: 回退到4547方案（方案1）

**原因**:
1. 当前ECEF方案存在根本性数值问题
2. 修复需要重写射线投射逻辑，风险高
3. 4547方案已验证可行，可以保证60ms性能

**长期优化**: 如果必须使用4326 DSM
1. 保持crop在4547坐标系
2. 只在costs.py改用4326（需要添加4547→WGS84转换）
3. 或者寻找4547版本的DSM

## 回退命令

```bash
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528

# 回退修改的文件
git checkout HEAD -- pixloc/crop/ray_casting.py
git checkout HEAD -- pixloc/crop/proj2map.py
git checkout HEAD -- pixloc/utils/get_depth.py
git checkout HEAD -- pixloc/pixlib/geometry/costs.py
git checkout HEAD -- main.py

# 或者使用4547 DSM路径
# 修改 main.py 第518行，使用原来的DSM路径
```

---

**结论**: ECEF空间的射线投射在小尺度下存在数值精度问题，不适合用于crop的射线-DSM求交。建议回退到4547方案。
