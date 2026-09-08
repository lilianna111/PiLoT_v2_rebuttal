import os
from pathlib import Path
import numpy as np
import pyproj


ECEF_TO_WGS84 = pyproj.Transformer.from_crs(
    "EPSG:4978", "EPSG:4326", always_xy=True
)


def is_candidate_npy(name: str) -> bool:
    return name.endswith("_bbox_dom.npy") or name.endswith("_final_points.npy") or name.endswith("_dom.npy")


def sample_valid_points(arr, max_points=20):
    """
    从 (H,W,3) 或 (N,3) 中抽样有效点
    """
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[2] == 3:
        pts = arr.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        pts = arr
    else:
        return None

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]

    return pts


def try_interpret_as_ecef(points):
    """
    尝试把点解释为 ECEF，检查转换到 lon/lat/alt 后是否合理
    """
    results = []
    ok_count = 0

    for p in points:
        x, y, z = map(float, p)
        try:
            lon, lat, alt = ECEF_TO_WGS84.transform(x, y, z)
            valid = (
                np.isfinite(lon) and np.isfinite(lat) and np.isfinite(alt) and
                -180 <= lon <= 180 and
                -90 <= lat <= 90 and
                -1000 <= alt <= 20000
            )
            if valid:
                ok_count += 1
            results.append((x, y, z, lon, lat, alt, valid))
        except Exception:
            results.append((x, y, z, None, None, None, False))

    return results, ok_count


def inspect_one_npy(path: Path):
    print("=" * 100)
    print(f"文件: {path}")

    try:
        arr = np.load(path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    print(f"shape: {arr.shape}, dtype: {arr.dtype}")

    if arr.size == 0:
        print("空数组")
        return

    arr_num = np.asarray(arr)
    finite_mask = np.isfinite(arr_num)
    finite_ratio = finite_mask.mean() if arr_num.size > 0 else 0.0
    print(f"有限值比例: {finite_ratio:.4f}")

    try:
        print(f"全局最小值: {np.nanmin(arr_num):.6f}")
        print(f"全局最大值: {np.nanmax(arr_num):.6f}")
    except Exception:
        pass

    pts = sample_valid_points(arr, max_points=12)
    if pts is None:
        print("不是标准点云格式，预期是 (H,W,3) 或 (N,3)")
        return

    if len(pts) == 0:
        print("没有有效 3D 点")
        return

    print("\n抽样点（原始值）:")
    for i, p in enumerate(pts[:5]):
        print(f"  [{i}] {p}")

    # 检查数值尺度
    norms = np.linalg.norm(pts, axis=1)
    print(f"\n抽样点模长范围: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}")

    # 尝试解释为 ECEF
    ecef_results, ok_count = try_interpret_as_ecef(pts)

    print("\n按 ECEF 解释后的抽样结果:")
    for i, item in enumerate(ecef_results[:5]):
        x, y, z, lon, lat, alt, valid = item
        if valid:
            print(f"  [{i}] lon={lon:.8f}, lat={lat:.8f}, alt={alt:.3f}")
        else:
            print(f"  [{i}] 无法合理解释为 ECEF")

    print(f"\nECEF 合理点比例: {ok_count}/{len(pts)}")

    # 简单判断
    # 地球半径大约 6371km，所以 ECEF 点模长一般在 6.3e6 左右
    ecef_like_norm = np.mean((norms > 6.0e6) & (norms < 6.5e6))
    planar_like_range = np.nanmax(np.abs(pts[:, :2])) < 1e6

    print("\n判断:")
    if ok_count >= max(3, len(pts) // 2) and ecef_like_norm > 0.5:
        print("  很像 ECEF XYZ")
    elif planar_like_range:
        print("  更像局部/投影平面坐标，不太像 ECEF")
    else:
        print("  不能直接确定，建议结合生成代码再判断")


def main(root_dir):
    root = Path(root_dir)
    if not root.exists():
        print(f"路径不存在: {root}")
        return

    npy_files = [p for p in root.rglob("*.npy") if is_candidate_npy(p.name)]

    if not npy_files:
        print(f"没有找到目标 NPY 文件: {root}")
        return

    print(f"扫描目录: {root}")
    print(f"找到 {len(npy_files)} 个候选 NPY 文件\n")

    for f in sorted(npy_files):
        inspect_one_npy(f)


if __name__ == "__main__":
    root_dir = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"
    main(root_dir)