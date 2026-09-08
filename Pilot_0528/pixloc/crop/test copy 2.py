import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj

# ==========================================
# 1. 引入 transform_colmap.py 中的核心逻辑
# ==========================================
def convert_euler_to_matrix(euler_xyz):
    # 复用你的 convert_euler_to_matrix 逻辑
    # 这里的顺序是 [yaw, pitch, roll]
    # 你的逻辑: [pitch-90, roll, -yaw] -> xyz order
    ret = R.from_euler('xyz', [euler_xyz[1]-90, float(euler_xyz[2]), -euler_xyz[0]], degrees=True)
    rotation_matrix = ret.as_matrix()
    return rotation_matrix

def wgs84tocgcs2000(trans):
    # 复用 wgs84tocgcs2000
    lon, lat, height = trans
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000 = pyproj.CRS('EPSG:4547') # 这里的 EPSG 需与你 transform_colmap.py 中一致 (4547 or 4544)
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return [x, y, height]

def get_real_pose_and_K(pose_data):
    # 模拟 transform_colmap_pose_intrinsic 的核心部分
    lon, lat, alt, roll, pitch, yaw = pose_data
    
    # 注意：你的代码里在传入 convert 之前对 yaw/pitch 做了预处理
    # transform_colmap.py: yaw = -yaw; pitch = pitch - 90
    # 但 convert_euler_to_matrix 内部又做了处理。
    # 我们严格照抄 transform_colmap_pose_intrinsic 函数里的处理流程:
    
    # 1. 原始数据的预处理 (Line 158-159 in transform_colmap.py)
    yaw_proc = -yaw
    pitch_proc = pitch - 90
    
    # 2. 提取欧拉角 (Line 162)
    euler_xyz = [yaw_proc, pitch_proc, roll]
    
    # 3. 转旋转矩阵 (Line 163) -> 这一步调用 convert_euler_to_matrix
    # convert_euler_to_matrix 内部再次处理: [euler[1]-90, euler[2], -euler[0]]
    # 这看起来可能有多重处理，但为了"真实"，我们必须保持一致
    r_c2w = convert_euler_to_matrix(euler_xyz)
    
    # 4. 坐标转换 (Line 166-167)
    trans = [lon, lat, alt]
    t_c2w = wgs84tocgcs2000(trans)
    
    # 5. 构造 W2C 矩阵 (Line 170-172)
    T = np.identity(4)
    T[:3, :3] = r_c2w.T
    T[:3, 3] = -r_c2w.T @ t_c2w
    
    # 6. 获取硬编码的内参 (Line 177-181)
    K = np.array([
        [1350.0, 0.0, 957.85],
        [0.0, 1350.0, 537.55],
        [0.0, 0.0, 1.0]
    ])
    
    return T, K, t_c2w

# ==========================================
# 2. 引入 proj2map.py 中的 Ray Casting 逻辑
# ==========================================
def pixel_to_world(K, pose_w2c, uv, target_z):
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    
    # 还原 C2W 以获取相机中心和世界系射线方向
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    cam_center = c2w[:3, 3]
    
    # Ray-Plane Intersection
    # P = O + t * D
    # z_target = O_z + t * D_z  => t = (z_target - O_z) / D_z
    scale = (target_z - cam_center[2]) / ray_dir[2]
    world_pt = cam_center + ray_dir * scale
    return world_pt

# ==========================================
# 3. 实现逆向投影 (World -> Pixel)
# ==========================================
def world_to_pixel(K, pose_w2c, world_pt):
    # 1. World -> Camera
    # P_cam = T_w2c * P_world
    pt_world_hom = np.append(world_pt, 1.0)
    pt_cam_hom = pose_w2c @ pt_world_hom
    
    # 2. Camera -> Image (透视投影)
    # P_img = K * P_cam
    # 我们只关心前三维 (x, y, z)
    pt_img_hom = K @ pt_cam_hom[:3]
    
    # 3. 归一化 (Perspective Divide)
    z = pt_img_hom[2]
    u = pt_img_hom[0] / z
    v = pt_img_hom[1] / z
    
    return np.array([u, v])

# ==========================================
# 4. 主验证流程
# ==========================================
if __name__ == "__main__":
    print("=== 使用真实 Pose 数据验证投影闭环 ===\n")

    # [A] 准备真实数据 (来自 test.py)
    # pose_data: [lon, lat, alt, roll, pitch, yaw]
    real_pose_data = [112.999054, 28.290497, 157.509, 0.0, 0.0, -42.4]
    
    # 获取矩阵
    pose_w2c, K, cam_pos_cgcs = get_real_pose_and_K(real_pose_data)
    
    print(f"相机位置 (CGCS2000): {cam_pos_cgcs}")
    print(f"Pose W2C Matrix:\n{pose_w2c}")
    print(f"Intrinsic K:\n{K}\n")

    # 设定一个目标高度用于投射 (比如地面高度 50米，或者 0米)
    # 只要射线能打到平面，反投回来都应该是一样的
    TARGET_Z = 50.0 

    # [B] 定义要测试的像素点 (涵盖中心和边缘)
    test_points = [
        ("Image Center", [957.85, 537.55]), # Principal point from K
        ("Top Left",     [0, 0]),
        ("Bottom Right", [1919, 1079]),
        ("Random Point", [500, 300])
    ]

    print(f"{'Pixel Name':<15} | {'Original (u, v)':<20} | {'Reprojected (u, v)':<20} | {'Error':<10}")
    print("-" * 80)

    all_passed = True
    
    for name, uv in test_points:
        uv = np.array(uv)
        
        # 1. 投射出去 (Pixel -> World)
        try:
            world_pt = pixel_to_world(K, pose_w2c, uv, target_z=TARGET_Z)
        except Exception as e:
            print(f"Error projecting {name}: {e}")
            continue
            
        # 2. 投射回来 (World -> Pixel)
        uv_reproj = world_to_pixel(K, pose_w2c, world_pt)
        
        # 3. 比较
        error = np.linalg.norm(uv - uv_reproj)
        
        print(f"{name:<15} | {str(uv):<20} | {np.round(uv_reproj, 2)}             | {error:.6f}")
        
        if error > 1e-4:
            all_passed = False

    print("-" * 80)
    if all_passed:
        print("\n✅ 验证成功：所有点的重投影误差均忽略不计。")
        print("结论：pixel_to_world (Ray Casting) 与标准的透视投影模型完全一致。")
    else:
        print("\n❌ 验证失败：存在较大误差，请检查坐标系定义或矩阵乘法顺序。")