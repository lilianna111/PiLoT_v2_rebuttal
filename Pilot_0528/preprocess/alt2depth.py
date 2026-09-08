import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def euler_to_rotation_matrix(pitch_deg, roll_deg, yaw_deg):
    """
    将欧拉角转换为旋转矩阵 (假设顺序为 Yaw -> Pitch -> Roll，这是航空常用顺序，可根据实际调整)
    注意：DJI 的 Gimbal 数据通常需要仔细校对坐标系定义
    """
    r_x = np.radians(roll_deg)
    r_y = np.radians(pitch_deg)
    r_z = np.radians(yaw_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r_x), -np.sin(r_x)],
                   [0, np.sin(r_x), np.cos(r_x)]])

    Ry = np.array([[np.cos(r_y), 0, np.sin(r_y)],
                   [0, 1, 0],
                   [-np.sin(r_y), 0, np.cos(r_y)]])

    Rz = np.array([[np.cos(r_z), -np.sin(r_z), 0],
                   [np.sin(r_z), np.cos(r_z), 0],
                   [0, 0, 1]])

    # 旋转矩阵 R_wc (Camera to World)
    # 这里的乘法顺序取决于你的坐标系定义，通常是 Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R

def process_and_visualize(image_path, npy_path, pose_info, intrinsic_K):
    # 1. 读取数据
    if not os.path.exists(image_path) or not os.path.exists(npy_path):
        print(f"错误: 文件不存在 \n{image_path}\n{npy_path}")
        return

    # 读取原图 (BGR -> RGB)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 读取高度图 (H, W)
    # 假设 npy 中存储的是每个像素对应的绝对海拔高度 (Absolute Altitude)
    pixel_heights = np.load(npy_path)
    
    H, W = pixel_heights.shape
    
    # 2. 计算几何深度 (Geometric Depth)
    # 准备相机参数
    camera_alt = pose_info['alt']
    fx = intrinsic_K[0, 0]
    fy = intrinsic_K[1, 1]
    cx = intrinsic_K[0, 2]
    cy = intrinsic_K[1, 2]

    # 生成像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 归一化相机坐标 (z=1)
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    z_norm = np.ones_like(x_norm)
    
    # 堆叠为向量 (3, N)
    rays_cam = np.stack([x_norm, y_norm, z_norm], axis=0).reshape(3, -1)
    
    # 计算旋转矩阵并旋转射线
    R_wc = euler_to_rotation_matrix(pose_info['pitch'], pose_info['roll'], pose_info['yaw'])
    rays_world = R_wc @ rays_cam  # (3, N)
    
    # 提取射线的垂直分量 (Z分量)
    # 如果是 NED 坐标系，Z轴向下，Altitude 是负 Z；如果是 ENU，Z轴向上
    # 这里假设通用情况：我们只关心射线在垂直方向上的投影比例
    # 如果相机向下看，rays_world_z 应该是负值（或正值取决于定义），我们取绝对值处理投影关系
    rays_world_z = rays_world[2, :] 
    
    # 计算高度差 (垂直距离)
    # 展平高度图
    pixel_heights_flat = pixel_heights.reshape(-1)
    
    # delta_H = 相机高度 - 像素点高度
    # 这是一个标量场，表示每个像素点离相机平面的垂直距离
    vertical_dist = camera_alt - pixel_heights_flat
    
    # 3. 核心深度公式
    # True Depth (直线距离) = Vertical Distance / cos(theta)
    # cos(theta) 等于 射线单位向量 的 Z 分量
    # 但我们这里 rays_world 还没归一化，所以公式是:
    # Scale = vertical_dist / rays_world_z
    # Depth = Scale * norm(rays_cam)  <-- 因为 rays_cam 的 z=1
    
    # 为了避免除以0，加一个极小值
    epsilon = 1e-6
    # 注意符号：如果相机在上方，vertical_dist > 0，rays_world_z 应该指向下方
    # 如果你的坐标系 Z 轴向上，rays_world_z 应该是负的。我们取符号一致即可。
    scale = vertical_dist / (np.abs(rays_world_z) + epsilon)
    
    # 真正的欧氏距离 (Euclidean Distance / Range)
    rays_cam_norm = np.linalg.norm(rays_cam, axis=0)
    depth_map = scale * rays_cam_norm
    
    # 重塑回图像形状
    depth_map = depth_map.reshape(H, W)
    
    # 简单的过滤：去除异常大的值（天空或无限远）或负值
    depth_map[depth_map < 0] = 0
    depth_map[depth_map > 1000] = 1000 # 限制最大显示深度为1000米，方便可视化
    
    # 4. 可视化
    plt.figure(figsize=(15, 6))
    
    # 左图：原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Image: {os.path.basename(image_path)}")
    plt.axis('off')
    
    # 右图：深度图
    plt.subplot(1, 2, 2)
    # 使用 'inferno' 或 'magma' 色图，这在深度图中很常见（黑=近/远，亮=远/近）
    # 通常：近处亮（黄色），远处暗（黑色/紫色）或者反过来
    plt.imshow(depth_map, cmap='inferno_r') 
    plt.colorbar(label='Depth (meters)')
    plt.title("Generated Depth Map")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ================= 配置区域 =================
# 请替换为你本地的实际路径
image_file = "0_0.png"   # 你的原图路径
npy_file = "0_0_height.npy" # 你的高度图路径

# 根据 txt 文件第一行数据填入 (0_0.png)
# 113.002559 28.291724 178.933397 -1.091164 1.689589 159.902282
current_pose = {
    'alt': 178.933397,   # 相机高度
    'pitch': -1.091164,  # 请确认这是 Pitch
    'roll': 1.689589,    # 请确认这是 Roll
    'yaw': 159.902282    # 请确认这是 Yaw
}

# 你的相机内参 (需要根据实际情况修改)
# 举例: DJI Mavic 3 大概 fx,fy 在 3000 左右，如果是 resize 过的图要相应缩放
K = np.array([
    [1000, 0, 960],  # fx, 0, cx
    [0, 1000, 540],  # 0, fy, cy
    [0, 0, 1]
])

# 运行
# process_and_visualize(image_file, npy_file, current_pose, K)