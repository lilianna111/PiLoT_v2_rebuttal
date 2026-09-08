import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_normal_from_npy(npy_path, output_depth_path, output_normal_path):
    # 1. 加载 .npy 文件
    depth = np.load(npy_path).astype(np.float32)
    if len(depth.shape) == 3:
        depth = depth.squeeze()

    # 2. 处理无效值
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 计算梯度 (Sobel 算子)
    zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

    # 4. 构造法向量 (这里建议 Z 设为 0.1 左右以匹配你的起伏需求)
    normal = np.dstack((-zx, -zy, np.ones_like(depth) * 0.1))
    
    # 5. 归一化
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-6)

    # 6. 将分量映射到 [0, 255]
    normal_img = (normal + 1.0) * 127.5
    normal_img = normal_img.astype(np.uint8)

    # --- 7. 分开保存 ---
    
    # 保存左边的图：深度可视化图 (模拟 plt 的 jet 效果)
    # 先归一化到 0-255，再应用 JET 伪彩色
    depth_min, depth_max = np.min(depth), np.max(depth)
    depth_adj = (depth - depth_min) / (depth_max - depth_min + 1e-6) * 255
    depth_color = cv2.applyColorMap(depth_adj.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(output_depth_path, depth_color)
    
    # 保存右边的图：法向图
    cv2.imwrite(output_normal_path, cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
    
    # 8. 保持显示供确认
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Depth (Saved)")
    plt.imshow(depth, cmap='jet')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Normal (Saved)")
    plt.imshow(normal_img)
    plt.axis('off')
    plt.show()

# 使用示例
input_npy = '/media/amax/AE0E2AFD0E2ABE69/datasets/renders_high/DJI_20250612193930_0012_V/2_0.npy'
out_depth = '/media/amax/AE0E2AFD0E2ABE69/datasets/renders_high/DJI_20250612193930_0012_V_depth_vis.png'
out_normal = '/media/amax/AE0E2AFD0E2ABE69/datasets/renders_high/DJI_20250612193930_0012_V_normal.png'

generate_normal_from_npy(input_npy, out_depth, out_normal)