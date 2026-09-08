import os
import cv2
import numpy as np
from utils import read_DSM_config
from transform_colmap import transform_colmap_pose_intrinsic
from proj2map_test import generate_ref_map

# --- 核心函数：支持 NPY 的同步旋转与裁剪 ---

def apply_transform_to_npy(npy_data, M, new_size, x_min, y_min, target_w, target_h):
    """
    对 (H, W, 3) 的 NPY 数据进行旋转和裁剪，保持与图像像素对齐
    """
    # 1. 旋转 (使用 INTER_NEAREST 保证坐标数值不被插值改变)
    rotated_npy = cv2.warpAffine(npy_data, M, new_size, flags=cv2.INTER_NEAREST, borderValue=0)
    # 2. 裁剪
    cropped_npy = rotated_npy[y_min:y_min + target_h, x_min:x_min + target_w]
    return cropped_npy

def rotate_all_with_mask(image, npy_data, angle, mask=None, crop_padding=0):
    """
    同步旋转图像、NPY 和 Mask，并根据有效区域裁边
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    new_h = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))
    
    # 构造旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    
    # 1. 旋转图像
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    
    # 2. 旋转 Mask 并确定裁剪边界
    if mask is not None:
        rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
        ys, xs = np.where(rotated_mask > 0)
        if len(xs) > 0:
            # 这里的 crop_padding 设为 0 可以最大限度去掉边缘
            x_min, x_max = max(int(xs.min()) - crop_padding, 0), min(int(xs.max()) + crop_padding + 1, new_w)
            y_min, y_max = max(int(ys.min()) - crop_padding, 0), min(int(ys.max()) + crop_padding + 1, new_h)
            
            # 执行裁剪
            final_img = rotated_img[y_min:y_max, x_min:x_max]
            final_mask = rotated_mask[y_min:y_max, x_min:x_max]
            
            # 3. 同步处理 NPY
            final_npy = apply_transform_to_npy(npy_data, M, (new_w, new_h), x_min, y_min, 
                                               final_img.shape[1], final_img.shape[0])
            
            return final_img, final_npy, final_mask
            
    return rotated_img, None, None

# --- 主流程 ---

# 1. 基础配置
query_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/images/interval1_HKairport01/interval1"
ref_save_dir = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"
ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif"
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3.npy"

pose_data = [112.999054, 28.290497, 157.509, 0.0, 37.1, -42.4]
name = "1"

# 2. 地图初始化与 Stage 1 裁剪
geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
query_prior_poses_dict, query_intrinsics_dict, _, osg_dict = transform_colmap_pose_intrinsic(pose_data)

# 执行 Stage 1 并保存中间文件
generate_ref_map(name, query_intrinsics_dict, query_prior_poses_dict, area_minZ, dsm_data, dsm_trans, dom_data, 
                 ref_save_dir, ref_save_dir, debug=True, use_bbox=True)

# 3. 开始 Stage 2：同步旋转对齐
print(f"--- 正在执行像素级对齐旋转 ---")

# 加载 Stage 1 生成的文件
img_aabb = cv2.imread(os.path.join(ref_save_dir, f"{name}_bbox_dom.jpg"))
mask_aabb = cv2.imread(os.path.join(ref_save_dir, f"{name}_bbox_dom_mask.png"), cv2.IMREAD_GRAYSCALE)
npy_aabb = np.load(os.path.join(ref_save_dir, f"{name}_bbox_dom.npy")) # 注意加载正确路径

yaw_angle = osg_dict[0]

# 执行同步变换 (padding设为0以减少边框)
final_img, final_npy, final_mask = rotate_all_with_mask(img_aabb, npy_aabb, yaw_angle, mask=mask_aabb, crop_padding=0)

# 4. 保存最终成果
if final_img is not None:
    # 保存图像
    cv2.imwrite(os.path.join(ref_save_dir, f"{name}_final_img.jpg"), final_img)
    # 保存 NPY (现在它与 final_img 的像素一一对应)
    np.save(os.path.join(ref_save_dir, f"{name}_final_points.npy"), final_npy)
    # 保存最终 Mask (用于下游任务过滤有效区域)
    cv2.imwrite(os.path.join(ref_save_dir, f"{name}_final_mask.png"), final_mask)
    
    print(f"✅ 成功生成对齐数据集：")
    print(f"   图像: {final_img.shape} -> _final_img.jpg")
    print(f"   坐标: {final_npy.shape} -> _final_points.npy")
    print(f"   掩码: {final_mask.shape} -> _final_mask.png")