from re import T
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import time
from pathlib import Path
from osgeo import gdal
# from immatch.utils.transform import parse_pose_list, parse_intrinsic_list
import os
from PIL import Image

def read_DSM_config(ref_dsm, npy_save_path):
    # 读取DSM
    start = time.time()
    dataset = gdal.Open(ref_dsm)
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()
    
    if Path(npy_save_path).exists():
        area = np.load(npy_save_path)
    else:
        area = band.ReadAsArray()
        np.save(npy_save_path, area)

    # 屏蔽无效值（-9999），仅保留有效高程进行 min 统计
    valid_mask = (area > 0) & (area < 10000)
    if not np.any(valid_mask):
        raise ValueError("DSM 无有效高程值，检查是否全为 -9999")
    
    area_minZ = np.min(area[valid_mask])

    print("data size:", band.XSize, "x", band.YSize)
    print("area_minZ (valid minimum elevation):", area_minZ)
    print("Loading time cost:", time.time() - start, "s")

    dataset = None
    return geotransform, area, area_minZ

def pixel_to_world(K, pose_w2c, uv, target_z):
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    cam_center = c2w[:3, 3]
    scale = (target_z - cam_center[2]) / ray_dir[2]
    world_pt = cam_center + ray_dir * scale
    return world_pt

def geo_coords_to_dsm_index(x, y, transform):
    col, row = ~transform * (x, y)
    return int(row), int(col)

def crop_dsm_dom(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, area_minZ,
                 crop_padding=10):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        print('----------',xyz,'--------------')
        idx = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        dsm_indices.append(idx)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    # 确保裁剪区域的高度和宽度不超过200像素
    # row_max = min(row_min + 100, dsm_data.shape[0])  
    # col_max = min(col_min + 100, dsm_data.shape[1])  #

    # 执行数据裁剪
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)
    zz = dsm_crop
    try:
        points_3d = np.stack([xx, yy, zz], axis=-1)
    except:
        # 确保zz与xx/yy形状一致
        zz = np.full_like(xx, fill_value=0.01, dtype=dsm_crop.dtype)

        # 计算dsm_crop在zz中的位置
        row_start = 0
        col_start = 0
        row_end = min(dsm_crop.shape[0], zz.shape[0])
        col_end = min(dsm_crop.shape[1], zz.shape[1])

        # 将dsm_crop的值填入zz的相应位置
        zz[row_start:row_end, col_start:col_end] = dsm_crop[:row_end, :col_end]

        # 处理NaN值（如果有）
        zz = np.nan_to_num(zz, nan=0.01)

        # 现在可以安全地堆叠三个数组
        points_3d = np.stack([xx, yy, zz], axis=-1)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    return dom_crop, points_3d, np.array(dsm_indices), world_points, min(col_max-col_min, row_max-row_min)

def generate_ref_map(query_intrinsics, query_poses, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path, crop_padding=10, debug=True):
    data = {}
    
    for name in query_intrinsics.keys():

        K_w2c = query_intrinsics[name]
        pose_w2c = query_poses[name]
        output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.jpg')
        output_npy_path = os.path.join(ref_depth_path, f'{name}_dom.npy')
        width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
        image_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
        print('image_points',image_points)
        print('--------------',name,'--------------')
        # image_points = [(width // 2, height // 2)]

        dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape = crop_dsm_dom(
            dom_data, dsm_data, dsm_transform,
            K_w2c, pose_w2c, image_points, area_minZ, crop_padding
        )
        print(' world_coords', world_coords)
        if debug:
            import cv2

            # === 4. 准备图像并绘制多边形 ===
            dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()  # [H, W, 3]，注意复制避免修改原图
            cv2.polylines(dom_vis, [dsm_indices.reshape((-1, 1, 2))],
                        isClosed=True, color=(255, 0, 0), thickness=20)

            # 可选：画点
            for idx, (x, y) in enumerate(dsm_indices):
                cv2.circle(dom_vis, (x, y), radius=20, color=(0, 255, 255), thickness=-1)
                cv2.putText(dom_vis, str(idx), (x + 5, y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=10,
                color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

            scale = 0.2  # 缩放比例，可调成 0.1 / 0.3 等
            new_size = (int(dom_vis.shape[1] * scale), int(dom_vis.shape[0] * scale))  # (width, height)

            # 缩放图像
            dom_vis_resized = cv2.resize(dom_vis, new_size, interpolation=cv2.INTER_AREA)
            
            os.makedirs(os.path.join("/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V",'debug_view'), exist_ok=True)
            save_path = os.path.join("/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V",'debug_view',name+'_view.jpg')
            
            cv2.imwrite(save_path, cv2.cvtColor(dom_vis_resized, cv2.COLOR_RGB2BGR))
            
            # H_direct, _ = cv2.findHomography(np.array(image_points), np.array(dsm_indices), method=cv2.RANSAC, ransacReprojThreshold=3.0)
            # import pdb;pdb.set_trace()
            print(name, np.min([np.linalg.norm(dsm_indices[1]-dsm_indices[0]),np.linalg.norm(dsm_indices[2]-dsm_indices[3])]) * 0.5)
            # # === 5. 显示图像 ===
            # plt.figure(figsize=(12, 8))  # 你可以换成 (20, 15) 等更大的值
            # plt.imshow(cv2.cvtColor(dom_vis, cv2.COLOR_BGR2RGB))
            # plt.title("World Points Projected to DOM")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
           
        # 确保 dtype 是 uint8
        cropped_dom_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)

        # [C, H, W] -> [H, W, C]
        if cropped_dom_img_uint8.shape[0] == 3:
            cropped_dom_img_uint8 = np.transpose(cropped_dom_img_uint8, (1, 2, 0))
        Image.fromarray(cropped_dom_img_uint8).save(output_img_path)

        np.save(output_npy_path, point_cloud_crop)
        data[name] = {
                'imgr_name':name+'_dom',
                'exr_name':name+'_dom',
                'min_shape': min_shape
            }
    return data

