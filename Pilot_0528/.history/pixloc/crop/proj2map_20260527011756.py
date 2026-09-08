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
import cv2

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

    # print("data size:", band.XSize, "x", band.YSize)
    print("area_minZ (valid minimum elevation):", area_minZ)
    # print("Loading time cost:", time.time() - start, "s")

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



def ray_cast_to_dsm(DSM_path, K, pose, ref_npy_path, geotransform, uv, dsm_transform, ray_area, ray_area_minZ, num_sample=4000):
    """
    使用 DSM 做射线投射，返回射线与 DSM 的最近交点 (world xyz) 以及 DSM 像素索引。
    """
    from .ray_casting import TargetLocation
    # from ray_casting import TargetLocation
    try:
        locator = TargetLocation({"ray_casting": {}}, use_dsm=False)
        hit_point = locator.predict_center_alt(DSM_path, pose, ref_npy_path, geotransform, K, ray_area, ray_area_minZ, num_sample=num_sample, object_pixel_coords=[uv[0], uv[1]])
        hit_rc = geo_coords_to_dsm_index(hit_point[0], hit_point[1], dsm_transform)
        return hit_point, hit_rc
    except Exception:
        return None, None

def crop_dsm_dom(dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, area_minZ,
                 crop_padding=10):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        # print('----------',xyz,'--------------')
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

import numpy as np
import cv2

def crop_dsm_dom_point(DSM_path, pose, ref_npy_path, geotransform, dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, area_minZ, ray_area, ray_area_minZ,
                 crop_padding=10):
    world_points = []
    dsm_indices = []
    for uv in image_points:
        xyz, rc = ray_cast_to_dsm(DSM_path, K, pose, ref_npy_path, geotransform, uv, dsm_transform, ray_area, ray_area_minZ)
        # if xyz is None:
        #     # 回退：与最小高度平面相交
        #     xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        rc = geo_coords_to_dsm_index(xyz[0], xyz[1], dsm_transform)
        world_points.append(xyz)
        dsm_indices.append(rc)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    # 先做一个矩形裁剪，避免对整幅图做透视
    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    # DSM 像素 -> 实际地理坐标
    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)

    # 将角点转换到局部坐标系（以裁剪左上为原点）
    dsm_poly_local = []
    for r, c in dsm_indices:
        dsm_poly_local.append([c - col_min, r - row_min])  # (x, y) = (col, row)
    dsm_poly_local = np.array(dsm_poly_local, dtype=np.float32)

    # 目标输出大小：保持四角点仍是四角点（按多边形边长确定）
    width_top = np.linalg.norm(dsm_poly_local[1] - dsm_poly_local[0])
    width_bottom = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[3])
    height_left = np.linalg.norm(dsm_poly_local[3] - dsm_poly_local[0])
    height_right = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[1])
    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)

    dst_corners = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    # 透视变换矩阵：将 DSM/DOM 的局部四边形映射到规则矩形
    H = cv2.getPerspectiveTransform(dsm_poly_local, dst_corners)

    # DOM: [C, H, W] -> [H, W, C] warp 后再转回
    dom_hwc = np.transpose(dom_crop, (1, 2, 0))
    dom_warp = cv2.warpPerspective(dom_hwc, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    dom_warp = np.transpose(dom_warp, (2, 0, 1))

    # DSM 高程
    # 使用双线性插值获得更平滑的 DSM 高程
    dsm_warp = cv2.warpPerspective(dsm_crop.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # 将 X/Y 坐标一起透视，得到每个像素真实的地理坐标
    x_warp = cv2.warpPerspective(xx.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    y_warp = cv2.warpPerspective(yy.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # 组成 xyz 点云
    xyz = np.stack([x_warp, y_warp, dsm_warp], axis=-1)
    xyz = np.nan_to_num(xyz, nan=0.0)

    dsm_indices = [(col, row) for row, col in dsm_indices]
    return dom_warp, xyz, np.array(dsm_indices), world_points, min(out_w, out_h)

def generate_ref_map(DSM_path, pose, ref_npy_path, geotransform, query_intrinsics, query_poses, name, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path, ray_area, ray_area_minZ, crop_padding=2, debug=True):


    K_w2c = query_intrinsics
    pose_w2c = query_poses
    output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.png')
    output_npy_path = os.path.join(ref_depth_path, f'{name}_dsm.npy')
    width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
    image_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
    # print('image_points',image_points)
    # print('--------------',name,'--------------')
    # image_points = [(width // 2, height // 2)]

    dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape = crop_dsm_dom_point(
        DSM_path, pose, ref_npy_path, geotransform,
        dom_data, dsm_data, dsm_transform,
        K_w2c, pose_w2c, image_points, area_minZ, ray_area, ray_area_minZ, crop_padding
    )
    # print(' world_coords', world_coords)
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
            color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)#
        scale = 0.2  # 缩放比例，可调成 0.1 / 0.3 等
        new_size = (int(dom_vis.shape[1] * scale), int(dom_vis.shape[0] * scale))  # (width, height)

        # 缩放图像
        dom_vis_resized = cv2.resize(dom_vis, new_size, interpolation=cv2.INTER_AREA)
        
        os.makedirs(os.path.join("/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V",'debug_view'), exist_ok=True)
        save_path = os.path.join("/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V",'debug_view',name+'_view.png')
        
        cv2.imwrite(save_path, cv2.cvtColor(dom_vis_resized, cv2.COLOR_RGB2BGR))
        
        # H_direct, _ = cv2.findHomography(np.array(image_points), np.array(dsm_indices), method=cv2.RANSAC, ransacReprojThreshold=3.0)
        # import pdb;pdb.set_trace()
        # print(name, np.min([np.linalg.norm(dsm_indices[1]-dsm_indices[0]),np.linalg.norm(dsm_indices[2]-dsm_indices[3])]) * 0.5)
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
    data = {
            'imgr_name':name+'_dom',
            'exr_name':name+'_dsm',
            'min_shape': min_shape
        }
    return data

import numpy as np

def project_world_to_pixel(K, pose_w2c, world_pt):
    """
    将世界坐标点投影回像素坐标，用于验证 pixel_to_world 的准确性。
    
    Args:
        K (np.array): 3x3 内参矩阵
        pose_w2c (np.array): 4x4 世界到相机的变换矩阵 (Extrinsic)
        world_pt (np.array): [x, y, z] 世界坐标
        
    Returns:
        tuple: (u, v) 投影后的像素坐标
    """
    # 1. 将世界坐标转换为齐次坐标 [x, y, z, 1]
    pt_world_hom = np.append(world_pt, 1.0)
    
    # 2. 将世界坐标转换到相机坐标系: P_cam = T_w2c * P_world
    # 注意：你的 pixel_to_world 中 pose_w2c 被当作 w2c 使用
    pt_cam_hom = pose_w2c @ pt_world_hom
    pt_cam = pt_cam_hom[:3]
    
    # 3. 透视除法前，将相机坐标投影到归一化平面: P_img = K * P_cam
    pt_img_hom = K @ pt_cam
    
    # 4. 透视除法 (Perspective Divide) 归一化: u = x/z, v = y/z
    # z = pt_img_hom[2]
    if pt_img_hom[2] == 0:
        raise ValueError("Point is behind the camera or at z=0")
        
    u = pt_img_hom[0] / pt_img_hom[2]
    v = pt_img_hom[1] / pt_img_hom[2]
    
    return u, v