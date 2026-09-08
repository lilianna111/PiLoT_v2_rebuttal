import warnings
warnings.filterwarnings('ignore')
import torch
import yaml
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import logging
import sys
from datetime import datetime
# from utils.memory import RealTimeGPUMonitor
# import immatch

# from immatch.utils.transform import *
# from immatch.utils.visualize import plot_match_dji,plot_match
# from immatch.utils.util import aggregate_stats
# from immatch.localize.dom_localize import DOMLocalizer
# from immatch.localize.hlocalize_domdsm_exp import HSLocalizer
# from immatch.utils.transform_colmap_video import transform_colmap_pose_intrinsic
# from utils.generate_dji import generate_dji_seed
# from immatch.utils.proj2map_video_duibi_crop import generate_ref_map
# from immatch.localize.ray_localization_exp import RayLocalizer

# from utils.utils import *
from utils.config_manager import ConfigManager
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # 取消限制

# 加载 select/ 文件夹中的场景图片列表
SELECT_FOLDER = "select/200m"  # 指定要使用的 select 文件夹
# ALL_SCENE_IMAGES = ConfigManager.load_images_from_select_folder(SELECT_FOLDER)
ALL_SCENE_IMAGES = None


def rotate_ref_with_mask_and_crop(image, angle, mask=None, crop_padding=2):
    """
    旋转 reference 图像；若提供 mask，则按旋转后的有效区域自动裁边。

    返回:
        rotated_cropped_img, H_final
        其中 H_final 是从“原始 reference 图坐标”到“旋转并裁边后的图坐标”的 3x3 变换矩阵
    """
    rotated_img, rot_H = rotate_image(image, angle)

    if mask is None:
        return rotated_img, rot_H

    # 旋转 mask。mask 使用最近邻，边界填0
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    new_h = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

    ys, xs = np.where(rotated_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rotated_img, rot_H

    x_min = max(int(xs.min()) - crop_padding, 0)
    x_max = min(int(xs.max()) + crop_padding + 1, rotated_img.shape[1])
    y_min = max(int(ys.min()) - crop_padding, 0)
    y_max = min(int(ys.max()) + crop_padding + 1, rotated_img.shape[0])

    rotated_cropped_img = rotated_img[y_min:y_max, x_min:x_max]

    # 最终变换 = 裁边平移 * 旋转变换
    crop_T = np.array([
        [1.0, 0.0, -x_min],
        [0.0, 1.0, -y_min],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_final = crop_T @ rot_H

    return rotated_cropped_img, H_final

def setup_logging(log_file=None):
    """设置日志配置，同时输出到终端和文件"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/experiment_{timestamp}.log"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

# config_path = "target_config/config_blender_targetloc_dji_video_exp.yml"
config_path = "target_config/config_duibi.yml"
config_manager = ConfigManager(config_path)
config = config_manager.get_original_config()

def save_match_plot_multi_ratio(qname, mathes, ori_path, rot_r_save_path, match_save_path, ratio_suffix=""):
    """
    为多比例定位保存match图
    ratio_suffix: 比例后缀，如 "_ratio_10"
    """
    from immatch.utils.visualize import plot_match
    match_name = qname + ratio_suffix
    plot_match(ori_path, ori_path, rot_r_save_path, mathes, match_save_path, match_name, rot_R=None, percent=1.0)

def localize_single_ratio(qname, ratio_key, data, reference_intrinsic, reference_poses, reference_yaw, 
                         q_intrinsic, q_prior_pose, query_path, ref_rgb_path, ref_depth_path, rot_save_path, 
                         rot_flag, match_save_path, seed_num, model, 
                         fr_result, fh_result, ray_tracing_flag, homography_flag, 
                         homography_save_path, ray_trace_save_path,
                         dsm_data, dsm_trans, q_points, gt_dict, ratio_suffix=""):
    """
    为单个比例执行完整的匹配和定位流程
    ratio_suffix: 用于保存结果文件的后缀，如 "_ratio10"
    """
    logger = logging.getLogger()
    matches_list = []
    matches_number_list = []
    rot_matches_refer_points = []
    rot_matrices = []
    ori_path = os.path.join(query_path, qname + '.jpg')
    
    for i in range(seed_num):
        rname = qname + '_' + str(i)
        
        # 使用对应的比例数据
        if ratio_key in data:
            imgr_path = os.path.join(ref_rgb_path, data[ratio_key]['imgr_name'] + '.jpg')
            exr_path = os.path.join(ref_depth_path, data[ratio_key]['exr_name'] + '.npy')
            min_shape = data[ratio_key]['min_shape']
        else:
            imgr_path = os.path.join(ref_rgb_path, data[rname]['imgr_name'] + '.jpg')
            exr_path = os.path.join(ref_depth_path, data[rname]['exr_name'] + '.npy')
            min_shape = data[rname]['min_shape']
        
        resize = resize_value(min_shape)
        
        if rot_flag:
            imgr = cv2.imread(imgr_path)
            mask_path = imgr_path.replace('.jpg', '_mask.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
            roted_img, rot_R = rotate_ref_with_mask_and_crop(imgr, reference_yaw[rname], mask=mask, crop_padding=2)
            rot_r_save_path = os.path.join(rot_save_path, data[ratio_key if ratio_key in data else rname]['imgr_name'] + '.jpg')
            cv2.imwrite(rot_r_save_path, roted_img)
        else:
            rot_r_save_path = imgr_path
            rot_R = None
        

        with RealTimeGPUMonitor(interval=0.05):
            mathes, _, _, _ = model.match_pairs(ori_path, rot_r_save_path)


        matches_list.append(mathes)
        matches_number_list.append(len(mathes))
        rot_matrices.append(rot_R)
        
        rmatches = np.column_stack((mathes[:, 2], mathes[:, 3]))
        if rot_R is not None:
            ones_column = np.ones((rmatches.shape[0], 1))
            ref_pts2d_homo = np.hstack((rmatches, ones_column))
            ref_pts2d_homo = np.dot(np.linalg.inv(rot_R), ref_pts2d_homo.T)
            rot_reference_match = (ref_pts2d_homo[:2, ] / ref_pts2d_homo[2,:]).T
        else:
            rot_reference_match = rmatches
        rot_matches_refer_points.append(rot_reference_match)
    
    # 检查是否有匹配结果
    if len(matches_number_list) == 0 or all(count == 0 for count in matches_number_list):
        logger.warning(f"警告: {qname}{ratio_suffix} 没有找到任何匹配点")
        return
    
    mathes_index = np.argmax(matches_number_list)
    mathes = matches_list[mathes_index]
    rname = qname + '_' + str(mathes_index)
    imgr_path = os.path.join(ref_rgb_path, data[ratio_key if ratio_key in data else rname]['imgr_name'] + '.jpg')
    exr_path = os.path.join(ref_depth_path, data[ratio_key if ratio_key in data else rname]['exr_name'] + '.npy')
    
    # 获取旋转后的图片路径和旋转矩阵
    current_rot_R = rot_matrices[mathes_index] if mathes_index < len(rot_matrices) else None
    if rot_flag and current_rot_R is not None:
        rot_r_save_path_local = os.path.join(rot_save_path, data[ratio_key if ratio_key in data else rname]['imgr_name'] + '.jpg')
    else:
        rot_r_save_path_local = imgr_path
    
    # 保存match图
    ori_path_local = os.path.join(query_path, qname + '.jpg')
    save_match_plot_multi_ratio(qname, mathes, ori_path_local, rot_r_save_path_local, match_save_path, ratio_suffix)
    
    # rot match point
    mathes[:, 2:] = rot_matches_refer_points[mathes_index]
    
    # 射线追踪定位
    if ray_tracing_flag:
        localizer = DOMLocalizer(
            K1=q_intrinsic,
            points_path=exr_path,
            matches=mathes
        )
        pred_matrix = localizer.main_localize()
        
        ray_localizer = RayLocalizer(dsm_data, dsm_trans, pred_matrix, q_intrinsic, qname + ratio_suffix, fr_result, imgr_path)
        ray_points_save_path = os.path.join(ray_trace_save_path, qname + f'{ratio_suffix}_ray_points.npz')
        _, points_3d = ray_localizer.localize_from_json(q_points, ori_path, ray_points_save_path, current_rot_R)
        ray_localizer.print_error(qname + ratio_suffix, points_3d, q_points, gt_dict, fr_result)
    
    # 单应性定位
    if homography_flag:
        t_localizer = HSLocalizer(mathes, exr_path)
        save_warp_path = os.path.join(homography_save_path, qname + f'{ratio_suffix}.jpg')
        save_warp_point_path = os.path.join(homography_save_path, qname + f'{ratio_suffix}point')
        H_q2r = t_localizer.main_localize_warp_fig(ori_path, imgr_path, save_warp_path)
        if H_q2r is None:
            if fh_result is not None:
                t_localizer.print_error(qname + ratio_suffix, [], q_points, gt_dict, fh_result, hq=True)
        else:
            q_pts, r_pts, point_3d = t_localizer.mapping_from_json(q_points, ori_path, imgr_path, H_q2r, exr_path, save_warp_point_path)
            if fh_result is not None:
                t_localizer.print_error(qname + ratio_suffix, point_3d, q_points, gt_dict, fh_result)


def print_gpu_memory():
    if torch.cuda.is_available():
        # 1. 当前张量实际占用的显存
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        # 2. PyTorch 缓存管理器预留的总显存（包含未使用的缓存）
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        # 3. 历史峰值显存（非常重要，OOM 通常是因为这个值爆了）
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"当前张量占用: {allocated:.2f} MB")
        print(f"当前缓存占用: {reserved:.2f} MB")
        print(f"历史峰值占用: {max_allocated:.2f} MB")
        print("-" * 30)
    else:
        print("CUDA不可用")


def run_single_experiment(method, modality, scene, map_type=None):
    """运行单个实验，保持原有的所有结构和输出路径"""
    logger = logging.getLogger()
    logger.info(f"\n{'='*60}")
    logger.info(f"开始实验: {method.upper()} + {modality.upper()} + {scene.upper()}")
    if map_type:
        logger.info(f"地图类型: {map_type.upper()}")
    logger.info(f"{'='*60}")
    
    # 使用配置管理器更新配置
    current_config = config_manager.update_config_for_experiment(method, modality, scene, map_type)
    
    try:
        ##------------------initial-----------------------------
        match_method = current_config['method']
        device = current_config['device']
        with open(f"configs/{match_method}.yml", "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["example"]
            
        model = immatch.__dict__[args["class"]](args)

        ##-------------------debug----------------------------
        debug = current_config['debug']
        debug_path = current_config.get('debug_path', 'output/debug_crop_visualization')
        test_all_ratios = current_config.get('test_all_ratios', False)
        

        ##-------------------rot------------------------------
        rot_config = current_config["rot"]
        rot_flag = rot_config["rot_flag"]
        rot_save_path = rot_config["rot_save_path"]
        os.makedirs(rot_save_path, exist_ok=True)

        ##------------------seed-------------------------------
        seed_config = current_config["seed"]
        seed_int_trans_x = seed_config["seed_int_trans_x"]
        seed_num_trans_x = seed_config["seed_num_trans_x"]
        seed_int_trans_y = seed_config["seed_int_trans_y"]
        seed_num_trans_y = seed_config["seed_num_trans_y"]
        seed_int_trans_z = seed_config["seed_int_trans_z"]
        seed_num_trans_z = seed_config["seed_num_trans_z"]
        seed_int_yaw = seed_config["seed_int_yaw"]
        seed_num_yaw = seed_config["seed_num_yaw"]

        ##-----------------query------------------------------
        query_config = current_config["query"]
        query_path = query_config["query_path"]
        query_txt_path = query_config["query_txt_path"]
        pixel_json_path = query_config["pixel_json_path"]
        gt_path = query_config["GT_path"]

        ##-----------------reference------------------------------
        ref_config = current_config["ref"]
        ref_path = ref_config["ref_path"]
        ref_DSM_path = ref_config["ref_DSM_path"]
        ref_npy_path = ref_config["ref_npy_path"]
        ref_DOM_path = ref_config["ref_DOM_path"]
        ref_rgb_path = ref_config["ref_rgb_path"]
        ref_depth_path = ref_config["ref_depth_path"]

        os.makedirs(ref_rgb_path, exist_ok=True)
        os.makedirs(ref_depth_path, exist_ok=True)
        seed_num = seed_num_trans_x * seed_num_trans_y * seed_num_trans_z * seed_num_yaw
        logger.info(f'------------------seed_num: {seed_num} ------------------')

        ##------------------save_path---------------------------
        result_config = current_config["result"]
        match_save_path = result_config["match_save_path"]
        os.makedirs(match_save_path, exist_ok=True)
        homography_flag = result_config["homography_flag"]
        if homography_flag:
            homography_save_path = result_config["homography_save_path"]
            os.makedirs(homography_save_path, exist_ok=True)
            homography_result_json = result_config["homography_json"]
            # 确保homography_result_json的目录存在
            os.makedirs(os.path.dirname(homography_result_json), exist_ok=True)
        else:
            # 如果homography_flag为false，设置为None
            homography_save_path = None
            homography_result_json = None
        ray_tracing_flag = result_config["ray_trace_flag"]
        # ray_tracing_result_txt = result_config["ray_trace_txt"]
        ray_trace_save_path = result_config["ray_trace_save_path"]
        ray_trace_json = result_config["ray_trace_json"]
        # 确保ray_tracing相关目录存在
        # os.makedirs(os.path.dirname(ray_tracing_result_txt), exist_ok=True)
        os.makedirs(ray_trace_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(ray_trace_json), exist_ok=True)

        # init map
        geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)

        ## Read prior pose and intrinsic
        query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(query_path, query_txt_path)

        # 统一在下方基于场景与模态的分支进行过滤

        ##load gt points
        gt_dict = load_gt_points(gt_path)

        ##load query point
        query_pixel_dict = load_pixel_point(pixel_json_path) if pixel_json_path else {}
        
        # 如果使用 select 文件夹，则过滤图像
        if ALL_SCENE_IMAGES:
            # 处理场景名称映射：先尝试原始名称，再尝试基础名称
            # 例如：scene7_ther -> 先找 scene7_ther，找不到再找 scene7
            
            # 优先使用原始场景名，如果找不到则使用基础场景名
            target_scene = scene
            # 查找当前场景的图片列表
            if target_scene in ALL_SCENE_IMAGES:
                all_images = ALL_SCENE_IMAGES[target_scene]
                # 根据模态过滤图像
                filtered_images = [img for img in all_images if modality in img['img_path'].lower()]
                
                if filtered_images:
                    # 获取所有需要的图像名称（不包含扩展名）
                    selected_img_names = {img['img_name'] for img in filtered_images}
                    # 过滤 query_prior_poses_dict 和 query_intrinsics_dict
                    query_prior_poses_dict = {k: v for k, v in query_prior_poses_dict.items() if k in selected_img_names}
                    query_intrinsics_dict = {k: v for k, v in query_intrinsics_dict.items() if k in selected_img_names}
                    q_intrinsics_info = {k: v for k, v in q_intrinsics_info.items() if k in selected_img_names}
                    osg_dict = {k: v for k, v in osg_dict.items() if k in selected_img_names}
                    query_pixel_dict = {k: v for k, v in query_pixel_dict.items() if k in selected_img_names}
                    
                    logger.info(f"✅ 从 {SELECT_FOLDER}/{target_scene}.txt 加载图像：{len(selected_img_names)} 张图像 (模态: {modality})")
                    logger.info(f"📋 选中的图像：{', '.join(sorted(list(selected_img_names)[:10]))}{'...' if len(selected_img_names) > 10 else ''}")
                else:
                    logger.warning(f"⚠️ 在 {SELECT_FOLDER}/{target_scene}.txt 中未找到 {modality} 模态的图像")
                    return False
            else:
                logger.warning(f"⚠️ 未找到 {SELECT_FOLDER}/{target_scene}.txt 文件 (场景: {scene})")
                return False
        else:
            logger.info("ℹ️ 未使用 select 文件夹，将使用 query_path 中的所有图像")

        # 打开ray_trace文件
        with open(ray_trace_json, "a", encoding="utf-8") as fr_result:
            # 如果homography_flag为true，也打开homography文件
            if homography_flag:
                fh_result = open(homography_result_json, "a", encoding="utf-8")
            else:
                fh_result = None
            # 单张图片测试模式
            qcfg = current_config.get('query', {})
            single_enable = bool(qcfg.get('single_query_enable', False))
            single_query_names = qcfg.get('single_query_names', None)
            single_query_name = qcfg.get('single_query_name', None)

            run_qnames = None
            if single_enable:
                # 允许既支持单个名称也支持列表
                if single_query_names is None and single_query_name is not None:
                    single_query_names = [single_query_name]
                if single_query_names:
                    run_qnames = [q for q in query_prior_poses_dict.keys() if q in set(single_query_names)]
                    if not run_qnames:
                        print("警告: single_query_* 未匹配到任何可用的 query 名称，将不执行该场景。")
                        return False
                else:
                    print("警告: 开启了 single_query_enable 但未提供 single_query_name(s)，将不执行该场景。")
                    return False
            else:
                run_qnames = list(query_prior_poses_dict.keys())

            ###循环开始
            for qname in run_qnames:
                q_prior_pose = query_prior_poses_dict[qname]
                q_intrinsic = query_intrinsics_dict[qname]
                osg_angle = osg_dict[qname]
                q_points = query_pixel_dict[qname]
                # print(osg_angle)
              
                [img_q_w, img_q_h, f_mm_q] = q_intrinsics_info[qname]

                #                 # generate seed
                import time
                s_time = time.time()
                reference_poses, reference_intrinsic, reference_yaw = generate_dji_seed(q_intrinsic, q_prior_pose, qname, osg_angle,
                                                                    seed_int_trans_x, seed_int_trans_y, seed_int_trans_z, seed_int_yaw,
                                                                    seed_num_trans_x, seed_num_trans_y, seed_num_trans_z, seed_num_yaw)
                # crop map
                # 如果debug模式，设置crop可视化保存路径
                crop_vis_save_path = None
                if debug:
                    # 使用配置管理器获取debug路径
                    crop_vis_save_path = config_manager.get_debug_path(scene, modality, method, map_type)
                    logger.info(f"🎨 Debug模式：生成crop区域可视化图像...")
                    logger.info(f"📁 保存路径: {crop_vis_save_path}")
                
                # 从格式化后的路径中提取实验名（例如从 "data/scene1_thermal/sat/roma_angle5/reference/rgb" 提取 "angle5"）
                experiment_name = 'default'
                try:
                    # ref_rgb_path 格式：data/{scene}_{modality}/{map_type}/{method}_angle5/reference/rgb
                    # 格式化后：data/scene1_thermal/sat/roma_angle5/reference/rgb
                    # 从路径中提取 {method}_angle5 部分，然后提取 angle5
                    path_parts = os.path.normpath(ref_rgb_path).split(os.sep)
                    # 查找 reference 目录的前一个目录（通常是 {method}_angle5）
                    if 'reference' in path_parts:
                        ref_idx = path_parts.index('reference')
                        if ref_idx > 0:
                            method_dir = path_parts[ref_idx - 1]  # reference 的上一级目录
                            if '_' in method_dir:
                                parts = method_dir.rsplit('_', 1)
                                if len(parts) == 2:
                                    # 提取下划线后的部分作为实验名（如 angle5）
                                    experiment_name = parts[1]
                except Exception:
                    pass  # 如果提取失败，使用默认值
                
                # 判断使用哪个函数
                if test_all_ratios:
                    
                    data = generate_ref_map(reference_intrinsic, reference_poses, area_minZ, dsm_data, dsm_trans, dom_data, 
                        ref_rgb_path, ref_depth_path, debug, crop_vis_save_path, 
                        test_ratio=None, use_bbox=True)  # 使用中心点+短边构建斜向正方形裁剪
                
                # 如果debug模式，跳过匹配和目标定位，直接进入下一张图
                if debug:
                    logger.info(f"✅ Debug模式：{qname} 的Crop可视化完成")
                    continue  # 直接进入下一张query图像的循环
                
                # 如果启用了test_all_ratios，需要分别对每个比例执行定位
                if test_all_ratios:
                    # data字典的key格式可能是：
                    # - {qname}_0_ratio_10, {qname}_0_ratio_20 等（比例格式）
                    # - {qname}_0_bbox（外接矩形格式）
                    # 因为seed_num=1，所以只有一个seed: {qname}_0
                    seed_name = f'{qname}_0'
                    
                    # 调试：打印data字典的内容
                    logger.info(f"  Debug: data字典包含的key: {list(data.keys())[:10]}...")
                    logger.info(f"  Debug: seed_name = {seed_name}")
                    
                    # 为每个比例执行定位
                    ratios_to_test = []
                    for key in data.keys():
                        if key.startswith(seed_name + '_ratio_'):
                            # 提取比例: seed_name_ratio_10 -> 10, seed_name_ratio_ori -> ori
                            ratio_str = key[len(seed_name)+7:]  # 去掉前缀和'_ratio_'后缀（7个字符）
                            
                            if ratio_str == 'ori':
                                # 原始比例
                                ratios_to_test.append(('ratio_ori', 'ori'))
                                logger.info(f"  Debug: 找到原始比例 ori, key={key}")
                            else:
                                try:
                                    ratio_val = int(ratio_str) / 100.0
                                    ratios_to_test.append((f'ratio_{int(ratio_val*100)}', ratio_val))
                                    logger.info(f"  Debug: 找到比例 {ratio_val*100:.0f}%, key={key}")
                                except ValueError:
                                    pass
                        elif key == seed_name + '_bbox':
                            # 支持_bbox格式（外接矩形）
                            ratios_to_test.append(('bbox', 'bbox'))
                            logger.info(f"  Debug: 找到外接矩形 bbox, key={key}")
                    
                    logger.info(f"  将对 {len(ratios_to_test)} 个不同的crop比例执行定位...")
                    
                    for ratio_label, ratio_val in ratios_to_test:
                        # 确定该比例对应的数据键
                        if ratio_val == 'bbox':
                            ratio_key = f'{seed_name}_bbox'
                            ratio_suffix = '_bbox'
                            display_label = "外接矩形 (BBox)"
                        else:
                            ratio_key = f'{seed_name}_ratio_{int(ratio_val*100)}'
                            ratio_suffix = f'_ratio_{int(ratio_val*100)}'
                            display_label = f"{ratio_label} (目标比例: {ratio_val*100:.0f}%)"
                        
                        if ratio_key not in data:
                            logger.warning(f"  跳过 {ratio_label}: 数据不存在 ({ratio_key})")
                            continue
                        
                        logger.info(f"  处理 {display_label}")
                        
                        # 为这个比例执行匹配和定位
                        localize_single_ratio(
                            qname, ratio_key, data, reference_intrinsic, reference_poses, reference_yaw,
                            q_intrinsic, q_prior_pose, query_path, ref_rgb_path, ref_depth_path, rot_save_path,
                            rot_flag, match_save_path, seed_num, model,
                            fr_result, fh_result, ray_tracing_flag, homography_flag,
                            homography_save_path, ray_trace_save_path,
                            dsm_data, dsm_trans, q_points, gt_dict, ratio_suffix
                        )
                        
                    continue  # 进入下一个query图像
                    
        # 如果debug模式，显示总结信息
        # if debug:
        #     logger.info(f"🎉 Debug模式：所有query图像的Crop可视化完成！")
        #     logger.info(f"📁 图像保存在: {crop_vis_save_path}")
        #     logger.info(f"🚪 已跳过所有匹配和定位步骤")
        
        logger.info(f"✅ 实验完成: {method.upper()} + {modality.upper()} + {scene.upper()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 实验失败: {method.upper()} + {modality.upper()} + {scene.upper()}")
        logger.error(f"错误信息: {str(e)}")
        return False

def main():
    """主函数：运行所有实验"""
    # 设置日志
    logger, log_file = setup_logging()
    logger.info("="*60)
    logger.info("开始执行实验程序")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)
    
    torch.cuda.memory._record_memory_history(max_entries=100000)


        # 初始化显存监控
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("🔍 显存监控已启动")
        print(f"{'='*60}")
        print("程序开始时的显存状态:")
        print_gpu_memory()
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计，开始新的测量
        torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        print("⚠️  CUDA不可用，无法监控显存")

    try:
        # 使用配置管理器获取实验配置
        experiment_configs = config_manager.get_experiment_configs()
        
        # 按map_type分组显示
        map_type_groups = {}
        for method, modality, scene, map_type in experiment_configs:
            if map_type not in map_type_groups:
                map_type_groups[map_type] = []
            map_type_groups[map_type].append((method, modality, scene, map_type))
        
        for map_type, experiments in map_type_groups.items():
            logger.info(f"\n准备执行 {map_type.upper()} 地图类型的实验...")
            logger.info(f"{map_type.upper()} 地图类型将执行 {len(experiments)} 个实验:")
            for i, (method, modality, scene, map_type) in enumerate(experiments, 1):
                logger.info(f"  {i:2d}. {method.upper()} + {modality.upper()} + {scene.upper()}")
        
        # === 预检查与确认（基于配置模板检查每个 scene/modality 的输入是否存在）===
        available_experiments = []
        skipped_experiments = []
        for method, modality, scene, map_type in experiment_configs:
            is_available, missing_files = config_manager.check_experiment_availability(method, modality, scene, map_type)
            if is_available:
                available_experiments.append((method, modality, scene, map_type))
            else:
                skipped_experiments.append(((method, modality, scene, map_type), missing_files))
        
        logger.info(f"\n候选实验总数: {len(experiment_configs)} | 可用: {len(available_experiments)} | 缺失: {len(skipped_experiments)}")
        
        # 原计划实验完整列表
        logger.info("\n原计划实验列表:")
        for i, (method, modality, scene, map_type) in enumerate(experiment_configs, 1):
            logger.info(f"  {i:2d}. {method.upper()} + {modality.upper()} + {scene.upper()} + {map_type.upper()}")
        
        # 可执行实验完整列表
        logger.info("\n可执行实验列表:")
        if available_experiments:
            for i, (method, modality, scene, map_type) in enumerate(available_experiments, 1):
                logger.info(f"  {i:2d}. {method.upper()} + {modality.upper()} + {scene.upper()} + {map_type.upper()}")
        else:
            logger.info("  （无可执行实验）")
        
        # 打印加载的数据统计信息（在决定运行前显示）
        if ALL_SCENE_IMAGES:
            print(f"\n{'='*60}")
            print("📊 加载的数据统计信息（全局）")
            print(f"{'='*60}")
            logger.info(f"\n{'='*60}")
            logger.info("📊 加载的数据统计信息（全局）")
            logger.info(f"{'='*60}")
            
            # 统计所有模态的数据（因为不同实验可能使用不同模态）
            modalities_in_config = set()
            for method, modality, scene, map_type in experiment_configs:
                modalities_in_config.add(modality)
            
            print(f"📁 数据路径: {SELECT_FOLDER}")
            logger.info(f"📁 数据路径: {SELECT_FOLDER}")
            
            for mod in sorted(modalities_in_config):
                total_images = 0
                scene_counts = {}
                
                for scene_name, images in ALL_SCENE_IMAGES.items():
                    filtered = [img for img in images if mod in img['img_path'].lower()]
                    count = len(filtered)
                    scene_counts[scene_name] = count
                    total_images += count
                
                if total_images > 0:
                    print(f"\n📋 模态 {mod.upper()} - 按场景统计:")
                    logger.info(f"\n📋 模态 {mod.upper()} - 按场景统计:")
                    for scene_name in sorted(scene_counts.keys()):
                        count = scene_counts[scene_name]
                        if count > 0:
                            print(f"   {scene_name}: {count} 张")
                            logger.info(f"   {scene_name}: {count} 张")
                    print(f"📈 {mod.upper()} 总计: {total_images} 张图像")
                    logger.info(f"📈 {mod.upper()} 总计: {total_images} 张图像")
            
            print(f"{'='*60}\n")
            logger.info(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("📊 加载的数据统计信息")
            print(f"{'='*60}")
            print(f"📁 数据路径: query_path (未使用 select 文件夹)")
            print(f"{'='*60}\n")
            logger.info(f"\n{'='*60}")
            logger.info("📊 加载的数据统计信息")
            logger.info(f"{'='*60}")
            logger.info(f"📁 数据路径: query_path (未使用 select 文件夹)")
            logger.info(f"{'='*60}\n")
        
        # 根据预检查结果决定最终执行列表
        final_experiments = experiment_configs
        if skipped_experiments:
            resp = input("\n是否仅运行“可执行实验列表”中的实验? [y=仅可用 / 其他=取消]: ").strip().lower()
            if resp == 'y':
                final_experiments = available_experiments
                logger.info(f"将运行 {len(final_experiments)} 个可用实验，跳过 {len(skipped_experiments)} 个缺失实验。")
            else:
                logger.info("已取消运行。")
                return
        else:
            logger.info("\n所有计划实验均可执行。")
            resp = input("\n是否现在运行所有计划实验? [y=运行 / 其他=取消]: ").strip().lower()
            if resp == 'y':
                final_experiments = experiment_configs
                logger.info(f"将运行 {len(final_experiments)} 个实验。")
            else:
                logger.info("已取消运行。")
                return
        
        logger.info(f"\n总共 {len(final_experiments)} 个实验配置")
        
        success_count = 0
        failed_experiments = []
        
        # 直接按最终列表顺序执行
        for i, (method, modality, scene, map_type) in enumerate(final_experiments, 1):
            logger.info(f"\n进度: {i}/{len(final_experiments)}")
            success = run_single_experiment(method, modality, scene, map_type)
            if success:
                success_count += 1
            else:
                failed_experiments.append((method, modality, scene, map_type))
        
        # 总结
        logger.info(f"\n{'='*60}")
        logger.info("实验总结:")
        logger.info(f"成功: {success_count}/{len(final_experiments)}")
        logger.info(f"失败: {len(failed_experiments)}")
        
        if failed_experiments:
            logger.info("\n失败的实验:")
            for method, modality, scene, map_type in failed_experiments:
                logger.info(f"  - {method.upper()} + {modality.upper()} + {scene.upper()} + {map_type.upper()}")
        
        logger.info(f"{'='*60}")
        logger.info(f"实验完成，详细日志已保存到: {log_file}")
    finally:
        # 显存监控总结
        if torch.cuda.is_available():
            print(f"\n{'='*60}")
            print("📊 程序结束时的显存状态:")
            print(f"{'='*60}")
            print_gpu_memory()
            
            # 保存显存快照（可选，用于详细分析）
            try:
                snapshot_path = f"memory_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_path)
                print(f"📸 显存快照已保存为: {snapshot_path}")
            except Exception as e:
                print(f"⚠️  保存快照失败: {e}")
            
            # 停止记录
            torch.cuda.memory._record_memory_history(enabled=None)
            print("✅ 显存监控已停止")

if __name__ == "__main__":
    main()