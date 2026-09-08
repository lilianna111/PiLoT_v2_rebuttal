import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)     # Linux 默认 fork → 改成 spawn
from this import d
import threading
from pixloc.utils.data import Paths
import os
import queue
import glob
import cv2
import shutil
import argparse
import ast
import numpy as np
from tqdm import tqdm
from pprint import pformat
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
from pixloc.pixlib.datasets.view import read_image_list
import time
import yaml
import copy
import logging
import torch
from multiprocessing import Process, Queue, Event
from pixloc.crop.utils import read_DSM_config
from PIL import Image
from pixloc.crop.transform_colmap import transform_colmap_pose_intrinsic
from pixloc.crop.proj2map import generate_ref_map
import torch.nn.functional as F
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S")
'''
1. config init rot, init translation, datapath
'''
def get_init(pose_file):
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # if '14000' in parts[0]:
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                origin = WGS84_to_ECEF(translation)
                break
    return euler_angles, translation, origin
def load_poses(pose_file, origin = None):
    """Load poses from the pose file."""
    pose_dict = {}
    translation_list = []
    euler_angles_list = []
    name_list = []
    init_euler = None
    init_trans = None
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                if '_' not in parts[0]:
                    name = parts[0][:-4] +'_0.png'
                else:
                    name = parts[0]
                name_list.append(name)
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
                pose_dict[name] = {}
                pose_dict[name]['T_w2c_4x4'] = copy.deepcopy(T_in_ECEF_c2w)
                T_in_ECEF_c2w[:3, 1] = -T_in_ECEF_c2w[:3, 1]  # Y轴取反，投影后二维原点在左上角
                T_in_ECEF_c2w[:3, 2] = -T_in_ECEF_c2w[:3, 2]  # Z轴取反
                T_in_ECEF_c2w[:3, 3] -= origin  # t_c2w - origin
                render_T_w2c = np.eye(4)
                render_T_w2c[:3, :3] = T_in_ECEF_c2w[:3, :3].T
                render_T_w2c[:3, 3] = -T_in_ECEF_c2w[:3, :3].T @ T_in_ECEF_c2w[:3, 3]
                
                
                pose_dict[name]['euler'] = [pitch, roll, yaw]
                pose_dict[name]['trans'] = [lon, lat, alt]
                
                
                render_T_w2c = Pose.from_Rt(render_T_w2c[:3, :3], render_T_w2c[:3, 3])
                pose_dict[name]['T_w2c'] = render_T_w2c.to_flat()
    return pose_dict
def world_to_depth(points_3d, T_w2c):
    """
    利用现有的 torch 库实现坐标转换
    Args:
        points_3d: [3, H, W] 世界坐标点云 (CGCS2000)
        T_w2c: [4, 4] 世界到相机位姿矩阵
    Returns:
        depth: [1, H, W] 深度图
    """
    # 1. 确保都在同一个设备上 (CPU/GPU)
    device = points_3d.device
    if not isinstance(T_w2c, torch.Tensor):
        T_w2c = torch.tensor(T_w2c, device=device, dtype=torch.float32)
    else:
        T_w2c = T_w2c.to(device)

    # 2. 提取旋转(R)和平移(t)
    # T_w2c = [R, t]
    R = T_w2c[:3, :3]  # [3, 3]
    t = T_w2c[:3, 3]   # [3]

    # 3. 核心计算: P_cam = R * P_world + t
    # 我们只需要 Z 分量 (深度)，所以只取矩阵的第3行
    # R[2, :] 是 Z 轴的旋转向量
    # t[2]    是 Z 轴的平移
    
    # [1, 3] @ [3, H, W] -> [1, H, W] using einsum for clarity
    # 'j'是3维向量维度，'hw'是图像维度
    z_val = torch.einsum('j, jhw -> hw', R[2], points_3d) + t[2]
    
    # 4. 增加一个维度变为 [1, H, W] 以匹配深度图格式
    # depth = z_val.unsqueeze(0)
    depth = z_val
    
    # 5. 处理无效深度（可选：把背后的点设为0）
    depth[depth < 0] = 0
    
    return depth
def process_map_crop(pose_data, name, 
                     map_data_pack,      # 包含 dsm_data, dom_data 等的大礼包
                     paths_pack):        # 包含输出路径
    """
    单帧处理函数：只负责计算和裁剪，不负责加载大地图
    """

    # 1. 解包地图数据 (这些数据在内存里，很快)
    geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = map_data_pack
    ref_rgb_path, ref_depth_path = paths_pack

    # 2. 坐标转换 (Read prior pose and intrinsic)
    query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
    
    # 3. 执行裁剪 (Crop map)
    # 这一步会把图片保存到硬盘
    data = generate_ref_map(
        query_intrinsics_dict, 
        query_prior_poses_dict, 
        name, 
        area_minZ, 
        dsm_data, 
        dsm_trans, 
        dom_data, 
        ref_rgb_path, 
        ref_depth_path,
        crop_padding=10, 
        debug=True
    )
    print("✅ process_map_crop Ended! data:", data)
    # 我们需要返回位姿信息，以便 worker 传给下游
    return data, osg_dict, query_prior_poses_dict
def load_dsm_to_gpu(dsm_path, device):
    import rasterio
    import numpy as np
    import torch
    
    print(f"🚀 [GPU Loader] Loading DSM from: {dsm_path}")
    try:
        with rasterio.open(dsm_path) as src:
            dsm_data = src.read(1)
            transform = src.transform
            
            # === 🚑 核心修复：彻底清洗数据 ===
            # 1. 处理 NaN (非常重要！)
            if np.isnan(dsm_data).any():
                print("⚠️ Found NaN values in DSM, replacing with 0.")
                dsm_data = np.nan_to_num(dsm_data, nan=0.0)
            
            # 2. 处理 NoData
            nodata = src.nodata
            if nodata is not None:
                dsm_data[dsm_data == nodata] = 0
            
            # 3. 处理极小值 (有些格式用负无穷表示无效)
            dsm_data[dsm_data < -1000] = 0
            
            # 4. 处理极大值 (防止溢出)
            dsm_data[dsm_data > 10000] = 0
            # =================================
            
            dsm_tensor = torch.from_numpy(dsm_data).float()
            dsm_tensor = dsm_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            dsm_meta = {
                'min_x': transform.c, 'max_y': transform.f,
                'res_x': transform.a, 'res_y': transform.e,
                'height': dsm_data.shape[0], 'width': dsm_data.shape[1]
            }
            return dsm_tensor, dsm_meta
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None, None

class DualProcessTask:        
    def __init__(self, config, init_euler = None, init_trans = None, name = None):
        # 用 multiprocessing 队列/事件
        self.task_q   = Queue(maxsize=2)     # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)     # 定位 → 渲染
        self.stop_evt = Event()
        self.render_config = config["render_config"]
        default_confs = config["default_confs"] 
        self.conf = default_confs['from_render_test'] # from_render_test
        # conf初始化
        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name = default_confs['dataset_name']
        # self.euler_angles, self.translation = self.render_config['init_rot'], self.render_config['init_trans']
        
        if name is not None:
            dataset_name = name
            output_name = name
        # if init_euler is not None:
        #     self.render_config['init_rot'], self.render_config['init_trans'] = init_euler, init_trans
        #     self.euler_angles = init_euler
        #     self.translation = init_trans
        self.refine_conf = default_confs['refine']
        self.mul = self.refine_conf['mul']
        # self.estimated_pose = os.path.join(folder_path, 'estimation', 'FPVLoc@'+output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        output_folder = "/home/amax/Documents/datasets/outputs/FPVLoc"
        # output_folder = "outputs"
        self.outputs = os.path.join(output_folder, output_name)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)
        else:
            shutil.rmtree(self.outputs)
            os.makedirs(self.outputs)
        self.estimated_pose = os.path.join(output_folder, output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        
        self.gt_pose = os.path.join(folder_path, 'poses', dataset_name +'.txt') #
        
        self.last_frame_info = {}
        self.last_frame_info['observations'] = []
        self.last_frame_info['refine_conf'] = self.refine_conf
        
        print(f'conf:\n{pformat(self.conf)}')
        # 初始化先验位姿和内参
        self.name_q = None
        # camera
        self.query_resize_ratio = default_confs['cam_query']['width'] / default_confs['cam_query']['max_size']
        fx, fy, cx, cy = default_confs['cam_query']['params'] 
        w, h = default_confs['cam_query']['width'], default_confs['cam_query']['height']
        
        raw_query_camera = np.array([w, h, cx, cy, fx, fy])
        self.render_camera_osg = raw_query_camera / self.query_resize_ratio
        
        default_confs['cam_query']['params']  = np.array(default_confs['cam_query']['params']) / self.query_resize_ratio
        default_confs['cam_query']['width'], default_confs['cam_query']['height'] = default_confs['cam_query']['width'] / self.query_resize_ratio, default_confs['cam_query']['height']/self.query_resize_ratio
        cam_query = default_confs["cam_query"]
        self.query_camera = Camera.from_colmap(cam_query) #! 2.
        
        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list = glob.glob(img_path + "/*.png") + glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        self.img_list = sorted(self.img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.img_list = self.img_list[:1]
        self.query_list = read_image_list(self.img_list, scale = self.query_resize_ratio, distortion=cam_query['distortion'], query_camera = raw_query_camera)
        
        self.dd = None
        self.name_r = None

        self.render_camera = generate_render_camera(self.render_camera_osg).float()
        self.render_config['render_camera'] = self.render_camera_osg 

        # 是否padding, num init  pose
        self.num_init_pose = default_confs['num_init_pose']
        self.padding = default_confs['padding']
        self.euler_angles, self.translation, self.origin = get_init(self.gt_pose)
        self.render_config['init_rot'], self.render_config['init_trans'] = self.euler_angles, self.translation
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)
        
        self.device = 'cuda'
        self.origin = torch.tensor(self.origin, device=self.device)
        self.query_camera, self.render_camera = self.query_camera.to(self.device), self.render_camera.to(self.device)

        self.pose_q.put_nowait((self.euler_angles, self.translation))
        self.pose_q.put_nowait((self.euler_angles, self.translation))
    # ---------------- crop线程 ----------------        
    def rendering_worker(self):
        import torch
        print("✅ Rendering Worker Started! pid:", os.getpid())
        # === 1. 配置路径 ===
        ref_DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"
        ref_DOM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DOM.tif"
        ref_npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.npy"
        # 检查路径是否存在，防患于未然
        if not os.path.exists(ref_DSM_path):
            raise FileNotFoundError(f"找不到文件: {ref_DSM_path}")
        
        # 输出路径指向 worker 的 output
        ref_rgb_path = self.outputs 
        ref_depth_path = self.outputs
        
        # 打包路径方便传递
        paths_pack = (ref_rgb_path, ref_depth_path)

        # === 2. 初始化地图 (只做一次！！！) ===
        logging.info(f"Loading Map from {ref_DOM_path}...")
        # map_data_pack 包含了所有重型数据
        map_data_pack = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
        logging.info("Map Loaded Successfully.")

        idx = 0
        fps_log_every = 0

        while True:
            try:
                # 尝试从 pose_q 队列中获取一个位姿数据
                # timeout=1 表示如果队列空了，最多等1秒，不要死等
                item = self.pose_q.get(timeout=1)     # <—— 带超时
            except queue.Empty:
                # 如果等了1秒还是空的，检查是否收到了“停止信号”(stop_evt)
                if self.stop_evt.is_set():            # 有人喊停就走
                    break # 喊停了，直接退出循环
                continue # 没喊停，就继续下一轮尝试获取
            if item is None:                          # 对称哨兵,如果收到 None，说明上游明确告知没有更多数据了
                break
            euler, trans = item  # 解包数据：欧拉角 (旋转) 和 平移向量
            euler_pitch, euler_roll, euler_yaw = euler[0], euler[1], euler[2]
            euler_crop = [euler_roll, euler_pitch, euler_yaw]
             
            # 强制转成扁平的 6 元素列表
            pose_data = list(trans) + list(euler_crop)
            
            logging.info(f"Format fixed: (Euler, Trans) -> {pose_data}")
           
            name = str(idx)

            
            t0 = time.perf_counter()

            # === 4. 调用处理函数 ===
            # 注意：我们传入已经加载好的 map_data_pack
            try:
                _, osg_dict, query_prior_poses = process_map_crop(
                    pose_data, 
                    name, 
                    map_data_pack, 
                    paths_pack
                )
                print("✅ Cropping Worker Started! osg_dict:", osg_dict)
                # 因为 generate_ref_map 是写盘的，我们需要读回来
                img_file = os.path.join(ref_rgb_path, f'{name}_dom.png')
                npy_file = os.path.join(ref_depth_path, f'{name}_dsm.npy')

                # 读取 RGB -> Tensor [3, H, W] Normalized
                pil_img = Image.open(img_file)
                color_np = np.array(pil_img)
                # color_tensor = torch.from_numpy(color_np).float().permute(2, 0, 1) / 255.0
                # color_tensor = color_np
                
                # 读取 Depth/3D -> Tensor [3, H, W]
                points_3d_np = np.load(npy_file)
                points_3d_tensor = torch.from_numpy(points_3d_np).float().permute(2, 0, 1)
                pose_tensor = torch.from_numpy(query_prior_poses).float()
                depth_map_tensor = world_to_depth(points_3d_tensor, pose_tensor)

                t_crop = (time.perf_counter() - t0) * 1e3
                print("✅ Cropping Worker Started! ")
                # print("depth_map_tensor.ndim", depth_map_tensor.ndim)
                TARGET_SIZE = (512, 512)  # 下游 zero_pad 期望的大小

                # 1. 处理 Color (NumPy HWC)
                # 使用 cv2.resize, 注意 cv2 是 (width, height)
                # 输入 color_np: (1548, 1582, 3) -> 输出 (512, 512, 3)
                color_np_resized = cv2.resize(color_np, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

                # 2. 处理 Depth (Tensor 1HW)
                # 输入 depth_map_tensor: (1, 1548, 1582) -> 输出 (1, 512, 512)
                # 增加一个 batch 维度用于 interpolate: (1, 1, H, W)
                depth_batch = depth_map_tensor.unsqueeze(0).unsqueeze(0)
                
                # 使用双线性插值或最近邻插值 (对于坐标值，bilinear 通常更平滑，但 nearest 更保真)
                # 这里用 bilinear 没问题，因为存的是 XYZ 坐标，插值就是求中间点的坐标
                depth_resized = F.interpolate(depth_batch, size=TARGET_SIZE, mode='bilinear', align_corners=False)
                
                # 变回 (H, W) 或 (1, H, W)
                # 刚才讨论过，下游 zero_pad 期望 depth 是 2D (H, W)，所以这里顺手 squeeze 掉
                depth_map_final = depth_resized.squeeze(0).squeeze(0)  # -> (512, 512)
                
                # === 6. 发送给定位 ===
                # 这里的 query_prior_poses 是 CGCS2000 下的 T_w2c 矩阵
                try:
                    self.task_q.put((color_np_resized,depth_map_final, euler, trans), timeout=1)
                except queue.Full:
                    break
            except Exception as e:
                logging.error(f"Crop Error at frame {idx}: {e}")
                # 出错不要崩，跳过这一帧
                continue

            # 日志
            fps_log_every += 1
            idx += 1
            if fps_log_every % 30 == 0:
                logging.info(f"Crop processing: {t_crop:.2f} ms")

        # === 7. 清理退出 ===
        self.stop_evt.set()
        self.task_q.put(None)
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Cropping Worker done')
    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        from pixloc.localization import RenderLocalizer, SimpleTracker
        localizer = RenderLocalizer(self.conf)
        results = []   # 存盘缓冲
        fps_log_every = 30
        flag = True
        last_euler, last_trans = None, None
        dsm_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载一次，永久使用
        self.dsm_tensor, self.dsm_meta = load_dsm_to_gpu(dsm_path, device)
        gt_depth_list = []
        txt_path = "/home/amax/Documents/datasets/depth/DJI_20250612194622_0018_V.txt" 
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # parts[0] 是文件名, parts[1] 是深度值
                        val = float(parts[1])
                        gt_depth_list.append(val)
            logging.info(f"Loaded {len(gt_depth_list)} depths from txt.")
        else:
            logging.error("GT Depth txt file not found!")
        
        for idx, (img_path, img_tensor) in enumerate(zip(self.img_list, self.query_list)):
            # 1) 拿到渲染结果（阻塞）
            item = self.task_q.get()
            if item is None:        # 渲染端提前喊停
                break
            color, depth, render_euler, render_trans = item
            # 2) 反投影
            # t1 = time.time()
            if last_trans is None:
                last_euler, last_trans = render_euler, render_trans
            P3d, T_w2c_mod, T_init, dd = self.back_project(depth, render_euler, render_trans, last_euler, last_trans)
            # P3d, T_w2c_mod, T_init, dd = self.back_project(depth, render_euler, render_trans, render_euler, render_trans)
            
            # t2 = time.time()
            # print('反投影耗时：', t2-t1)
            t0 = time.time()
            current_gt_depth = gt_depth_list[idx]
            # 2) 调用定位
            ret = localizer.run_query(
                img_path, self.query_camera, self.render_camera,
                color,  # render_frame
                query_T = T_init,
                render_T= T_w2c_mod,
                Points_3D_ECEF = P3d,
                query_resize_ratio = self.query_resize_ratio,
                dd = dd,
                gt_pose_dict=self.gt_pose_dict,
                last_frame_info = self.last_frame_info,
                image_query = img_tensor,
                gt_depth = current_gt_depth,
                dsm_tensor=self.dsm_tensor, 
                dsm_meta=self.dsm_meta
            )
            qname = os.path.basename(img_path)
            results.append(f"{qname} {' '.join(map(str, ret['translation']))} "
                           f"{' '.join(map(str, [ret['euler_angles'][1], ret['euler_angles'][0], ret['euler_angles'][2]]))}")
            if idx > len(self.gt_pose_dict)-3:
                ret['euler_angles']=copy.deepcopy(self.gt_pose_dict[f"{idx}_0.png"]['euler'])
                ret['translation'] = copy.deepcopy(self.gt_pose_dict[f"{idx}_0.png"]['trans'])
            else:
                ret['euler_angles']=copy.deepcopy(self.gt_pose_dict[f"{idx+2}_0.png"]['euler'])
                ret['translation'] = copy.deepcopy(self.gt_pose_dict[f"{idx+2}_0.png"]['trans'])

            last_euler, last_trans = ret['euler_angles'], ret['translation'] 

            # self.last_frame_info['observations'] = ret['observations']
            # self.last_frame_info['euler_angles'] = ret['euler_angles']
            # self.last_frame_info['translation'] = ret['translation']
            # cv2.imwrite(f'{self.outputs}/{idx}_query.png', img_tensor)
            if idx % fps_log_every == 0:
                logging.info("loc %.2f ms", (time.time()-t0)*1e3)
            
            # 3) 把得到的新姿态塞回 pose_q，供下一帧渲染
            if idx < len(self.img_list)-1:
                self.pose_q.put((ret['euler_angles'], ret['translation']))
            # print(f"Localization Thread calculated frames: {idx} | Pitch, Roll, Yaw: {ret['euler_angles'].tolist()} | Longitude, Latitude, Altitude: {ret['translation']}")
            # 4) 暂存结果
            # qname = os.path.basename(img_path)
            # results.append(f"{qname} {' '.join(map(str, ret['translation']))} "
            #                f"{' '.join(map(str, [ret['euler_angles'][1], ret['euler_angles'][0], ret['euler_angles'][2]]))}")

            if self.stop_evt.is_set():
                break
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))
        
        # self.flush_pose_and_send_sentinel()
        # 收尾：给渲染端塞哨兵 + 关队列 + 事件
        self.pose_q.put(None)
        self.stop_evt.set()
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Localization process done')
    def flush_pose_and_send_sentinel(self):
        # 1) 清空 pose_q 中所有旧条目
        try:
            while True:
                self.pose_q.get_nowait()
        except queue.Empty:
            pass

        # 2) 然后往里放一个 None，作为哨兵
        self.pose_q.put(None)
    def start_threads(self):
        # 启动定位线程 
        self.localization_thread_instance = threading.Thread(target=self.localization_thread)
        self.localization_thread_instance.start()

        # 启动渲染线程
        self.rendering_thread_instance = threading.Thread(target=self.rendering_thread)
        self.rendering_thread_instance.start()

    def stop_threads(self):
        self.localization_thread_instance.join()
        self.rendering_thread_instance.join()

    def back_project(self, depth_frame, euler_angles, translation, query_euler_angles, query_translation, num_samples = 500, device = 'cuda'):
        # 
        if not torch.is_tensor(depth_frame):
            depth = torch.as_tensor(depth_frame, device=device)
        else:
            depth = depth_frame.to(device)

        # 2) 把 T_render_in_ECEF_c2w 也转为 GPU tensor
        T_render_in_ECEF_c2w = torch.as_tensor(
            euler_angles_to_matrix_ECEF(euler_angles, translation),
            device=device, dtype=torch.float32
        )  # shape (4,4) 或 (3,4)

        # 3) 生成随机像素坐标也用 torch
        H, W = int(self.render_camera_osg[1]), int(self.render_camera_osg[0])
        # [num_samples]
        ys = torch.randint(0, H, size=(num_samples,), device=device)
        xs = torch.randint(0, W, size=(num_samples,), device=device)
        points2d = torch.stack((xs, ys), dim=1)  # (N,2)
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v3(points2d, depth, T_render_in_ECEF_c2w, self.render_camera,  euler_angles, translation,  query_euler_angles, query_translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul)
        # if dd is not None:
        #     self.dd = 
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd

    # 假设的计算位姿函数
    def calculate_pose(self, render_frame, points_3d):
        # 根据渲染图和3D点计算位姿
        return "Pose"

    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        evaluate(self.estimated_pose, self.gt_pose)
    def run(self):
        ctx = mp.get_context("spawn")        # 保持 spawn
        p_render = ctx.Process(target=self.rendering_worker, daemon=True)
        p_loc    = ctx.Process(target=self.localization_worker, daemon=True)

        p_render.start(); p_loc.start()
        p_loc.join()                         # 先等定位结束
        p_render.join(5)                    # 最多等 30 s
        if p_render.is_alive():              # 兜底：仍卡住就强退
            p_render.terminate()
            p_render.join()
def parse_args():
    parser = argparse.ArgumentParser(description="你的程序说明")

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/feicuiwan_m4t.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="DJI_20251221131951_0001_V_1",
        help="实验名称，用于日志标记"
    )

    parser.add_argument(
        "--init_euler",
        type=str,
        default="[0.0, 0.0, 137.1]",
        help="初始欧拉角（格式如：[roll, pitch, yaw]）"
    )

    parser.add_argument(
        "--init_trans",
        type=str,
        default="[113.001091, 28.295298, 176.064 ]",
        help="初始平移（格式如：[x, y, z]）"
    )

    args = parser.parse_args()


    return args
# 主程序入口
if __name__ == "__main__":
    args = parse_args()
    init_euler = args.init_euler
    init_trans = args.init_trans
    name = args.name
    config_file = args.config
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_config = config["render_config"]
    default_confs = config["default_confs"]
    default_paths = config["default_paths"]
    dual_task = DualProcessTask(config,  name=name)
    
    dual_task.run()
    dual_task.eval()
    






