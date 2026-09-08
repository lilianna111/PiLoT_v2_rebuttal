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
from pixloc.crop.transform_colmap import transform_colmap_pose_intrinsic
from pixloc.crop.proj2map import generate_ref_map
from pixloc.crop.utils import read_DSM_config
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
        crop_padding=2, 
        debug=True
    )
    print("✅ process_map_crop Ended! data:", data)

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
    # ---------------- 裁剪线程（取代原渲染） ----------------        
    def rendering_worker(self):
        import torch
        print("✅ Rendering Worker Started! pid:", os.getpid())
        # === 1. 配置路径 ===
        ref_DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"
        ref_DOM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DOM.tif"
        ref_npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian.npy"
        if not os.path.exists(ref_DSM_path):
            raise FileNotFoundError(f"找不到文件: {ref_DSM_path}")
        # 输出路径指向 worker 的 output
        ref_rgb_path = self.outputs 
        ref_depth_path = self.outputs
        paths_pack = (ref_rgb_path, ref_depth_path)

        # === 2. 初始化地图 (只做一次) ===
        logging.info(f"Loading Map from {ref_DOM_path}...")
        map_data_pack = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
        logging.info("Map Loaded Successfully.")

        idx = 0
        fps_log_every = 0

        # === 3. 主循环：按位姿裁剪 ===
        while True:
            try:
                item = self.pose_q.get(timeout=1)
            except queue.Empty:
                if self.stop_evt.is_set():
                    break
                continue
            if item is None:
                break
            euler, trans = item  # 欧拉角 (pitch, roll, yaw) 与平移 (lon, lat, alt)
            euler_pitch, euler_roll, euler_yaw = euler[0], euler[1], euler[2]
            euler_crop = [euler_roll, euler_pitch, euler_yaw]
            pose_data = list(trans) + list(euler_crop)
            name = str(idx)

            t0 = time.perf_counter()
            try:
                _ = process_map_crop(
                    pose_data, 
                    name, 
                    map_data_pack, 
                    paths_pack
                )
            except Exception as exc:
                logging.error(f"process_map_crop 失败 name={name}: {exc}")
                continue

            img_file = os.path.join(ref_rgb_path, f'{name}_dom.png')
            npy_file = os.path.join(ref_depth_path, f'{name}_dsm.npy')
            if (not os.path.exists(img_file)) or (not os.path.exists(npy_file)):
                logging.error(f"裁剪结果缺失: {img_file} or {npy_file}")
                continue

            color = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if color is None:
                logging.error(f"读取裁剪 RGB 失败: {img_file}")
                continue
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)    #作用：将BGR格式转换为RGB格式

            points3d = np.load(npy_file)
            valid_mask = np.isfinite(points3d).all(axis=-1) & (points3d[..., 2] > 0)    #valid_mask: 有效掩码，用于标识哪些点是有效的
            if not np.any(valid_mask):
                logging.error(f"裁剪结果无有效3D点: {npy_file}")
                continue

            # --- 对齐参考尺寸到渲染相机分辨率（保证与 query 一致） ---
            target_w, target_h = int(self.render_camera_osg[0]), int(self.render_camera_osg[1])
            color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            pts_resized = []
            for ch in range(3):
                pts_resized.append(cv2.resize(points3d[..., ch], (target_w, target_h), interpolation=cv2.INTER_LINEAR))
            points3d = np.stack(pts_resized, axis=-1).astype(np.float32)
            resize_save_path = os.path.join(self.outputs, f'{name}_1_resized.png')
            # 注意：内存中是RGB，OpenCV保存需要BGR，否则颜色错误
            cv2.imwrite(resize_save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            logging.info(f"Saved resized image: {resize_save_path}")

            # padding 到 16 的倍数，使用复制边界避免 0 值；同步处理有效掩码
            mask = valid_mask.astype(np.uint8)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if self.padding:
                pad_h = (16 - target_h % 16) % 16
                pad_w = (16 - target_w % 16) % 16
                if pad_h or pad_w:
                    color = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                    points3d_ch = []
                    for ch in range(3):
                        points3d_ch.append(cv2.copyMakeBorder(points3d[..., ch], 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE))
                    points3d = np.stack(points3d_ch, axis=-1)
            pad_path = "/home/amax/Documents/datasets/output/test"
            padded_save_path = os.path.join(pad_path, f'{name}.png')
            cv2.imwrite(padded_save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            logging.info(f"Saved padded image: {padded_save_path}")
            mask = mask.astype(bool)

            # 预先采样 3D 点（避免后端再随机），只保留有效点
            points_valid = points3d[mask]
            if points_valid.shape[0] == 0:
                logging.error(f"裁剪结果无有效3D点(经过padding/mask后): {npy_file}")
                continue
            sample_num = min(500, points_valid.shape[0])
            sel = np.random.choice(points_valid.shape[0], sample_num, replace=False)
            points_sampled = points_valid[sel]
            coords_all = np.argwhere(mask)  # (N,2) -> row, col on padded image
            coords_sampled = coords_all[sel]

            # 保存调试信息：采样点的像素位置（padding 后）与图像尺寸
            debug_npz = os.path.join(self.outputs, f'{name}_debug_samples.npz')
            np.savez_compressed(
                debug_npz,
                coords_sampled=coords_sampled,
                img_shape=color.shape[:2],
                sample_num=sample_num
            )
            logging.info(f"Saved sampled coords: {debug_npz}, head={coords_sampled[:5].tolist()}")

            t_render = (time.perf_counter() - t0) * 1e3  # ms
            fps_log_every += 1
            if fps_log_every % 30 == 0:
                logging.info("crop: %.2f ms", t_render)

            try:
                # 将裁剪得到的 RGB 与采样后的 3D 点（EPSG:4547）及原始位姿送入队列
                self.task_q.put((color, points_sampled, euler, trans), timeout=1)
            except queue.Full:
                break
            idx += 1

        self.stop_evt.set()
        self.task_q.put(None)
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Render/Crop process done')
    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        from pixloc.localization import RenderLocalizer, SimpleTracker
        localizer = RenderLocalizer(self.conf)
        results = []   # 存盘缓冲
        fps_log_every = 30
        flag = True
        last_euler, last_trans = None, None
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
            
            # 控制传入优化的3D点规模，避免显存爆
            max_pts_opt = 2000
            if P3d.shape[0] > max_pts_opt:
                perm = torch.randperm(P3d.shape[0], device=P3d.device)
                keep = perm[:max_pts_opt]
                P3d = P3d[keep]

            # t2 = time.time()
            # print('反投影耗时：', t2-t1)
            print("P3d: ", P3d)
            t0 = time.time()
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
                image_query = img_tensor
            )
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
            print(f"Localization Thread calculated frames: {idx} | Pitch, Roll, Yaw: {ret['euler_angles'].tolist()} | Longitude, Latitude, Altitude: {ret['translation']}")
            # 4) 暂存结果
            qname = os.path.basename(img_path)
            results.append(f"{qname} {' '.join(map(str, ret['translation']))} "
                           f"{' '.join(map(str, [ret['euler_angles'][1], ret['euler_angles'][0], ret['euler_angles'][2]]))}")

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
        default="configs/default.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="default_experiment",
        help="实验名称，用于日志标记"
    )

    parser.add_argument(
        "--init_euler",
        type=str,
        default="[0.0, 0.0, 0.0]",
        help="初始欧拉角（格式如：[25.0, 0.0, 314.9993]）"
    )

    parser.add_argument(
        "--init_trans",
        type=str,
        default="[0.0, 0.0, 0.0]",
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
    






