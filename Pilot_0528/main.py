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
import csv
import json
import sys
import numpy as np
from tqdm import tqdm
from pprint import pformat
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate, euler_angles_to_matrix_ECEF_w2c
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import ECEF_to_WGS84, euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
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
    level=logging.WARNING,
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


def add_translation_prior_ecef_xy(translation, dx_m, dy_m):
    """Add a direct ECEF (dX, dY, 0) offset to a WGS84 position."""
    base_ecef = np.asarray(WGS84_to_ECEF(translation), dtype=np.float64)
    delta_ecef = np.array([dx_m, dy_m, 0.0], dtype=np.float64)
    prior_ecef = base_ecef + delta_ecef
    return ECEF_to_WGS84(prior_ecef), prior_ecef, delta_ecef


def translation_error_ecef(reference_translation, estimated_translation):
    reference_ecef = np.asarray(WGS84_to_ECEF(reference_translation), dtype=np.float64)
    estimated_ecef = np.asarray(WGS84_to_ECEF(estimated_translation), dtype=np.float64)
    return estimated_ecef - reference_ecef


TRANSLATION_FRAME_FIELDS = [
    "frame", "image", "injected_dx_m", "injected_dy_m",
    "gt_ecef_x", "gt_ecef_y", "gt_ecef_z",
    "crop_prior_ecef_x", "crop_prior_ecef_y", "crop_prior_ecef_z",
    "pred_ecef_x", "pred_ecef_y", "pred_ecef_z",
    "prior_error_x_m", "prior_error_y_m", "prior_error_z_m",
    "prior_error_xy_m", "prior_error_3d_m",
    "final_error_x_m", "final_error_y_m", "final_error_z_m",
    "final_error_xy_m", "final_error_3d_m",
    "final_yaw_error_deg", "final_rotation_error_deg",
    "optimization_success", "failure_reason",
    "crop_ms", "localization_ms", "back_project_ms",
    "feature_ms", "optimizer_ms", "postprocess_ms",
]


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
def process_map_crop(ref_DSM_path, pose_data, ref_npy_path, name, map_data_pack, paths_pack,
                     ray_area, ray_area_minZ, locator=None):        # 包含输出路径
    """
    单帧处理函数：只负责计算和裁剪，不负责加载大地图
    """

    # 1. 解包地图数据 (这些数据在内存里，很快)
    geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = map_data_pack
    ref_rgb_path, ref_depth_path = paths_pack

    # 2. 坐标转换 (Read prior pose and intrinsic)
    t_pose0 = time.perf_counter()
    query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
    t_pose1 = time.perf_counter()
    
    # 3. 执行裁剪 (Crop map)
    # 这一步会把图片保存到硬盘
    data = generate_ref_map(
        ref_DSM_path, pose_data,
        ref_npy_path, geotransform,
        query_intrinsics_dict, 
        query_prior_poses_dict, 
        name, 
        area_minZ, 
        dsm_data, 
        dsm_trans, 
        dom_data, 
        ref_rgb_path, 
        ref_depth_path,
        ray_area,
        ray_area_minZ,
        crop_padding=0, 
        debug=False,
        save_to_disk=False,
        locator=locator,
    )
    timings = data.get('timings', {}).copy()
    timings['pose_transform_s'] = t_pose1 - t_pose0
    timings['crop_total_s'] = timings.get('total_s', 0.0) + timings['pose_transform_s']
    data['timings'] = timings
    return data

class DualProcessTask:        
    def __init__(self, config, init_euler = None, init_trans = None, name = None, sample_num=300,
                 output_folder=None, translation_sensitivity=False,
                 prior_dx_m=0.0, prior_dy_m=0.0, random_seed=0):
        # 用 multiprocessing 队列/事件
        self.task_q   = Queue(maxsize=2)     # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)     # 定位 → 渲染
        self.stop_evt = Event()
        self.localization_failure_evt = Event()
        self.render_config = config["render_config"]
        default_confs = config["default_confs"] 
        self.conf = default_confs['from_render_test'] # from_render_test
        self.sample_num = sample_num
        self.translation_sensitivity = bool(translation_sensitivity)
        self.prior_dx_m = float(prior_dx_m)
        self.prior_dy_m = float(prior_dy_m)
        self.random_seed = int(random_seed)
        if not np.isfinite([self.prior_dx_m, self.prior_dy_m]).all():
            raise ValueError("prior_dx_m and prior_dy_m must be finite")
        if not self.translation_sensitivity and (self.prior_dx_m != 0 or self.prior_dy_m != 0):
            raise ValueError("non-zero ECEF translation requires --translation_sensitivity")
        # conf初始化
        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name = default_confs['dataset_name']
        # self.euler_angles, self.translation = self.render_config['init_rot'], self.render_config['init_trans']
        
        if name is not None:
            dataset_name = name
            output_name = name
        self.dataset_name = dataset_name
        # if init_euler is not None:
        #     self.render_config['init_rot'], self.render_config['init_trans'] = init_euler, init_trans
        #     self.euler_angles = init_euler
        #     self.translation = init_trans
        self.refine_conf = default_confs['refine']
        self.mul = self.refine_conf['mul']
        # self.estimated_pose = os.path.join(folder_path, 'estimation', 'FPVLoc@'+output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        # output_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth"
        # output_folder = "/media/amax/PS2000/ral/crop_1"
        if output_folder is None:
            output_folder = "/media/amax/PortableSSD/0908/angle"
        # output_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth_weixing"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pre_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/output/test"
        depth_txt_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/depth_1"
        angle_txt_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/angle"
        # output_folder = "outputs"
        self.outputs = os.path.join(output_folder, output_name)
        self.pre_path = os.path.join(pre_path, output_name)
        self.depth_txt_path = os.path.join(depth_txt_folder, output_name + ".txt")
        self.angle_txt_path = os.path.join(angle_txt_folder, output_name + ".txt")
        experiment_outputs = [
            self.outputs,
            os.path.join(output_folder, output_name + ".txt"),
            os.path.join(output_folder, output_name + "_frames.csv"),
            os.path.join(output_folder, output_name + "_config.json"),
            os.path.join(output_folder, output_name + "_status.json"),
        ]
        existing_outputs = [path for path in experiment_outputs if os.path.exists(path)]
        if self.translation_sensitivity and existing_outputs:
            raise FileExistsError(f"Experiment output already exists: {existing_outputs[0]}")
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)
        else:
            shutil.rmtree(self.outputs)
            os.makedirs(self.outputs)
        if not os.path.exists(self.pre_path):
            os.makedirs(self.pre_path)
        else:
            shutil.rmtree(self.pre_path)
            os.makedirs(self.pre_path)
            
        self.estimated_pose = os.path.join(output_folder, output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        
        self.gt_pose = os.path.join(folder_path, 'poses', dataset_name +'.txt') #
        
        self.last_frame_info = {}
        self.last_frame_info['observations'] = []
        self.last_frame_info['refine_conf'] = self.refine_conf
        self.gt_reset_report = os.path.join(output_folder, dataset_name + "_reset.txt")
        self.sample_num_report = os.path.join(output_folder, dataset_name + f"_{self.sample_num}.txt")
        self.translation_frames_report = os.path.join(output_folder, dataset_name + "_frames.csv")
        self.translation_config_report = os.path.join(output_folder, dataset_name + "_config.json")
        self.run_status_report = os.path.join(output_folder, dataset_name + "_status.json")
        
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
        self.save_crop_frames = not self.translation_sensitivity

        # 是否padding, num init  pose
        self.num_init_pose = default_confs['num_init_pose']
        self.padding = default_confs['padding']
        self.euler_angles, self.translation, self.origin = get_init(self.gt_pose)
        gt_init_translation = list(self.translation)
        if self.translation_sensitivity:
            self.translation, prior_ecef, delta_ecef = add_translation_prior_ecef_xy(
                self.translation, self.prior_dx_m, self.prior_dy_m
            )
            self.origin = prior_ecef.tolist()
        else:
            prior_ecef = np.asarray(self.origin, dtype=np.float64)
            delta_ecef = np.zeros(3, dtype=np.float64)
        self.render_config['init_rot'], self.render_config['init_trans'] = self.euler_angles, self.translation
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)

        if self.translation_sensitivity:
            actual_delta_ecef = translation_error_ecef(gt_init_translation, self.translation)
            config_record = {
                "experiment": "translation_only_ecef_xy",
                "sequence": dataset_name,
                "expected_frames": len(self.img_list),
                "sample_num": self.sample_num,
                "random_seed": self.random_seed,
                "prior_dx_m": self.prior_dx_m,
                "prior_dy_m": self.prior_dy_m,
                "prior_dz_m": 0.0,
                "prior_xy_magnitude_m": float(np.hypot(self.prior_dx_m, self.prior_dy_m)),
                "yaw_noise_deg": 0.0,
                "gt_reset_enabled": False,
                "save_crop_frames": self.save_crop_frames,
                "initial_gt_wgs84": gt_init_translation,
                "initial_gt_ecef": WGS84_to_ECEF(gt_init_translation),
                "delta_ecef_m": delta_ecef.tolist(),
                "initial_prior_wgs84": self.translation,
                "initial_prior_ecef": prior_ecef.tolist(),
                "actual_prior_error_ecef_m": actual_delta_ecef.tolist(),
            }
            with open(self.translation_config_report, "w") as f:
                json.dump(config_record, f, indent=2)
            self._write_run_status("running", 0)
            logging.warning(
                "Translation-only ECEF prior: dX=%.3f m dY=%.3f m actual=%s",
                self.prior_dx_m,
                self.prior_dy_m,
                np.round(actual_delta_ecef, 6).tolist(),
            )
        
        self.device = 'cuda'
        self.origin = torch.tensor(self.origin, device=self.device)
        self.query_camera, self.render_camera = self.query_camera.to(self.device), self.render_camera.to(self.device)

        self.pose_q.put_nowait((self.euler_angles, self.translation))
        if not self.translation_sensitivity:
            self.pose_q.put_nowait((self.euler_angles, self.translation))

    @staticmethod
    def _normalize_pose_name(name):
        base = os.path.basename(name)
        if '_' not in base:
            return base[:-4] + '_0.png'
        return base

    def _compute_frame_error_against_gt(self, qname, pred_translation, pred_euler_angles):
        gt_name = self._normalize_pose_name(qname)
        gt_info = self.gt_pose_dict.get(gt_name)
        if gt_info is None:
            return None, None, None, None

        gt_translation = np.asarray(gt_info['trans'], dtype=float)
        gt_euler = np.asarray(gt_info['euler'], dtype=float)
        pred_translation = np.asarray(pred_translation, dtype=float)
        pred_euler_angles = np.asarray(pred_euler_angles, dtype=float)

        gt_eval_euler = [gt_euler[1], gt_euler[0], gt_euler[2]]
        pred_eval_euler = [pred_euler_angles[1], pred_euler_angles[0], pred_euler_angles[2]]
        R_gt, t_gt = euler_angles_to_matrix_ECEF_w2c(gt_eval_euler, gt_translation)
        R_pred, t_pred = euler_angles_to_matrix_ECEF_w2c(pred_eval_euler, pred_translation)

        e_t = float(np.linalg.norm(-t_gt + t_pred, axis=0))
        cos = np.clip((np.trace(np.dot(R_gt.T, R_pred)) - 1) / 2, -1.0, 1.0)
        e_R = float(np.rad2deg(np.abs(np.arccos(cos))))
        return e_t, e_R, gt_euler.tolist(), gt_translation.tolist()

    def _pose_error_record(self, qname, translation, euler_angles):
        gt_name = self._normalize_pose_name(qname)
        gt_info = self.gt_pose_dict.get(gt_name)
        if gt_info is None:
            return None
        gt_translation = np.asarray(gt_info['trans'], dtype=np.float64)
        gt_euler = np.asarray(gt_info['euler'], dtype=np.float64)
        gt_ecef = np.asarray(WGS84_to_ECEF(gt_translation), dtype=np.float64)
        estimated_ecef = np.asarray(WGS84_to_ECEF(translation), dtype=np.float64)
        error_ecef = estimated_ecef - gt_ecef
        yaw_error = abs((float(euler_angles[2]) - float(gt_euler[2]) + 180.0) % 360.0 - 180.0)
        return {
            "gt_ecef": gt_ecef.tolist(),
            "estimated_ecef": estimated_ecef.tolist(),
            "error_ecef": error_ecef.tolist(),
            "error_xy_m": float(np.linalg.norm(error_ecef[:2])),
            "error_3d_m": float(np.linalg.norm(error_ecef)),
            "yaw_error_deg": float(yaw_error),
        }

    def _translation_frame_record(self, frame, img_path, crop_trans, crop_euler,
                                  pred_trans=None, pred_euler=None,
                                  success=True, failure_reason="", timings=None):
        row = {field: "" for field in TRANSLATION_FRAME_FIELDS}
        row.update({
            "frame": frame,
            "image": os.path.basename(img_path),
            "injected_dx_m": self.prior_dx_m,
            "injected_dy_m": self.prior_dy_m,
            "optimization_success": int(success),
            "failure_reason": failure_reason,
        })
        prior = self._pose_error_record(img_path, crop_trans, crop_euler)
        if prior is not None:
            gt_ecef = prior["gt_ecef"]
            crop_ecef = prior["estimated_ecef"]
            error_ecef = prior["error_ecef"]
            row.update({
                "gt_ecef_x": gt_ecef[0],
                "gt_ecef_y": gt_ecef[1],
                "gt_ecef_z": gt_ecef[2],
                "crop_prior_ecef_x": crop_ecef[0],
                "crop_prior_ecef_y": crop_ecef[1],
                "crop_prior_ecef_z": crop_ecef[2],
                "prior_error_x_m": error_ecef[0],
                "prior_error_y_m": error_ecef[1],
                "prior_error_z_m": error_ecef[2],
                "prior_error_xy_m": prior["error_xy_m"],
                "prior_error_3d_m": prior["error_3d_m"],
            })
        if pred_trans is not None and pred_euler is not None:
            final = self._pose_error_record(img_path, pred_trans, pred_euler)
            _, rotation_error, _, _ = self._compute_frame_error_against_gt(
                img_path, pred_trans, pred_euler
            )
            if final is not None:
                pred_ecef = final["estimated_ecef"]
                error_ecef = final["error_ecef"]
                row.update({
                    "pred_ecef_x": pred_ecef[0],
                    "pred_ecef_y": pred_ecef[1],
                    "pred_ecef_z": pred_ecef[2],
                    "final_error_x_m": error_ecef[0],
                    "final_error_y_m": error_ecef[1],
                    "final_error_z_m": error_ecef[2],
                    "final_error_xy_m": final["error_xy_m"],
                    "final_error_3d_m": final["error_3d_m"],
                    "final_yaw_error_deg": final["yaw_error_deg"],
                    "final_rotation_error_deg": rotation_error,
                })
        if timings:
            row.update(timings)
        return row

    def _write_run_status(self, status, successful_frames, failure_record=None):
        if not self.translation_sensitivity:
            return
        record = {
            "status": status,
            "sequence": self.dataset_name,
            "expected_frames": len(self.img_list),
            "successful_frames": successful_frames,
            "attempted_frames": successful_frames + int(failure_record is not None),
        }
        if failure_record is not None:
            record.update(failure_record)
        with open(self.run_status_report, "w") as f:
            json.dump(record, f, indent=2)

    def _write_reset_report(self, total_frames, reset_count, sample_num, reset_records):
        lines = [
            f"total_frames {total_frames}",
            f"reset_threshold_translation_m 20.0",
            f"reset_threshold_rotation_deg 20.0",
            f"reset_count {reset_count}",
            f"sample_num {sample_num}",
        ]
        for record in reset_records:
            lines.append(
                "frame {frame} image {image} error_t_m {error_t_m:.6f} error_r_deg {error_r_deg:.6f}".format(
                    **record
                )
            )
        with open(self.gt_reset_report, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _disable_optimizer_debug_prints(localizer):
        opts = localizer.refiner.optimizer
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        for opt in opts:
            cost_fn = getattr(opt, "cost_fn", None)
            if cost_fn is None:
                continue
            if hasattr(cost_fn, "lm_timing_print"):
                cost_fn.lm_timing_print = False
            if hasattr(cost_fn, "angle_debug_print"):
                cost_fn.angle_debug_print = False
            if hasattr(cost_fn, "depth_debug_print"):
                cost_fn.depth_debug_print = False
            if hasattr(cost_fn, "depth_debug_verbose"):
                cost_fn.depth_debug_verbose = False
            if hasattr(cost_fn, "depth_verify_query_print"):
                cost_fn.depth_verify_query_print = False

    # ---------------- 裁剪线程（取代原渲染） ----------------        
    def rendering_worker(self):
        import torch
        from pixloc.crop.ray_casting import TargetLocation
        if self.translation_sensitivity:
            np.random.seed(self.random_seed)
        # === 1. 配置路径 ===
        # ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DOM.tif"
        # ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/hangtianDSM.tif"
        # ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/feicuiwan_mix.npy"
        # ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DOM.tif"
        # ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DSM.tif"
        # ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/feicuiwan.npy"
        # ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/hangtianDOM.tif"
        # ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/hangtianDSM.tif"
        # ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/hangtian.npy"
        # ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/hangtianDOM.tif"
        # ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DSM.tif"
        # ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.5/0.5/feicuiwan_mix.npy"
        ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif"
        ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
        ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3.npy"
        if not os.path.exists(ref_DSM_path):
            raise FileNotFoundError(f"找不到文件: {ref_DSM_path}")
        # 输出路径指向 worker 的 output
        
        ref_rgb_path = self.pre_path
        ref_depth_path = self.pre_path
        paths_pack = (ref_rgb_path, ref_depth_path)

        # === 2. 初始化地图 (只做一次) ===
        logging.info(f"Loading Map from {ref_DOM_path}...")
        map_data_pack = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
        _, ray_area, ray_area_minZ, _, _, _ = map_data_pack
        locator = TargetLocation({"ray_casting": {}}, use_dsm=False)
        logging.info("Map Loaded Successfully.")

        idx = 0
        fps_log_every = 0
        sample_num_report_lines = []

        def send_crop_failure(reason, euler, trans):
            failure_item = {
                "failure": True,
                "frame": idx,
                "reason": reason,
                "render_euler": list(euler),
                "render_trans": list(trans),
            }
            while True:
                try:
                    self.task_q.put(failure_item, timeout=1)
                    break
                except queue.Full:
                    if self.stop_evt.is_set():
                        break

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

            crop_t0 = time.perf_counter()
            try:
                crop_data = process_map_crop(
                    ref_DSM_path,
                    pose_data, 
                    ref_npy_path, 
                    name, 
                    map_data_pack, 
                    paths_pack,
                    ray_area,
                    ray_area_minZ,
                    locator=locator,
                )
            except Exception as exc:
                logging.error(f"process_map_crop 失败 name={name}: {exc}")
                if self.translation_sensitivity:
                    send_crop_failure(f"crop_exception: {exc}", euler, trans)
                    break
                continue
            color = crop_data['dom_crop']
            points3d = crop_data['point_cloud_crop']
            valid_mask = np.isfinite(points3d).all(axis=-1) & (points3d[..., 2] > 0)    #valid_mask: 有效掩码，用于标识哪些点是有效的
            if not np.any(valid_mask):
                logging.error(f"裁剪结果无有效3D点: frame={name}")
                if self.translation_sensitivity:
                    send_crop_failure("crop_has_no_valid_3d_points", euler, trans)
                    break
                continue

            # --- 对齐参考尺寸到渲染相机分辨率（保证与 query 一致） ---
            target_w, target_h = int(self.render_camera_osg[0]), int(self.render_camera_osg[1])
            color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            points3d = cv2.resize(points3d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # resize_save_path = os.path.join(self.outputs, f'{name}_1_resized.png')
            # # 注意：内存中是RGB，OpenCV保存需要BGR，否则颜色错误
            # cv2.imwrite(resize_save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            # logging.info(f"Saved resized image: {resize_save_path}")

            # padding 到 16 的倍数，使用复制边界避免 0 值；同步处理有效掩码
            mask = valid_mask.astype(np.uint8)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if self.padding:
                pad_h = (16 - target_h % 16) % 16
                pad_w = (16 - target_w % 16) % 16
                if pad_h or pad_w:
                    color = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                    points3d = cv2.copyMakeBorder(points3d, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            if self.save_crop_frames:
                pad_path = self.outputs
                padded_save_path = os.path.join(pad_path, f'{name}.png')
                cv2.imwrite(padded_save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            # logging.info(f"Saved padded image: {padded_save_path}")
            mask = mask.astype(bool)

            # 预先采样 3D 点（避免后端再随机），只保留有效点
            points_valid = points3d[mask]
            if points_valid.shape[0] == 0:
                logging.error(f"裁剪结果无有效3D点(经过padding/mask后): frame={name}")
                if self.translation_sensitivity:
                    send_crop_failure("crop_has_no_valid_points_after_resize", euler, trans)
                    break
                continue
            if points_valid.shape[0] < self.sample_num:
                msg = f"裁剪结果有效3D点过少: frame={name}, count={points_valid.shape[0]}"
                logging.warning(msg)
                sample_num_report_lines.append(msg)
            
            sample_num = min(self.sample_num, points_valid.shape[0])
            sel = np.random.choice(points_valid.shape[0], sample_num, replace=False)
            points_sampled = points_valid[sel]
            coords_all = np.argwhere(mask)  # (N,2) -> row, col on padded image
            coords_sampled = coords_all[sel]

            p2d_r_np = np.stack([coords_sampled[:, 1], coords_sampled[:, 0]], axis=1).astype(np.float32)
            
            visible_r_np = np.ones((sample_num,), dtype=bool)

            # 转 Tensor 并增加维度 (1, 1, N, 2)
            # 这一步在 Worker 里做，传给 Queue 
            p2d_r = torch.from_numpy(p2d_r_np).view(1, 1, -1, 2)
            visible_r = torch.from_numpy(visible_r_np).view(1, 1, -1)

            # # 保存调试信息：采样点的像素位置（padding 后）与图像尺寸
            # debug_npz = os.path.join(self.outputs, f'{name}_debug_samples.npz')
            # np.savez_compressed(
            #     debug_npz,
            #     coords_sampled=coords_sampled,
            #     img_shape=color.shape[:2],
            #     sample_num=sample_num,
            #     p2d_r=p2d_r,
            #     visible_r=visible_r
            # )
            # logging.info(f"Saved sampled coords: {debug_npz}, head={coords_sampled[:5].tolist()}")

            t_render = (time.perf_counter() - crop_t0) * 1e3  # ms
            fps_log_every += 1
            # if fps_log_every % 30 == 0:
            #     logging.info("crop: %.2f ms", t_render)

            try:
                # p2d_r/visible_r 先不用于优化，改由后端重投影，避免尺度/内参错位
                while True:
                    try:
                        self.task_q.put(
                            (color, points_sampled, p2d_r, visible_r, euler, trans, t_render, sample_num),
                            timeout=1,
                        )
                        break
                    except queue.Full:
                        if self.stop_evt.is_set():
                            break
                        continue
            except Exception as exc:
                logging.error(f"task_q put 失败 frame={name}: {exc}")
                break
            idx += 1

        self.stop_evt.set()
        if not self.translation_sensitivity:
            with open(self.sample_num_report, "w") as f:
                f.write(f"total_frames {len(sample_num_report_lines)}\n")
                f.write("\n".join(sample_num_report_lines))
        while True:
            try:
                self.task_q.put(None, timeout=1)
                break
            except queue.Full:
                continue
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        # logging.info('Render/Crop process done')
    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        from pixloc.localization import RenderLocalizer, SimpleTracker
        if self.translation_sensitivity:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
        localizer = RenderLocalizer(self.conf)
        self._disable_optimizer_debug_prints(localizer)
        results = []   # 存盘缓冲
        flag = True
        last_euler, last_trans = None, None
        reset_count = 0
        reset_records = []
        crop_times_ms = []
        localization_times_ms = []
        back_project_times_ms = []
        feature_extract_times_ms = []
        optimizer_times_ms = []
        postprocess_times_ms = []
        sample_num = self.sample_num
        failure_record = None
        translation_file = None
        translation_writer = None
        if self.translation_sensitivity:
            translation_file = open(self.translation_frames_report, "w", newline="")
            translation_writer = csv.DictWriter(
                translation_file, fieldnames=TRANSLATION_FRAME_FIELDS
            )
            translation_writer.writeheader()
            translation_file.flush()
        # 读取 GT 深度列表（按行顺序对应帧序号）
        depth_txt_path = self.depth_txt_path 
        angle_txt_path = self.angle_txt_path
        gt_depth_list = []
        if os.path.exists(depth_txt_path):
            with open(depth_txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            gt_depth_list.append(float(parts[1]))
                        except ValueError:
                            logging.warning(f"无法解析深度值: {line.strip()}")
            logging.info(f"已从 {depth_txt_path} 读取 {len(gt_depth_list)} 条深度数据")
        else:
            logging.error(f"未找到 GT 深度文件: {depth_txt_path}")
        
        # 读取 GT 角度列表（roll 和 pitch，按行顺序对应帧序号）
        # 格式：图片名 roll pitch
        # 例如：2_0.png 0.0 40.9
        gt_roll_list = []
        gt_pitch_list = []
        if os.path.exists(angle_txt_path):
            with open(angle_txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # parts[0] 是图片名，parts[1] 是 roll，parts[2] 是 pitch
                            gt_roll_list.append(float(parts[1]))
                            gt_pitch_list.append(float(parts[2]))
                        except ValueError:
                            logging.warning(f"无法解析角度值: {line.strip()}")
            logging.info(f"已从 {angle_txt_path} 读取 {len(gt_roll_list)} 条角度数据")
        else:
            logging.warning(f"未找到 GT 角度文件: {angle_txt_path}，将不使用角度约束")
        
        warned_missing_depth = False
        warned_missing_angle = False
        for idx, (img_path, img_tensor) in enumerate(zip(self.img_list, self.query_list)):
            # 1) 拿到渲染结果（阻塞）
            while True:
                try:
                    item = self.task_q.get(timeout=1)
                    break
                except queue.Empty:
                    if self.stop_evt.is_set():
                        if self.translation_sensitivity:
                            failure_record = {
                                "failure_frame": idx,
                                "failure_image": os.path.basename(img_path),
                                "failure_reason": "render_worker_ended_before_frame",
                            }
                            self.localization_failure_evt.set()
                            break
                        return
                    continue
            if failure_record is not None:
                break
            if item is None:        # 渲染端提前喊停
                if self.translation_sensitivity and idx < len(self.img_list):
                    failure_record = {
                        "failure_frame": idx,
                        "failure_image": os.path.basename(img_path),
                        "failure_reason": "render_worker_ended_before_frame",
                    }
                    self.localization_failure_evt.set()
                break
            if isinstance(item, dict) and item.get("failure"):
                reason = item["reason"]
                translation_writer.writerow(self._translation_frame_record(
                    idx,
                    img_path,
                    item["render_trans"],
                    item["render_euler"],
                    success=False,
                    failure_reason=reason,
                ))
                translation_file.flush()
                failure_record = {
                    "failure_frame": idx,
                    "failure_image": os.path.basename(img_path),
                    "failure_reason": reason,
                }
                self.localization_failure_evt.set()
                self.stop_evt.set()
                logging.warning(
                    "Terminal crop failure: frame=%d image=%s reason=%s",
                    idx, os.path.basename(img_path), reason,
                )
                break
            color, depth, p2d_r, visible_r, render_euler, render_trans, crop_time_ms, sample_num = item
            crop_times_ms.append(float(crop_time_ms))
            # 2) 反投影
            localization_t0 = time.perf_counter()
            # t1 = time.time()
            if last_trans is None:
                last_euler, last_trans = render_euler, render_trans
            back_project_t0 = time.perf_counter()
            P3d, T_w2c_mod, T_init, dd, p2d_r_final, render_T_ecef, back_project_timing = self.back_project(depth, p2d_r, render_euler, render_trans, last_euler, last_trans)
            back_project_time_ms = (time.perf_counter() - back_project_t0) * 1e3
            back_project_times_ms.append(back_project_time_ms)
            # P3d, T_w2c_mod, T_init, dd = self.back_project(depth, render_euler, render_trans, render_euler, render_trans)
            
            # 控制传入优化的3D点规模，避免显存爆
            # max_pts_opt = 2000
            # if P3d.shape[0] > max_pts_opt:
            #     perm = torch.randperm(P3d.shape[0], device=P3d.device)
            #     keep = perm[:max_pts_opt]
            #     P3d = P3d[keep]

            # t2 = time.time()
            # print('反投影耗时：', t2-t1)
            # print("P3d: ", P3d)
            # 2) 调用定位
            ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
            if idx < len(gt_depth_list):
                gt_depth = gt_depth_list[idx]
                # print(f"idx: {idx} ; gt_depth: {gt_depth}")
            else:
                gt_depth = None
                if not warned_missing_depth:
                    logging.warning(f"GT 深度数量({len(gt_depth_list)})少于图像数量({len(self.img_list)}), 后续帧将不使用 gt_depth")
                    warned_missing_depth = True
            
            # 获取 GT 角度值
            if idx < len(gt_roll_list) and idx < len(gt_pitch_list):
                gt_roll = gt_roll_list[idx]
                gt_pitch = gt_pitch_list[idx]
                # print(f"idx: {idx} ; gt_roll: {gt_roll}° ; gt_pitch: {gt_pitch}°")
            else:
                gt_roll = None
                gt_pitch = None
                if not warned_missing_angle:
                    logging.warning(f"GT 角度数量({len(gt_roll_list)})少于图像数量({len(self.img_list)}), 后续帧将不使用 gt_roll 和 gt_pitch")
                    warned_missing_angle = True
            current_n = p2d_r_final.shape[-2] # 获取过滤后的点数
            visible_r_final = torch.ones((1, current_n), dtype=torch.bool, device=self.device)
            ret = localizer.run_query(
                img_path, self.query_camera, self.render_camera,
                color,  # render_frame
                query_T = T_init,
                render_T= T_w2c_mod,
                Points_3D_ECEF = P3d,
                query_resize_ratio = self.query_resize_ratio,
                p2d_r=p2d_r_final,
                visible_r=visible_r_final,
                dd = dd,
                gt_depth=gt_depth,
                gt_roll=gt_roll,
                gt_pitch=gt_pitch,
                dsm_path=ref_DSM_path,
                render_T_ecef=render_T_ecef,  # 新增：渲染帧的真实ECEF坐标
                gt_pose_dict=self.gt_pose_dict,
                last_frame_info = self.last_frame_info,
                image_query = img_tensor
            )
            localization_time_ms = (time.perf_counter() - localization_t0) * 1e3
            localization_times_ms.append(localization_time_ms)
            ret_timings = ret.get('timings_ms', {})
            feature_extract_ms = float(ret_timings.get('feature_extract_ms', 0.0))
            optimizer_ms = float(ret_timings.get('optimizer_ms', 0.0))
            postprocess_ms = float(ret_timings.get('postprocess_ms', 0.0))
            optimizer_levels_ms = ret_timings.get('optimizer_level_times_ms', [])
            optimizer_level_detail_ms = ret_timings.get('optimizer_level_detail_ms', [])
            optimizer_level_brief = []
            for detail in optimizer_level_detail_ms:
                residual_sum = sum(detail.get('residual_eval_ms', []))
                post_cost_sum = sum(detail.get('post_cost_ms', []))
                linear_sum = sum(detail.get('linear_solve_ms', []))
                res_breakdown = detail.get('residual_cost_breakdown', [])
                post_breakdown = detail.get('post_cost_breakdown', [])
                def _sum_breakdown(items, key):
                    return round(float(sum(float(x.get(key, 0.0)) for x in items)), 2)
                optimizer_level_brief.append({
                    'level': int(detail.get('level', -1)),
                    'iters': int(detail.get('num_iters', 0)),
                    'res': round(float(residual_sum), 2),
                    'res_detail': {
                        'proj': _sum_breakdown(res_breakdown, 'proj_ms'),
                        'depth': _sum_breakdown(res_breakdown, 'depth_ms'),
                        'depth_center': _sum_breakdown(res_breakdown, 'depth_center_ms'),
                        'depth_dsm': _sum_breakdown(res_breakdown, 'depth_dsm_ms'),
                        'depth_fd': _sum_breakdown(res_breakdown, 'depth_fd_ms'),
                        'angle': _sum_breakdown(res_breakdown, 'angle_ms'),
                        'angle_setup': _sum_breakdown(res_breakdown, 'angle_setup_ms'),
                        'angle_fd': _sum_breakdown(res_breakdown, 'angle_fd_ms'),
                        'system': _sum_breakdown(res_breakdown, 'system_ms'),
                    },
                    'post': round(float(post_cost_sum), 2),
                    'post_detail': {
                        'proj': _sum_breakdown(post_breakdown, 'proj_ms'),
                        'depth': _sum_breakdown(post_breakdown, 'depth_ms'),
                        'depth_center': _sum_breakdown(post_breakdown, 'depth_center_ms'),
                        'depth_dsm': _sum_breakdown(post_breakdown, 'depth_dsm_ms'),
                        'depth_fd': _sum_breakdown(post_breakdown, 'depth_fd_ms'),
                        'angle': _sum_breakdown(post_breakdown, 'angle_ms'),
                        'angle_setup': _sum_breakdown(post_breakdown, 'angle_setup_ms'),
                        'angle_fd': _sum_breakdown(post_breakdown, 'angle_fd_ms'),
                        'system': _sum_breakdown(post_breakdown, 'system_ms'),
                    },
                    'solve': round(float(linear_sum), 2),
                })
            bp_input_ms = float(back_project_timing.get('input_prepare_ms', 0.0))
            bp_pose_ms = float(back_project_timing.get('pose_convert_ms', 0.0))
            bp_samples_ms = float(back_project_timing.get('sample_gen_ms', 0.0))
            feature_extract_times_ms.append(feature_extract_ms)
            optimizer_times_ms.append(optimizer_ms)
            postprocess_times_ms.append(postprocess_ms)
            frame_timings = {
                "crop_ms": float(crop_time_ms),
                "localization_ms": float(localization_time_ms),
                "back_project_ms": float(back_project_time_ms),
                "feature_ms": feature_extract_ms,
                "optimizer_ms": optimizer_ms,
                "postprocess_ms": postprocess_ms,
            }
            if self.translation_sensitivity and not ret.get('success', False):
                reason = "optimizer_returned_success_false"
                translation_writer.writerow(self._translation_frame_record(
                    idx,
                    img_path,
                    render_trans,
                    render_euler,
                    success=False,
                    failure_reason=reason,
                    timings=frame_timings,
                ))
                translation_file.flush()
                failure_record = {
                    "failure_frame": idx,
                    "failure_image": os.path.basename(img_path),
                    "failure_reason": reason,
                }
                self.localization_failure_evt.set()
                self.stop_evt.set()
                logging.warning(
                    "Terminal localization failure: frame=%d image=%s",
                    idx, os.path.basename(img_path),
                )
                break
            pred_euler = ret['euler_angles']
            pred_trans = ret['translation']
            err_t, err_r, gt_euler, gt_trans = self._compute_frame_error_against_gt(
                img_path,
                pred_trans,
                pred_euler,
            )
            use_gt_reset = (
                not self.translation_sensitivity
                and err_t is not None
                and err_r is not None
                and (err_t > 30.0 or err_r > 30.0)
            )
            if self.translation_sensitivity:
                translation_writer.writerow(self._translation_frame_record(
                    idx,
                    img_path,
                    render_trans,
                    render_euler,
                    pred_trans=pred_trans,
                    pred_euler=pred_euler,
                    success=True,
                    timings=frame_timings,
                ))
                translation_file.flush()
            if use_gt_reset:
                reset_count += 1
                reset_records.append({
                    "frame": idx,
                    "image": os.path.basename(img_path),
                    "error_t_m": err_t,
                    "error_r_deg": err_r,
                })
                next_euler = gt_euler
                next_trans = gt_trans
                self.last_frame_info['observations'] = []
                self.last_frame_info.pop('euler_angles', None)
                self.last_frame_info.pop('translation', None)
            else:
                next_euler = pred_euler
                next_trans = pred_trans

            last_euler, last_trans = next_euler, next_trans
            # self.last_frame_info['observations'] = ret['observations']
            # self.last_frame_info['euler_angles'] = ret['euler_angles']
            # self.last_frame_info['translation'] = ret['translation']
            # cv2.imwrite(f'{self.outputs}/{idx}_query.png', img_tensor)
            # if idx % fps_log_every == 0:
            #     logging.info("loc %.2f ms", (time.time()-t0)*1e3)
            
            # 3) 把得到的新姿态塞回 pose_q，供下一帧渲染
            if idx < len(self.img_list)-1:
                while True:
                    try:
                        self.pose_q.put((next_euler, next_trans), timeout=1)
                        break
                    except queue.Full:
                        if self.stop_evt.is_set():
                            break
                        continue
            print(
                "Localization Thread calculated frames: {} | crop_ms={:.2f} | "
                "loc_ms={:.2f} | back_project_ms={:.2f} | feature_ms={:.2f} | "
                "optimizer_ms={:.2f} | post_ms={:.2f} | bp_input_ms={:.2f} | "
                "bp_pose_ms={:.2f} | bp_samples_ms={:.2f} | opt_levels_ms={} | opt_breakdown={} | Pitch, Roll, Yaw: {} | "
                "Longitude, Latitude, Altitude: {}".format(
                    idx,
                    float(crop_time_ms),
                    float(localization_time_ms),
                    float(back_project_time_ms),
                    feature_extract_ms,
                    optimizer_ms,
                    postprocess_ms,
                    bp_input_ms,
                    bp_pose_ms,
                    bp_samples_ms,
                    [round(float(x), 2) for x in optimizer_levels_ms],
                    optimizer_level_brief,
                    np.asarray(pred_euler).tolist(),
                    np.asarray(pred_trans).tolist(),
                )
            )
            # 4) 暂存结果
            qname = os.path.basename(img_path)
            results.append(f"{qname} {' '.join(map(str, pred_trans))} "
                           f"{' '.join(map(str, [pred_euler[1], pred_euler[0], pred_euler[2]]))}")

            if self.stop_evt.is_set():
                break
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))
        if translation_file is not None:
            translation_file.close()
        if not self.translation_sensitivity:
            self._write_reset_report(len(results), reset_count, sample_num, reset_records)
        if self.translation_sensitivity:
            if failure_record is None and len(results) == len(self.img_list):
                self._write_run_status("complete", len(results))
            else:
                if failure_record is None:
                    failure_record = {
                        "failure_frame": len(results),
                        "failure_image": "",
                        "failure_reason": "sequence_ended_early",
                    }
                self.localization_failure_evt.set()
                self._write_run_status("failed", len(results), failure_record)
        if crop_times_ms and localization_times_ms:
            logging.info(
                "Timing summary: crop avg %.2f ms | localization avg %.2f ms | "
                "back_project avg %.2f ms | feature avg %.2f ms | optimizer avg %.2f ms | "
                "post avg %.2f ms | crop max %.2f ms | localization max %.2f ms",
                float(np.mean(crop_times_ms)),
                float(np.mean(localization_times_ms)),
                float(np.mean(back_project_times_ms)) if back_project_times_ms else 0.0,
                float(np.mean(feature_extract_times_ms)) if feature_extract_times_ms else 0.0,
                float(np.mean(optimizer_times_ms)) if optimizer_times_ms else 0.0,
                float(np.mean(postprocess_times_ms)) if postprocess_times_ms else 0.0,
                float(np.max(crop_times_ms)),
                float(np.max(localization_times_ms)),
            )
        
        # self.flush_pose_and_send_sentinel()
        # 收尾：给渲染端塞哨兵 + 关队列 + 事件
        while True:
            try:
                self.pose_q.put(None, timeout=1)
                break
            except queue.Full:
                continue
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

    def back_project(self, points_3d_input, p2d_r_input, euler_angles, translation, query_euler_angles, query_translation, num_samples = None, device = 'cuda'):
        """
        修改后的 back_project:
        1. 不再进行随机采样 (不再生成 xs, ys)。
        2. 直接接收之前选好的 3D 点 (points_3d_input)。
        3. 调用 get_3D_samples_v3 进行坐标转换(EPSG->ECEF)、去中心化和位姿生成。
        """
        back_project_t0 = time.perf_counter()

        # 1. 确保输入点在 GPU 上
        if not torch.is_tensor(points_3d_input):
            # 假设输入是 [N, 3]
            points_3d = torch.as_tensor(points_3d_input, device=device, dtype=torch.float32)
        else:
            points_3d = points_3d_input.to(device)
        if p2d_r_input is not None:
            if not torch.is_tensor(p2d_r_input):
                 p2d_r_input = torch.as_tensor(p2d_r_input, device=device)
            else:
                 p2d_r_input = p2d_r_input.to(device)
        input_prepare_ms = (time.perf_counter() - back_project_t0) * 1e3


        # 2. 转换渲染帧的位姿 (欧拉角 -> 旋转矩阵)
        pose_t0 = time.perf_counter()
        T_render_in_ECEF_c2w = torch.as_tensor(
            euler_angles_to_matrix_ECEF(euler_angles, translation),
            device=device, dtype=torch.float32
        )
        pose_convert_ms = (time.perf_counter() - pose_t0) * 1e3

        # 3. 直接调用处理函数
        # 注意：mkpts_r 传 None，因为我们不需要 2D 像素坐标来反投影了
        # depth_mat 参数这里实际上传入的是 3D 点集
        sample_t0 = time.perf_counter()
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd, p2d_r_filtered, render_T_ecef = get_3D_samples_v3(
            mkpts_r=p2d_r_input, 
            depth_mat=points_3d, 
            T_c2w=T_render_in_ECEF_c2w, 
            camera=self.render_camera, 
            euler_angles=euler_angles, 
            translation=translation, 
            query_euler_angles=query_euler_angles, 
            query_translation=query_translation, 
            origin=self.origin, 
            num_init_pose=self.num_init_pose,
            mul=self.mul
        )
        sample_gen_ms = (time.perf_counter() - sample_t0) * 1e3

        timing = {
            'input_prepare_ms': float(input_prepare_ms),
            'pose_convert_ms': float(pose_convert_ms),
            'sample_gen_ms': float(sample_gen_ms),
        }
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd, p2d_r_filtered, render_T_ecef, timing
    # 假设的计算位姿函数
    def calculate_pose(self, render_frame, points_3d):
        # 根据渲染图和3D点计算位姿
        return "Pose"

    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        if self.translation_sensitivity and self.localization_failure_evt.is_set():
            logging.warning(
                "Skipping aggregate evaluation because the sequence failed: %s",
                self.run_status_report,
            )
            return
        evaluate(self.estimated_pose, self.gt_pose)
    def run(self):
        ctx = mp.get_context("spawn")        # 保持 spawn
        p_render = ctx.Process(target=self.rendering_worker, daemon=True)
        p_loc    = ctx.Process(target=self.localization_worker, daemon=True)

        p_render.start(); p_loc.start()
        p_loc.join()                         # 先等定位结束
        if p_loc.exitcode != 0:
            self.localization_failure_evt.set()
            self._write_run_status(
                "worker_process_error",
                0,
                {
                    "failure_frame": None,
                    "failure_image": "",
                    "failure_reason": f"localization_exit_{p_loc.exitcode}",
                },
            )
        p_render.join(5)                    # 最多等 30 s
        if p_render.is_alive():              # 兜底：仍卡住就强退
            p_render.terminate()
            p_render.join()
        return not self.localization_failure_evt.is_set()
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
        default="0018_V",
        help="实验名称，用于日志标记"
    )

    parser.add_argument(
        "--init_euler",
        type=str,
        default="[0.0, 37.1, -42.4]",
        help="初始欧拉角（格式如：[roll, pitch, yaw]）"
    )

    parser.add_argument(
        "--init_trans",
        type=str,
        default="[112.999054, 28.290497, 157.509]",
        help="初始平移（格式如：[x, y, z]）"
    )

    parser.add_argument(
        "--sample_num",
        type=int,
        default=300,
        help="每帧采样的3D点数量"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="/media/amax/PS2000/rebuttal/0906/seed300",
        help="输出目录"
    )

    parser.add_argument(
        "--translation_sensitivity",
        action="store_true",
        help="启用第一帧ECEF X/Y平移先验敏感性实验"
    )

    parser.add_argument(
        "--prior_dx_m",
        type=float,
        default=0.0,
        help="第一帧ECEF X分量扰动（米）"
    )

    parser.add_argument(
        "--prior_dy_m",
        type=float,
        default=0.0,
        help="第一帧ECEF Y分量扰动（米）"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="实验随机种子"
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
    dual_task = DualProcessTask(
        config,
        init_euler=init_euler,
        init_trans=init_trans,
        name=name,
        sample_num=args.sample_num,
        output_folder=args.output_folder,
        translation_sensitivity=args.translation_sensitivity,
        prior_dx_m=args.prior_dx_m,
        prior_dy_m=args.prior_dy_m,
        random_seed=args.random_seed,
    )
    run_complete = dual_task.run()
    if run_complete:
        dual_task.eval()
    else:
        sys.exit(2)
    
