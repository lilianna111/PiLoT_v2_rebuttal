from viz import SceneViz
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from transform import  WGS84_to_ECEF, get_rotation_enu_in_ecef, visualize_matches, ECEF_to_WGS84

def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x

def to_numpy(x): return todevice(x, 'numpy')

def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    centers = poses[:, :3, 3]  # 提取每个pose的平移向量 (N, 3)
    return np.median(pdist(centers))

def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)

viz = SceneViz()

def auto_cam_size(im_poses):
    return 0.3 * get_med_dist_between_poses(im_poses)
def load_poses(pose_file):
    """Load poses from the pose file."""
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
    return pose_dict
def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask

def filter_invalid_depth(depthmap):
    """Filter out invalid depth values (65504)"""
    valid_depth = (depthmap != 65504)
    depthmap[~valid_depth] = 0  # Set invalid depths to 0
    return depthmap, valid_depth

# Load all poses

pose_file = "/media/ubuntu/PS20001/poses/USA_seq5@8@cloudy@300-100@200.txt"
pose_dict = load_poses(pose_file)

# get query pose



intrinsics = np.array([[1158.8, 0, 480.0],[0, 1158.8, 270.0], [0, 0, 1]])
# Get all image files from multi directory
import os
import glob

multi_dir = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200"
image_files = sorted(glob.glob(os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy/GeoPixel", "*_0.png")))
depth_files = sorted(glob.glob(os.path.join(multi_dir, "*_1.png")))

print(f"Found {len(image_files)} image pairs")

# Process each image pair
for i, (img_file, depth_file) in enumerate(zip(image_files, depth_files)):
    if i % 10 !=0: continue
    print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_file)}")
    
    # Load image and depth
    imgr = cv2.imread(img_file)
    depth_path = os.path.dirname(depth_file)  + img_file.split('/')[-1].replace('_0.png', '_1.png')
    depthr = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.flip(depthr, 0)
    depth, valid_depth = filter_invalid_depth(depth)
    
    # Get pose for this image (extract image number from filename)
    img_name = os.path.basename(img_file)
    if img_name in pose_dict.keys():
        query_pose = pose_dict[img_name]
        lon, lat, alt, roll, pitch, yaw = map(float, query_pose)
        euler_angles = [pitch, roll, yaw]
        translation = [lon, lat, alt]
        rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
        rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        t_c2w = WGS84_to_ECEF(translation)
        query_T = np.eye(4)

        query_T[:3, :3] = R_c2w
        query_T[:3, 3] = t_c2w
        query_T[:3, 1] = -query_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
        query_T[:3, 2] = -query_T[:3, 2]  # Z轴取反
        q_pose = query_T
    else:
        print(f"Warning: No pose found for image {img_num}, skipping...")
        continue
    
    # Generate 3D points
    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depth, intrinsics, q_pose)
    
    # Set camera size based on distance between poses
    # if i == 0:
    #     # Calculate camera size from all poses
    #     all_poses_array = np.array([pose_dict.item()[str(j)] for j in range(len(image_files)) if str(j) in pose_dict.item()])
    #     if len(all_poses_array) > 1:
    #         auto_size = auto_cam_size(all_poses_array)
    #         cam_size = max(auto_size, 25)  # 确保相机大小至少为50
    #     else:
    #         cam_size = 25  # 增加默认相机大小
    #     print(f"Auto camera size: {auto_size if len(all_poses_array) > 1 else 'N/A'}, Final camera size: {cam_size}")
    
    # Add point cloud and camera to visualization
    valid_mask = valid_depth
    colors = rgb(imgr)
    viz.add_pointcloud(pts3d, colors, valid_mask)
    
    # Use different colors for different cameras
    color_intensity = i / max(1, len(image_files) - 1)
    viz.add_camera(pose_c2w=q_pose,
                    focal=intrinsics[0, 0],
                    color=(color_intensity * 255, (1 - color_intensity) * 255, 0),
                    image=colors,
                    cam_size=3)

print("Visualization complete!")
viz.show()