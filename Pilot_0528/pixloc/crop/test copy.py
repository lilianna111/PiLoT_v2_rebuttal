import os
from utils import read_DSM_config, generate_dji_seed
from transform_colmap import transform_colmap_pose_intrinsic
from proj2map_test import generate_ref_map




ref_DOM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DOM.tif"
ref_DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"
ref_npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.npy"
query_path = "/home/amax/Documents/datasets/images/DJI_20250612194622_0018_V"
ref_rgb_path = "/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V"
ref_depth_path = "/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V"
txt_path = "/home/amax/Documents/code/PiLoT/PiLoT/pixloc/crop/homography_result.txt"


# Init map
geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)

# Read prior pose and intrinsic
query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(query_path, txt_path)
seed_int_trans_x = 0
seed_num_trans_x = 1
seed_int_trans_y = 0
seed_num_trans_y = 1
seed_int_trans_z = 0
seed_num_trans_z = 1
seed_int_yaw = 1
seed_num_yaw = 1
# with open(homography_result_txt, 'w') as fh_result, open(ray_tracing_result_txt, 'w') as fr_result:
for qname in query_prior_poses_dict.keys():
    q_prior_pose = query_prior_poses_dict[qname]
    q_intrinsic = query_intrinsics_dict[qname]
    osg_angle = osg_dict[qname]
    # print(osg_angle)

    img_q_w, img_q_h, sensor_q_w, sensor_q_h, f_mm_q = q_intrinsics_info[qname]

    # generate seed
    reference_poses, reference_intrinsic, reference_yaw = generate_dji_seed(q_intrinsic, q_prior_pose, qname, osg_angle,
                                                                            seed_int_trans_x, seed_int_trans_y, seed_int_trans_z, seed_int_yaw,
                                                                            seed_num_trans_x, seed_num_trans_y, seed_num_trans_z, seed_num_yaw)




# crop map
data = generate_ref_map(query_intrinsics_dict, query_prior_poses_dict, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path)
# def generate_ref_map(query_intrinsics, query_poses, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path, crop_padding, debug):
# matches_list = []
# matches_number_list = []
# rot_matches_refer_points = []
# ori_path = os_path.join(query_path, qname + '.JPG')