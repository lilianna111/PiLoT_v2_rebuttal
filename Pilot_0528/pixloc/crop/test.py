import os
from utils import read_DSM_config, generate_dji_seed
from transform_colmap import transform_colmap_pose_intrinsic
from proj2map_test import generate_ref_map

query_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/images/interval1_HKairport01/interval1"
ref_rgb_path = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"
ref_depth_path = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"

ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif"
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3.npy"
# ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/dom_cgcs.tif"
# ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/dsm_cgcs.tif"
# ref_npy_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/domdsm_cgcs.npy"

# ref_DOM_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/hangtianDOM.tif"
# ref_DSM_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/hangtianDSM.tif"
# ref_npy_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/hangtian.npy"
# pose_data = [113.002498, 28.291678, 181.082, 0.0, 0.0, 160.1]
pose_data = [112.999054, 28.290497, 157.509, 0.0, 37.1, -42.4]
# pose_data = [14.04270934472524, 22.416065873493764, 141.49985356855902,  4.551786987694334, 1.3641392882272305, 103.15503822838967]
# pose_data = [112.993723, 28.292831, 157.388, 0.0, -44.9, 79.4]
name ="demo"
# latitude: 28.292831] [longitude: 112.993723] [rel_alt: 150.388 abs_alt: 167.555] [gb_yaw: 79.1 gb_pitch: -44.9 gb_roll: 0.0]

# Init map
geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)


# Read prior pose and intrinsic
# print('pose_data', pose_data)
query_prior_poses_dict, query_intrinsics_dict, _, _= transform_colmap_pose_intrinsic(pose_data)
# print('query_intrinsics_dict', query_intrinsics_dict)
# print('query_prior_poses_dict', query_prior_poses_dict)
# print('osg_dict', osg_dict)

# crop map
# data = generate_ref_map(ref_DSM_path, pose_data, ref_npy_path, geotransform, query_intrinsics_dict, query_prior_poses_dict, name, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path)
# query_intrinsics, query_poses, area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path
data = generate_ref_map(name,query_intrinsics_dict, query_prior_poses_dict, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path,debug=True)



# ref_DOM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DOM.tif"
# ref_DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DSM.tif"
# ref_npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/feicuiwan.npy"
# pose_data = [113.00255932833957, 28.291723905925195, 178.93339748028666, -1.091163513560782, 1.6895890397360818, 159.90228203432576]
# name ="alllow"
# geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
# query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
# data = generate_ref_map(ref_DSM_path, pose_data, ref_npy_path, geotransform, query_intrinsics_dict, query_prior_poses_dict, name, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path)




# ref_DOM_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/hangtianDOM.tif"
# ref_DSM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DSM.tif"
# ref_npy_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/feicuiwan_mix.npy"
# pose_data = [113.00247413532685, 28.291675168473542, 178.77944191824645, 0.7132644641713202, -0.5006934497423646, 159.85591462674637]
# name ="domhigh_dsmlow"
# geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
# query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
# data = generate_ref_map(ref_DSM_path, pose_data, ref_npy_path, geotransform, query_intrinsics_dict, query_prior_poses_dict, name, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path)



# ref_DOM_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/fcw_hangtian_DOM.tif"
# ref_DSM_path = "/home/amax/Documents/datasets/DSM/0.5/0.5/hangtianDSM.tif"
# ref_npy_path = "/home/amax/Documents/datasets/DSM/feicuiwan/feicuiwan/feicuiwan_mix.npy"
# pose_data = [113.00254345991806, 28.29174605895288, 180.35907868668437, -0.37969635859719286, 2.065958183651991, 160.03585508492833]
# name ="domlow_dsmhigh"
# geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
# query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
# data = generate_ref_map(ref_DSM_path, pose_data, ref_npy_path, geotransform, query_intrinsics_dict, query_prior_poses_dict, name, area_minZ, dsm_data, dsm_trans, dom_data, ref_rgb_path, ref_depth_path)


