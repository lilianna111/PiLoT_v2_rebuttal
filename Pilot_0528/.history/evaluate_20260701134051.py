# from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate, evaluate_all


# # gt_pose = "/home/amax/Documents/datasets/poses/DJI_20251221132259_0002_V.txt"
# # # estimated_pose = "/home/amax/Documents/datasets/outputs/FPVLoc/DJI_20251221132259_0002_V.txt"
# # estimated_pose_low = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_l/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_high = "/home/amax/Documents/datasets/outputs/FPVLoc_l/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_low = "/home/amax/Documents/datasets/outputs/FPVLoc/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_high_dom_low_dsm = "/home/amax/Documents/datasets/outputs/FPVLoc_domhigh/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_low_dom_high_dsm = "/home/amax/Documents/datasets/outputs/FPVLoc_dsmhigh/DJI_20251221132259_0002_V.txt"
# # # print("--------------------------------ya结果--------------------------------")
# # # evaluate(estimated_pose, gt_pose)
# # print("--------------------------------高精度结果--------------------------------")
# # evaluate(estimated_pose_high, gt_pose)
# # print("--------------------------------低精度结果--------------------------------")
# # evaluate(estimated_pose_low, gt_pose)
# # print("--------------------------------crop高精度结果--------------------------------")
# # evaluate(estimated_pose_crop_high, gt_pose)
# # print("--------------------------------crop低精度结果--------------------------------")
# # evaluate(estimated_pose_crop_low, gt_pose)
# # print("--------------------------------crop高精度DOM+低精度DSM结果--------------------------------")
# # evaluate(estimated_pose_crop_high_dom_low_dsm, gt_pose)
# # print("--------------------------------crop低精度DOM+高精度DSM结果--------------------------------")
# # evaluate(estimated_pose_crop_low_dom_high_dsm, gt_pose)

# gt_pose = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_low/DJI_20251221131951_0001_V.txt"
# estimated_pose = "/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221131951_0001_V.txt"
# # evaluate_XYZ_EULER(estimated_pose, gt_pose)
# print("crop")
# evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20250612194622_0018_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth_angle/DJI_20250612194622_0018_V.txt")
# # evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132451_0003_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132451_0003_V.txt")
# # evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132525_0004_V.txt")
# # # print("depth")
# # evaluate_all("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132525_0004_V.txt")

# #DJI_20251221132259_0002_V



from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate
import os

target_names = [
  "DJI_20250612182732_0001_V",
  # "DJI_20250612174308_0001_V",
  # "DJI_20250612182017_0001_V",
  "DJI_20250612183852_0005_V",
  "DJI_20250612194903_0021_V",
  "DJI_20250612193704_0010_V",
  # "DJI_20250612193930_0012_V",
  "DJI_20250612194150_0014_V",
  "DJI_20250612194622_0018_V",
]

for target_name in target_names:
    if os.path.exists(f"/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_0.3/{target_name}.txt"):
        print(f"--------------{target_name}------------------")
        evaluate(f"/media/amax/AE0E2AFD0E2ABE69/datasets/poses/{target_name}.txt", f"/media/amax/PS2000/ral/crop_0629_1/{target_name}.txt")
    # else:
    #     print(f"--------------{target_name} not found------------------")

# print("--------------{target_name}------------------")
# evaluate_XYZ_EULER(f"/media/amax/AE0E2AFD0E2ABE69/datasets/poses/{target_name}.txt", f"/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_dem/{target_name}.txt")
#   print("--------------经纬加噪声3m，yaw加3deg------------------")
#   evaluate_XYZ_EULER(f"/media/amax/AE0E2AFD0E2ABE69/datasets/poses/{target_name}.txt", f"/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/{target_name}_noise.txt")

# print("--------------estimated------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/DJI_20251221132525_0004_V.txt")
# print("--------------经纬加噪声3m，yaw加3deg------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/DJI_20251221132525_0004_V_noise.txt")
# print("--------------prior------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20250612194622_0018_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_high/DJI_20250612194622_0018_V_prior.txt")

