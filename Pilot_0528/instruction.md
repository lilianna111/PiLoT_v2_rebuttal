1. DJI视频提取得到srt文件
    - /home/amax/Documents/code/PiLoT/PiLoT/preprocess/srt2txt.py     
    - https://djitelemetryoverlay.com/subtitle-extractor/#   
2. 误差图可视化   /home/ubuntu/Documents/code/github/FPV/PiLoT/preprocess/vis/vis_same_track/vis_overlap_track.py
3. 特征图可视化（需要pt文件）
    - /home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/preprocess/vis_feature_seeds_v3_hand_onfeat.py
        - 修改seq_name = "6444"
        - 新建文件夹/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs/DJI_20250804192327_0002_V/6444/，里面有pt,query,ref（-1_0作ref）图
    - 生成pt文件：/home/ubuntu/Documents/code/github/FPV/PiLoT_long_video/run.sh
        - 修改main.py的start =6444，end=6445，注意 self.img_list = self.img_list[start:end]
        - /home/ubuntu/Documents/code/github/FPV/PiLoT_long_video/pixloc/localization/base_refiner.py的torch.save(inputs, "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs/DJI_20250804192327_0002_V/6444/"+name)
4. 棋盘格可视化    
    - /home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/preprocess/vis/vis_feicuiwan/query_ref_with_error.py
5. 反投影得到3D坐标   
    - /home/amax/Documents/code/PiLoT/PiLoT/preprocess/4_verify_matches.py
6. 读npy文件
    - /home/amax/Documents/code/PiLoT/PiLoT/preprocess/read_npy.py
7. 读取文件夹中所有 npy 文件的中心点深度
    - /home/amax/Documents/code/PiLoT/PiLoT/preprocess/read_center_depth.py

