import cv2

import re

import math

import os

# ==== 配置 ====

video_file = '/home/amax/Documents/datasets/videos/DJI_20251221131951_0001_V.mp4'  # 视频

srt_file   = '/home/amax/Documents/datasets/poses/DJI_20251221131951_0001_V.srt'  # SRT

output_dir = '/home/amax/Documents/datasets/images/DJI_20251221131951_0001_V'  # 抽帧保存目录

output_txt = '/home/amax/Documents/datasets/poses/DJI_20251221131951_0001_V.txt'  # 位姿 TXT

# ==== 时间段和帧率 ====

start_sec = 10

fps = 29.97

target_frame_count = 700

start_frame = math.floor(start_sec * fps)

# ==== 解析 SRT 并收集位姿 ====

pose_dict = {}

with open(srt_file, 'r', encoding='utf-8') as f:

  lines = f.readlines()

for i in range(len(lines)):

  line = lines[i]

  if line.startswith("FrameCnt:"):

    frame_id_match = re.search(r"FrameCnt: (\d+)", line)

    if not frame_id_match:

      continue

    frame_id = int(frame_id_match.group(1))

    info_line = lines[i + 1] if i + 1 < len(lines) else ""

    lat_match = re.search(r"latitude:\s*([-\d.]+)", info_line)

    lon_match = re.search(r"longitude:\s*([-\d.]+)", info_line)

    alt_match = re.search(r"abs_alt:\s*([-\d.]+)", info_line)

    roll_match  = re.search(r"gb_roll:\s*([-\d.]+)", info_line)

    pitch_match = re.search(r"gb_pitch:\s*([-\d.]+)", info_line)

    yaw_match   = re.search(r"gb_yaw:\s*([-\d.]+)", info_line)

    if all([lat_match, lon_match, alt_match, roll_match, pitch_match, yaw_match]):

      lat = float(lat_match.group(1))

      lon = float(lon_match.group(1))

      alt = float(alt_match.group(1))

      roll  = float(roll_match.group(1))

      pitch = float(pitch_match.group(1)) + 90

      yaw   = -float(yaw_match.group(1))

      pose_dict[frame_id] = [lon, lat, alt, roll, pitch, yaw]

# ==== 筛选抽帧 ID ====

sorted_ids = sorted(fid for fid in pose_dict if fid >= start_frame)

if len(sorted_ids) < target_frame_count:

  print(f"❌ 错误：从 frame {start_frame} 开始只找到 {len(sorted_ids)} 个有效帧，无法满足 {target_frame_count} 帧")

  exit()

selected_ids = sorted_ids[:target_frame_count]

# ==== 打开视频 ====

cap = cv2.VideoCapture(video_file)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频总帧数: {total_frames}")

os.makedirs(output_dir, exist_ok=True)

# ==== 遍历保存 ====

with open(output_txt, 'w') as f:

  for i, fid in enumerate(selected_ids):

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

    ret, frame = cap.read()

    if not ret:

      print(f"⚠️ 警告: 读取第 {fid} 帧失败，跳过")

      continue

    img_name = f"{i}_0.png"

    cv2.imwrite(os.path.join(output_dir, img_name), frame)

    values = pose_dict[fid]

    f.write(f"{img_name} {' '.join(map(str, values))}\n")

print(f"✅ 成功抽取 {len(selected_ids)} 帧，图像保存至 {output_dir}，位姿保存至 {output_txt}")