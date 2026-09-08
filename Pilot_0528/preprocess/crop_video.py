import cv2
import os

def trim_video_frames(input_path, output_path, start_frame, end_frame):
    # 1. 打开视频
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"总帧数: {total_frames}, FPS: {fps}")

    # 2. 校验输入范围
    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        print(f"错误：帧数范围无效 (0 - {total_frames-1})")
        return

    # 3. 设置视频写入器
    # mp4v 是 mp4 格式常用的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 跳转到起始帧 (关键步骤，避免从头读取)
    print(f"正在跳转到第 {start_frame} 帧...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    
    print("开始裁减处理...")
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # 写入帧
        out.write(frame)
        
        # 进度显示 (每100帧打印一次)
        if (current_frame - start_frame) % 100 == 0:
            print(f"正在处理: {current_frame}/{end_frame}")

        current_frame += 1
        
        # 到达结束帧则停止
        if current_frame >= end_frame:
            break

    # 5. 释放资源
    cap.release()
    out.release()
    print(f"完成！视频已保存至: {output_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    video_path = "/home/amax/Documents/datasets/DJI_20251221132525_0004_V.mp4"   # 替换你的视频路径
    save_path = "/home/amax/Documents/datasets/videos/crop/DJI_20251221132525_0004_V.mp4" # 输出路径
    
    # 比如：从第 100 帧截取到第 500 帧
    s_frame = 0
    e_frame = 600
    
    trim_video_frames(video_path, save_path, s_frame, e_frame)