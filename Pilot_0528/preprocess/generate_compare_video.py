

import cv2
import os

# ================= 配置区域 =================
# 输入文件夹路径
QUERY_DIR = '/media/amax/AE0E2AFD0E2ABE69/datasets/images/DJI_20251221132525_0004_V'  # 3840x2160 的图片文件夹
REF_DIR = '/media/amax/AE0E2AFD0E2ABE69/datasets/renders/DJI_20251221132525_0004_V'      # 1920x1080 的图片文件夹

# 输出视频文件名
OUTPUT_VIDEO = '/media/amax/AE0E2AFD0E2ABE69/datasets/videos/outputs/DJI_20251221132525_0004_V_rtk.mp4'
TARGET_W = 512
TARGET_H = 288
FPS = 30
BOX_W = 256 
BOX_H = 144
ALPHA = 1.0 
DRAW_BORDER = False
# ===========================================

def get_frame_number(filename):
    """
    针对 0_0.png, 10_0.png 这种格式的专用排序工具。
    提取下划线前面的数字作为排序依据。
    """
    try:
        # 分割字符串，例如 "10_0.png" -> ["10", "0.png"]
        # 取第一部分转为整数
        return int(filename.split('_')[0])
    except:
        # 如果遇到不符合规则的文件名，返回一个很大的数排到最后，避免报错
        return 99999999

def process_images():
    # 1. 读取文件
    query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ref_files = [f for f in os.listdir(REF_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 2. 【核心修复】使用下划线前的数字进行排序
    query_files.sort(key=get_frame_number)
    ref_files.sort(key=get_frame_number)

    # 3. 检查数量
    count = min(len(query_files), len(ref_files))
    if count == 0:
        print("错误：文件夹为空或未找到图片。")
        return

    # 4. 打印前几张确认顺序 (这一步很重要，运行后请看控制台)
    print("\n====== 文件排序检查 (确保数字是连续的) ======")
    print(f"Query 前3张: {query_files[:3]}")
    print(f"Ref   前3张: {ref_files[:3]}")
    print(f"如果不连续 (例如 0, 10, 1...)，请检查文件名格式。")
    print("===========================================\n")

    # 初始化视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (TARGET_W, TARGET_H))

    # 计算坐标
    start_x = (TARGET_W - BOX_W) // 2
    start_y = (TARGET_H - BOX_H) // 2
    end_x = start_x + BOX_W
    end_y = start_y + BOX_H

    for i in range(count):
        q_path = os.path.join(QUERY_DIR, query_files[i])
        r_path = os.path.join(REF_DIR, ref_files[i])

        img_q = cv2.imread(q_path)
        img_r = cv2.imread(r_path)

        if img_q is None or img_r is None:
            continue

        # 统一缩放到 960x540
        img_q_resized = cv2.resize(img_q, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        img_r_resized = cv2.resize(img_r, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        # 制作底图
        final_frame = img_q_resized.copy()

        # 提取中心区域
        roi_bg = final_frame[start_y:end_y, start_x:end_x]
        roi_fg = img_r_resized[start_y:end_y, start_x:end_x]

        # 混合
        blended_roi = cv2.addWeighted(roi_fg, ALPHA, roi_bg, 1 - ALPHA, 0)
        final_frame[start_y:end_y, start_x:end_x] = blended_roi

        # 画框
        if DRAW_BORDER:
            cv2.rectangle(final_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

        out.write(final_frame)
        
        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"已处理: {i + 1}/{count} ...")

    out.release()
    print("--------------------------------")
    print(f"处理完成！请查看: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_images()


# 【关键设置】偏移量
# 1  = Ref 列表去掉前 1 张 (Query:0 对应 Ref:1)
# 2  = Ref 列表去掉前 2 张 (Query:0 对应 Ref:2)
# -1 = Query 列表去掉前 1 张 (Query:1 对应 Ref:0)
# OFFSET = 0 
# # ===========================================

# def get_frame_number(filename):
#     """
#     【修改点2：通用排序函数】
#     既能处理 '0_0.png' (Query)，也能处理 '1.png' (Ref)
#     """
#     try:
#         # 1. 先去掉后缀: '0_0.png' -> '0_0', '1.png' -> '1'
#         name_no_ext = os.path.splitext(filename)[0]
        
#         # 2. 再按 '_' 切割取第一部分
#         # 如果是 '0_0' -> split('_') -> ['0', '0'] -> 取 [0] 也就是 '0'
#         # 如果是 '1'   -> split('_') -> ['1']      -> 取 [0] 也就是 '1'
#         return int(name_no_ext.split('_')[0])
#     except:
#         return 99999999

# def process_images():
#     # 1. 读取文件
#     query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     ref_files = [f for f in os.listdir(REF_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     # 2. 排序
#     query_files.sort(key=get_frame_number)
#     ref_files.sort(key=get_frame_number)

#     print("------ 初始文件检查 (前2张) ------")
#     print(f"Query: {query_files[:2]}") # 应该是 0_0.png, 1_0.png
#     print(f"Ref  : {ref_files[:2]}")   # 应该是 0.png, 1.png (如果文件夹里有0的话)

#     # 3. 应用偏移量
#     if OFFSET > 0:
#         print(f"\n[操作] Ref 列表扔掉前 {OFFSET} 张 (为了让 Query:0 对齐 Ref:1)")
#         ref_files = ref_files[OFFSET:]
#     elif OFFSET < 0:
#         abs_offset = abs(OFFSET)
#         print(f"\n[操作] Query 列表扔掉前 {abs_offset} 张")
#         query_files = query_files[abs_offset:]
    
#     # 4. 再次取最小值
#     count = min(len(query_files), len(ref_files))
    
#     if count == 0:
#         print("错误：对齐后没有剩余图片。")
#         return

#     # 5. 【关键确认】
#     print("\n====== 最终配对情况 (请仔细核对!) ======")
#     print(f"Query[0]: {query_files[0]}  <==对应==>  Ref[0]: {ref_files[0]}")
#     if count > 1:
#         print(f"Query[1]: {query_files[1]}  <==对应==>  Ref[1]: {ref_files[1]}")
#     print("===========================================\n")

#     # 初始化视频
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (TARGET_W, TARGET_H))

#     start_x = (TARGET_W - BOX_W) // 2
#     start_y = (TARGET_H - BOX_H) // 2
#     end_x = start_x + BOX_W
#     end_y = start_y + BOX_H

#     for i in range(count):
#         q_path = os.path.join(QUERY_DIR, query_files[i])
#         r_path = os.path.join(REF_DIR, ref_files[i])

#         img_q = cv2.imread(q_path)
#         img_r = cv2.imread(r_path)

#         if img_q is None or img_r is None:
#             continue

#         img_q_resized = cv2.resize(img_q, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
#         img_r_resized = cv2.resize(img_r, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

#         final_frame = img_q_resized.copy()

#         roi_bg = final_frame[start_y:end_y, start_x:end_x]
#         roi_fg = img_r_resized[start_y:end_y, start_x:end_x]

#         blended_roi = cv2.addWeighted(roi_fg, ALPHA, roi_bg, 1 - ALPHA, 0)
#         final_frame[start_y:end_y, start_x:end_x] = blended_roi

#         if DRAW_BORDER:
#             cv2.rectangle(final_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

#         out.write(final_frame)
        
#         if (i + 1) % 50 == 0:
#             print(f"已处理: {i + 1}/{count}")

#     out.release()
#     print(f"完成！视频已保存: {OUTPUT_VIDEO}")

# if __name__ == "__main__":
#     process_images()