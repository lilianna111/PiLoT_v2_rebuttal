import os
import numpy as np

def generate_depth_txt(source_folder, output_txt_path, center_row=270, center_col=480, output_ext='.jpg', image_folder=None):
    """
    读取文件夹中所有 npy 文件的中心点深度，并保存到 txt 文件中。
    
    格式: filename depth_value
    - 若提供 image_folder：在图片目录中按 npy 的 basename 前缀匹配完整图片名（如 1671606440.005628312.jpg）
    - 否则：filename = npy 同名 + output_ext
    支持浮点数坐标（会四舍五入到最近的整数索引）
    """
    
    # 如果是浮点数坐标，转换为整数索引
    center_row = int(round(center_row))
    center_col = int(round(center_col))
    
    # 1. 检查源文件夹
    if not os.path.exists(source_folder):
        print(f"❌ 错误: 文件夹不存在 - {source_folder}")
        return

    # 若提供图片目录，预先列出所有以 output_ext 结尾的文件，用于按前缀匹配完整名（如 1671606440.005628312.jpg）
    image_name_by_prefix = {}
    if image_folder and os.path.exists(image_folder):
        for f in os.listdir(image_folder):
            if f.endswith(output_ext):
                base = os.path.splitext(f)[0]
                prefix = base.split('.')[0] if '.' in base else base
                # 同一前缀多张图时，保留 basename 更长的（带小数部分的完整名）
                if prefix not in image_name_by_prefix or len(base) > len(os.path.splitext(image_name_by_prefix[prefix])[0]):
                    image_name_by_prefix[prefix] = f

    # 2. 获取所有 .npy 文件并排序
    # 过滤出 .npy 文件
    files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]
    
    # 关键步骤：按数字顺序排序 (例如确保 10_0.npy 在 2_0.npy 后面)
    # 假设文件名格式为 "数字_后缀.npy"，如 "0_0.npy"
    try:
        files.sort(key=lambda x: int(x.split('_')[0]))
    except:
        print("⚠️ 文件名不包含数字，将使用默认字母排序")
        files.sort()

    print(f"📂 找到 {len(files)} 个文件，准备处理...")

    valid_count = 0
    
    # 3. 打开 txt 文件准备写入
    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
        for filename in files:
            file_path = os.path.join(source_folder, filename)
            
            try:
                # 加载数据
                data = np.load(file_path)
                
                # 处理维度 (C, H, W) -> (H, W)
                if data.ndim == 3:
                    data = data.squeeze(0)
                
                h, w = data.shape
                
                # 检查索引是否越界
                # 输出用完整名字：若提供了 image_folder 则按前缀匹配完整图片名（如 1671606440.005628312.jpg），否则 npy 同名 + output_ext
                base = os.path.splitext(filename)[0]
                npy_prefix = base.split('.')[0] if '.' in base else base
                out_name = image_name_by_prefix.get(npy_prefix) if image_name_by_prefix else None
                if out_name is None:
                    out_name = base + output_ext

                if 0 <= center_row < h and 0 <= center_col < w:
                    # 读取深度值
                    depth_val = float(data[center_row, center_col])
                    
                    # 写入一行: 完整文件名 深度值
                    # 使用 .4f 保留4位小数
                    line = f"{out_name} {depth_val:.4f}\n"
                    f_out.write(line)
                    
                    valid_count += 1
                else:
                    # 如果越界，写入 -1 或者其他标记
                    print(f"⚠️ {filename}: 索引越界 ({h}x{w})")
                    f_out.write(f"{out_name} -1.0\n")

            except Exception as e:
                print(f"❌ 读取 {filename} 失败: {e}")
                f_out.write(f"{out_name} error\n")

    print("="*30)
    print(f"✅ 处理完成！")
    print(f"📄 结果已保存至: {output_txt_path}")
    print(f"📊 成功提取: {valid_count}/{len(files)}")
    print("="*30)

if __name__ == "__main__":
    # ================= 配置区域 =================
    targets = [
        "interval1_HKairport01",
        # "interval1_HKairport02",
        # "interval1_HKairport03"
     ]
    for target in targets:
        NPY_FOLDER = f"/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/renders/{target}"
        # 图片目录：用于按 npy 前缀匹配完整图片名（如 1671606440.005628312.jpg），不填则用 npy 同名 .jpg
        IMAGE_FOLDER = f"/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/images/{target}"
        OUTPUT_TXT = f"/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/depth/{target}.txt"
        ROW_IDX = 306
        COL_IDX = 256
        generate_depth_txt(NPY_FOLDER, OUTPUT_TXT, ROW_IDX, COL_IDX, image_folder=IMAGE_FOLDER)