import os

# --- 配置部分 ---
input_file = "/home/amax/Documents/datasets/poses/DJI_20250612194622_0018_V.txt"      # 输入文件名
output_file = "/home/amax/Documents/datasets/poses/test_0018_V.txt"    # 输出文件名
# ----------------

def process_pose_file(in_path, out_path):
    # 检查输入文件是否存在
    if not os.path.exists(in_path):
        print(f"错误：找不到文件 '{in_path}'，请确保文件在当前目录下。")
        return

    print(f"正在读取 {in_path} ...")
    
    count = 0
    with open(in_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # 去除首尾空白符并分割
            parts = line.strip().split()
            
            # 确保行不为空且有足够的数据列
            # 格式: Name X Y Z Roll Pitch Yaw (共7列)
            if len(parts) >= 7:
                # 修改 Pitch (第6列，索引为5)
                parts[5] = "0.0"
                # yaw_value = float(parts[6])
                # parts[6] = str(-yaw_value)
                
                # 重新组合并写入，加上换行符
                new_line = " ".join(parts) + "\n"
                f_out.write(new_line)
                count += 1
            else:
                # 如果遇到空行或格式不对的行，可以选择原样写入或跳过
                # 这里选择原样写入以防丢失信息
                f_out.write(line)

    print("-" * 30)
    print(f"处理完成！")
    print(f"共修改了 {count} 行数据。")
    print(f"结果已保存至: {out_path}")

# 执行函数
if __name__ == "__main__":
    process_pose_file(input_file, output_file)