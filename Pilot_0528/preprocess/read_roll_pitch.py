def extract_pose_columns(input_file, output_file):
    """
    从原始 pose 文件中提取: 图片名, Roll, Pitch
    """
    try:
        count = 0
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                # 去除首尾空白符并按空白符(空格或Tab)分割
                parts = line.strip().split()
                
                # 确保这一行数据足够完整 (防止空行或坏数据报错)
                if len(parts) >= 6:
                    # 索引对应关系:
                    # 0: 图片名
                    # 1: 经度
                    # 2: 纬度
                    # 3: 高度
                    # 4: Roll  <-- 需要
                    # 5: Pitch <-- 需要
                    # 6: Yaw
                    
                    filename = parts[0]
                    roll = parts[4]
                    pitch = parts[5]
                    
                    # 写入新文件，中间用空格隔开
                    f_out.write(f"{filename} {roll} {pitch}\n")
                    count += 1
                    
        print(f"处理完成！已提取 {count} 行数据。")
        print(f"结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    targets = [
        "interval1_HKairport01",
        "interval1_HKairport02",
        "interval1_HKairport03"
     ]
    # 在这里修改你的文件名
    for target in targets:
        input_path = f'/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/poses/{target}.txt'        # 输入文件路径
        output_path = f'/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/angle/{target}.txt'    # 输出文件路径
    
        # 为了演示，我先创建一个包含你提供数据的临时文件
        # 实际使用时，你可以删除下面这几行写入测试数据的代码

        # 运行提取函数
        extract_pose_columns(input_path, output_path)