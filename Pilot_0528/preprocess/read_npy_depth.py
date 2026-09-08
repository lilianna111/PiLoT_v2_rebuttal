import argparse
import os
import numpy as np
import re  # 1. 导入正则模块

def natural_keys(text):
    """
    用于自然排序的辅助函数 (Natural Sort).
    将字符串拆分为文本和数字列表，以便正确排序 (例如: 1, 2, 10 而不是 1, 10, 2).
    """
    # 将字符串拆分成数字和非数字块
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def iter_npy_files(folder_path):
    # 收集所有文件的完整路径，以便进行全局排序（如果文件在不同子文件夹中）
    # 或者保持目前的逐个文件夹排序。这里优化为当前文件夹内的自然排序。
    for root, _, files in os.walk(folder_path):
        # 2. 使用 key=natural_keys 进行排序
        for name in sorted(files, key=natural_keys):
            if name.lower().endswith(".npy"):
                yield os.path.join(root, name)

def load_depth_npy(npy_path):
    depth = np.load(npy_path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth

def read_center_depth(depth, width=1920, height=1080):
    expected_center_x = width // 2
    expected_center_y = height // 2
    h, w = depth.shape[:2]
    if (h, w) != (height, width):
        center_x = w // 2
        center_y = h // 2
    else:
        center_x = expected_center_x
        center_y = expected_center_y
    return depth[center_y, center_x], (center_x, center_y), (w, h)

def main():
    parser = argparse.ArgumentParser(description="Read center depth from all .npy files in a folder.")
    parser.add_argument("--root", default="/media/amax/AE0E2AFD0E2ABE69/datasets/renders", help="Folder containing depth .npy files")
    parser.add_argument("--width", type=int, default=1920, help="Expected image width")
    parser.add_argument("--height", type=int, default=1080, help="Expected image height")
    parser.add_argument("--output-dir", default="/media/amax/AE0E2AFD0E2ABE69/datasets/depth_1", help="Directory to save results")
    args = parser.parse_args()

    folder = os.path.join(args.root, "DJI_20250612194903_0021_V")
    output_path = os.path.join(args.output_dir, "DJI_20250612194903_0021_V.txt")

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # 这里会将生成器转换为列表
    npy_files = list(iter_npy_files(folder))
    
    # 如果您需要跨子文件夹的全局排序，可以在这里再排一次:
    # npy_files.sort(key=natural_keys) 
    
    if not npy_files:
        print("No .npy files found.")
        return


    with open(output_path, "w", encoding="utf-8") as output_file:
        for npy_path in npy_files:
            try:
                depth = load_depth_npy(npy_path)
                value, center_xy, shape_xy = read_center_depth(
                    depth, width=args.width, height=args.height
                )
                output_file.write(
                    f"{npy_path}\t{value}\n"
                )
                print(f"{npy_path}\tcenter={center_xy}\tshape={shape_xy}\tdepth={value}")
            except Exception as exc:
                output_file.write(f"{npy_path}\tERROR\t{exc}\n")
                print(f"{npy_path}\tERROR: {exc}")

if __name__ == "__main__":
    main()