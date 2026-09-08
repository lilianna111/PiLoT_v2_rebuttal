# import numpy as np

# # 1. 加载文件
# file_path = '/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V/6_dsm.npy'  # 替换成你的文件路径
# data = np.load(file_path)

# # 2. 打印内容
# print("数据内容：")
# print(data)

import numpy as np

# 1. 加载数据
file_path = '/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V/6_dsm.npy'
data = np.load(file_path)

print(f"数据形状: {data.shape}") 
# 预期输出应该是类似 (1080, 1920, 3) 或 (H, W, 3) 这样的格式

# 2. 获取高度和宽度
if data.ndim == 3:
    h, w, c = data.shape
    
    # 3. 计算中心索引 (网格的正中心)
    center_row = h // 2
    center_col = w // 2
    
    # 4. 读取该位置的坐标 (X, Y, Z)
    center_xyz = data[center_row, center_col]
    
    print(f"\n--- 结果 ---")
    print(f"网格中心索引 (Row, Col): ({center_row}, {center_col})")
    print(f"该点的真实坐标 (X, Y, Z):\n{center_xyz}")
    
    # 你的数据看起来像 UTM 坐标，可以顺便分开打印方便看：
    print(f"X (Easting):  {center_xyz[0]:.2f}")
    print(f"Y (Northing): {center_xyz[1]:.2f}")
    print(f"Z (Height):   {center_xyz[2]:.2f}")

else:
    print("数据形状不符合预期，请检查是否为 (H, W, 3)")



# import numpy as np
# from pyproj import Transformer
# import os
# import time

# def convert_npy_4547_to_wgs84(npy_path, output_path=None):
#     """
#     读取包含 (X, Y, Z) 的 npy 文件，将 X, Y 从 EPSG:4547 转为 WGS84 经纬度。
    
#     Args:
#         npy_path: 输入的 .npy 文件路径
#         output_path: 输出路径，如果为 None，则在原文件名后加 _wgs84 保存
#     """
    
#     print(f"Loading data from {npy_path}...")
#     # 1. 读取数据 [H, W, 3]
#     # 通道顺序假设为: 0: Easting(X), 1: Northing(Y), 2: Elevation(Z)
#     data = np.load(npy_path)
    
#     if data.shape[-1] != 3:
#         raise ValueError(f"Input npy shape expects (H, W, 3), but got {data.shape}")

#     # 分离通道
#     x_4547 = data[:, :, 0]
#     y_4547 = data[:, :, 1]
#     z_vals = data[:, :, 2]

#     # 2. 定义坐标转换器
#     # EPSG:4547 -> 投影坐标系 (单位: 米)
#     # EPSG:4326 -> 地理坐标系 (单位: 度, WGS84)
#     # always_xy=True 确保输入输出顺序为 (x, y) 即 (经度, 纬度)，而不是 (纬度, 经度)
#     transformer = Transformer.from_crs("epsg:4547", "epsg:4326", always_xy=True)

#     print("Transforming coordinates...")
#     start_time = time.time()

#     # 3. 执行转换 (pyproj 支持直接传入 numpy 数组，速度极快)
#     lon_wgs84, lat_wgs84 = transformer.transform(x_4547, y_4547)

#     # === 处理无效值 (Optional) ===
#     # 之前的代码中，无效区域被填充为 0。
#     # 坐标 (0, 0) 在 4547 投影下转换到经纬度虽然有数学意义，但不是你的数据点。
#     # 为了保持数据的一致性，我们将原本是 0 的位置重置为 0 (或者你可以设为 np.nan)
#     mask_invalid = (x_4547 == 0) & (y_4547 == 0)
#     lon_wgs84[mask_invalid] = 0
#     lat_wgs84[mask_invalid] = 0

#     print(f"Transformation cost: {time.time() - start_time:.4f}s")

#     # 4. 重新堆叠 [Lon, Lat, Height]
#     data_wgs84 = np.stack([lon_wgs84, lat_wgs84, z_vals], axis=-1)

#     # 5. 保存
#     if output_path is None:
#         base, ext = os.path.splitext(npy_path)
#         output_path = f"{base}_wgs84{ext}"
    
#     np.save(output_path, data_wgs84)
#     print(f"Saved WGS84 data to: {output_path}")
    
#     # === 验证打印 ===
#     # 找一个中心点打印一下看看是否正常
#     cy, cx = data.shape[0] // 2, data.shape[1] // 2
#     if not mask_invalid[cy, cx]:
#         print("\n=== Sample Verification (Center Pixel) ===")
#         print(f"Original (4547):  X={x_4547[cy, cx]:.2f}, Y={y_4547[cy, cx]:.2f}")
#         print(f"Converted (WGS84): Lon={lon_wgs84[cy, cx]:.8f}, Lat={lat_wgs84[cy, cx]:.8f}")
#         print("Note: WGS84 format is (Longitude, Latitude)")

# if __name__ == "__main__":
#     # 在这里修改你的文件路径
#     input_npy = "/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V/6_dsm.npy" 
#     output_npy = "/home/amax/Documents/datasets/crop/DJI_20250612194622_0018_V/6_dsm_wgs84.npy"
    
#     # 如果你想指定输出名，可以传入第二个参数，否则自动加后缀
#     convert_npy_4547_to_wgs84(input_npy, output_npy)