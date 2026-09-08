import rasterio

def get_height_from_dsm(dsm_path, x_geo, y_geo):
    """
    输入：DSM文件路径, X坐标, Y坐标
    输出：该位置的高度值 (Z)
    """
    with rasterio.open(dsm_path) as src:
        # 1. 检查坐标是否在图片范围内 (Bounds Check)
        if not (src.bounds.left <= x_geo <= src.bounds.right and 
                src.bounds.bottom <= y_geo <= src.bounds.top):
            return "Error: 坐标超出DSM范围"

        # 2. 核心步骤：将地理坐标 (x, y) 转换为 矩阵行列号 (row, col)
        # index() 方法会自动处理 GeoTransform 里的公式
        row, col = src.index(x_geo, y_geo)

        # 3. 读取该位置的像素值
        # 注意：dsm可能只有一个波段(band 1)
        z_value = src.read(1)[row, col]

        # 4. 检查是否是无效值 (NoData Value)
        # 很多DSM用 -9999 代表无效区域
        if z_value == src.nodata:
            return "No Data (无效区域)"
            
        return z_value

# --- 使用示例 ---
dsm_file = "/home/amax/Documents/datasets/DSM/feicuiwan/fcw_hangtian_DSM.tif"  # 你的DSM文件路径
with rasterio.open(dsm_file) as src:
    print(f"1. 坐标系 (CRS): {src.crs}")
    print("-" * 30)
    print(f"2. 边界范围 (Bounds):")
    print(f"   最左 (Left/MinX): {src.bounds.left}")
    print(f"   最下 (Bottom/MinY): {src.bounds.bottom}")
    print(f"   最右 (Right/MaxX): {src.bounds.right}")
    print(f"   最上 (Top/MaxY):    {src.bounds.top}")
    print("-" * 30)
target_x = 401772.436    # 你的 X 坐标
target_y = 3131148.02    # 你的 Y 坐标

height = get_height_from_dsm(dsm_file, target_x, target_y)
print(f"坐标 ({target_x}, {target_y}) 的高度是: {height}")