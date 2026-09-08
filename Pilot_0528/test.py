from pyproj import Transformer

# 定义转换器
# 源坐标系: EPSG:4326 (WGS84 经纬度)
# 目标坐标系: EPSG:4547 (CGCS2000 / 3-degree Gauss-Kruger CM 114E)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)

# 输入经纬度 (注意顺序: 经度, 纬度)
lon = 113.00249722780813
lat = 28.29167703629333

# 进行转换
x, y = transformer.transform(lon, lat)

print(f"输入 WGS84 坐标: ({lon}, {lat})")
print(f"转换后 EPSG:4547 坐标:")
print(f"X (Easting) : {x:.9f}")
print(f"Y (Northing): {y:.9f}")