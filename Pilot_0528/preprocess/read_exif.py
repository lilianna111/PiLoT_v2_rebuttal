import exifread
from pyexiv2 import Image
import os
def get_dji_exif(exif_file):
    # 打开exif文件
    with open(exif_file, "rb") as f:
        # 读取exif信息
        exif_data = exifread.process_file(f)
        img = Image(exif_file)
        all_info_xmp = img.read_xmp()
        # 获取GPS对应的标签
        gps_latitude_tag = "GPS GPSLatitude"
        gps_latitude_ref_tag = "GPS GPSLatitudeRef"
        gps_longitude_tag = "GPS GPSLongitude"
        gps_longitude_ref_tag = "GPS GPSLongitudeRef"
        gps_altitude_tag = "GPS GPSAltitude"
        yaw_tag = "Xmp.drone-dji.GimbalYawDegree"
        pitch_tag = "Xmp.drone-dji.GimbalPitchDegree"
        roll_tag = "Xmp.drone-dji.GimbalRollDegree"
        focal_tag = "EXIF FocalLength"
        Width_tag = "EXIF ExifImageWidth"
        Height_tag = "EXIF ExifImageLength"
        img_type = exif_file.split('/')[-1][-5]
        create_time_tag = "EXIF DateTimeOriginal"
        if gps_latitude_tag in exif_data and gps_latitude_ref_tag in exif_data and gps_longitude_tag in exif_data and gps_longitude_ref_tag in exif_data:
        
            # 获取GPS纬度和经度的分数值和方向值
            gps_latitude_value = exif_data[gps_latitude_tag].values
            gps_latitude_ref_value = exif_data[gps_latitude_ref_tag].values
            gps_longitude_value = exif_data[gps_longitude_tag].values
            gps_longitude_ref_value = exif_data[gps_longitude_ref_tag].values
            gps_altitude_value = exif_data[gps_altitude_tag].values
            # 将GPS纬度和经度的分数值转换为浮点数值
            gps_latitude = (float(gps_latitude_value[0].num) / float(gps_latitude_value[0].den) +
                            (float(gps_latitude_value[1].num) / float(gps_latitude_value[1].den)) / 60.0 +
                            (float(gps_latitude_value[2].num) / float(gps_latitude_value[2].den)) / 3600.0)
            gps_longitude = (float(gps_longitude_value[0].num) / float(gps_longitude_value[0].den) +
                             (float(gps_longitude_value[1].num) / float(gps_longitude_value[1].den)) / 60.0 +
                             (float(gps_longitude_value[2].num) / float(gps_longitude_value[2].den)) / 3600.0)
            gps_altitude = eval(str(gps_altitude_value[0]).split('/')[0]) / eval(str(gps_altitude_value[0]).split('/')[1])
            
            
            roll_value = eval(all_info_xmp[roll_tag])
            pitch_value = eval(all_info_xmp[pitch_tag])
            yaw_value = eval(all_info_xmp[yaw_tag])
            # create_time
            create_time = exif_data[create_time_tag].values
            # intrinsic  focal_value
            focal_value = float(exif_data["EXIF FocalLength"].values[0])
            print("focal_value: " , focal_value)
            width = float(exif_data[Width_tag].values[0])
            height = float(exif_data[Height_tag].values[0])
            # 根据GPS纬度和经度的方向值，判断正负号
            if gps_latitude_ref_value != "N":
                gps_latitude = -gps_latitude
            if gps_longitude_ref_value != "E":
                gps_longitude = -gps_longitude
            # 返回这些值
            print(pitch_value, roll_value, -yaw_value, gps_longitude, gps_latitude, gps_altitude)
            return roll_value, pitch_value+90, -yaw_value, gps_latitude, gps_longitude, gps_altitude
            
            # return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, focal_value, width, height, img_type, create_time
        else:
            # 如果不存在这些标签，返回None
            return None
img_folder = "/media/amax/PortableSSD/UrbanScene3D/PolyTech/Zhang/PolyTech_coarse/"
img_list = os.listdir(img_folder)
save_txt = "/media/liuxy/PortableSSD/UrbanScene3D/PolyTech/Zhang et al/PolyTech_coarse_exif.txt"
with open(save_txt, 'w') as f:
    for img_name in img_list:
        img_path = os.path.join(img_folder, img_name)
        roll, pitch, yaw, lat, lon, alt= get_dji_exif(img_path)
        f.write(f"{img_name} {lon} {lat} {alt} {roll} {pitch} {yaw}\n")
