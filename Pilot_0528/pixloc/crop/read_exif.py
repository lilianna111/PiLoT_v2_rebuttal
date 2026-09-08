import exifread
import pyexif
import os
import glob
import numpy as np

def get_dji_exif(exif_file, return_focal_lens=True):
    """
    从DJI图像文件中提取EXIF信息
    
    Args:
        exif_file (str): 图像文件路径
        return_focal_lens (bool): 是否返回焦距信息
        
    Returns:
        tuple: (roll, pitch, yaw, lat, lon, alt, [focal]) 或 None
    """
    try:
        # 打开exif文件
        with open(exif_file, "rb") as f:
            # 读取exif信息
            exif_data = exifread.process_file(f)
            
        # 读取XMP信息
        try:
            img = pyexif.ExifEditor(exif_file)
            all_info_xmp = img.getDictTags()
        except Exception as e:
            print(f"Warning: Cannot read XMP data from {exif_file}: {e}")
            return None
        
        # 获取GPS对应的标签
        gps_latitude_tag = "GPS GPSLatitude"
        gps_latitude_ref_tag = "GPS GPSLatitudeRef"
        gps_longitude_tag = "GPS GPSLongitude"
        gps_longitude_ref_tag = "GPS GPSLongitudeRef"
        gps_altitude_tag = "GPS GPSAltitude"
        
        # 云台角度标签
        yaw_tag = "GimbalYawDegree"
        pitch_tag = "GimbalPitchDegree"
        roll_tag = "GimbalRollDegree"
        
        # 其他标签
        focal_tag = "EXIF FocalLength"
        width_tag = "EXIF ExifImageWidth"
        height_tag = "EXIF ExifImageLength"
        create_time_tag = "EXIF DateTimeOriginal"
        
        # 检查必要的GPS信息是否存在
        required_gps_tags = [gps_latitude_tag, gps_latitude_ref_tag, 
                           gps_longitude_tag, gps_longitude_ref_tag, gps_altitude_tag]
        
        if not all(tag in exif_data for tag in required_gps_tags):
            print(f"Warning: Missing GPS tags in {exif_file}")
            return None
            
        # 检查云台角度信息是否存在
        required_gimbal_tags = [yaw_tag, pitch_tag, roll_tag]
        if not all(tag in all_info_xmp for tag in required_gimbal_tags):
            print(f"Warning: Missing gimbal angle tags in {exif_file}")
            return None
        
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
        
        # 处理高度信息
        altitude_str = str(gps_altitude_value[0])
        if '/' in altitude_str:
            parts = altitude_str.split('/')
            gps_altitude = float(parts[0]) / float(parts[1])
        else:
            gps_altitude = float(altitude_str)
        
        # 获取云台角度信息
        roll_value = float(all_info_xmp[roll_tag])
        pitch_value = float(all_info_xmp[pitch_tag])  # 修复：确保转换为float
        yaw_value = float(all_info_xmp[yaw_tag])
        
        # 根据GPS纬度和经度的方向值，判断正负号
        if gps_latitude_ref_value != "N":
            gps_latitude = -gps_latitude
        if gps_longitude_ref_value != "E":
            gps_longitude = -gps_longitude
        
        ImgHeight = all_info_xmp['ExifImageHeight']
        ImgWidth = all_info_xmp['ExifImageWidth']
        
        if return_focal_lens:
            if focal_tag in exif_data:
                focal_value = float(exif_data[focal_tag].values[0])
                return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, focal_value, ImgWidth, ImgHeight
            else:
                print(f"Warning: Focal length not found in {exif_file}")
                return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, None
        
        # 返回这些值
        return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude
        
    except Exception as e:
        print(f"Error reading EXIF data from {exif_file}: {e}")
        return None


def read_exif_data(folder_path):
    """
    读取文件夹中所有DJI图像的EXIF信息并返回字典
    
    Args:
        folder_path (str): 图像文件夹路径
        
    Returns:
        dict: 以文件名为键，EXIF信息为值的字典
    """
    exif_dict = {}
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return exif_dict
    
    # 支持多种图像格式，不区分大小写
    image_extensions = ['*.JPG']
    # ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    img_list = []
    
    for extension in image_extensions:
        # 使用os.path.join确保跨平台兼容性
        pattern = os.path.join(folder_path, extension)
        img_list.extend(glob.glob(pattern))
    
    # 排序文件列表
    img_list = sorted(img_list)
    
    print(f"Found {len(img_list)} images in {folder_path}")
    
    for image_path in img_list:
        try:
            # 获取文件名（跨平台兼容）
            filename = os.path.basename(image_path)
            filename = filename.split('.')[0]
            
            # 获取EXIF信息
            result = get_dji_exif(image_path)
            
            # 保存exif信息到字典中
            if result:
                # 将结果存储为结构化字典
                roll, pitch, yaw, lat, lon, alt, focal_value, img_w, img_h = result
                exif_dict[filename] = {
                    'roll': roll,
                    'pitch': pitch, 
                    'yaw': yaw,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'focal_value': focal_value,
                    'img_w':img_w,
                    'img_h':img_h,
                    'file_path': image_path
                }
                print(f"{filename}{focal_value},pitch:{pitch}")
                # print(f"Successfully processed: {filename}")
            else:
                print(f"Failed to extract EXIF data from: {filename}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return exif_dict
