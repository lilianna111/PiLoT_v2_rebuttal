from pathlib import Path
import numpy as np
import cv2
import os
import math
import sys
sys.path.append("../build/")
from ModelRenderScene import ModelRenderScene
from tqdm import tqdm
def write_pfm(file_path, image, scale=1.0):
    """
    将浮点图像写入 PFM 文件。
      - image: (H,W) 或 (H,W,3) 的浮点 numpy 数组
      - scale: 输出文件的 scale 值，默认1.0即可
               若实际写入时检测到系统是 little-endian，会将 scale 写为负值
    """
    if image.dtype != np.float32:
        # 若不是 float32, 先转一下
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        color = False
        height, width = image.shape
    elif len(image.shape) == 3 and image.shape[2] == 3:
        color = True
        height, width, _ = image.shape
    else:
        raise ValueError("write_pfm only supports HxW or HxWx3 float images.")

    # 判断系统大小端
    import sys
    byteorder = sys.byteorder
    if byteorder == 'little':
        scale = -scale  # 小端则写负值
    mode = 'PF\n' if color else 'Pf\n'

    with open(file_path, 'wb') as f:
        # 写header
        f.write(mode.encode('utf-8'))
        f.write(f"{width} {height}\n".encode('utf-8'))
        f.write(f"{scale}\n".encode('utf-8'))
        # 如果需要 flip: image = np.flipud(image)
        image.tofile(f)
class RenderImageProcessor:
    def __init__(self, config):
        self.config = config
        self.osg_config = self.config["osg"]
        self.renderer = self._initialize_renderer()
        self.outputs = Path(self.config["output"])

    def _initialize_renderer(self):
        # Construct paths for model

        model_path = self.osg_config["model_path"]
        render_camera = np.array(self.config['render_camera']) 
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        # sensor_width, sensor_height = (render_camera[-2]), (render_camera[-1])
        return ModelRenderScene(model_path, view_width, view_height, render_camera[-2], render_camera[-1], render_camera[2], render_camera[3])
    #
    def update_pose(self, Trans, Rot, fovy = None):
        if fovy is not None:
            self.fovy = fovy
        self.renderer.updateViewPoint(Trans, Rot)
        self.renderer.nextFrame() 
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    
    def get_depth_image(self):
        depthImgMat = np.array(self.renderer.getDepthImage(), copy=False).squeeze()
        
        return depthImgMat
    
    def save_color_image(self, outputs):
        self.renderer.saveColorImage(outputs)
    
    def delay_to_load_map(self, config_web):
        self.euler_angles = config_web['euler_angles']  #相机角度
        self.translation = config_web['translation']   #位置  
        for i in range(500):
            self.update_pose(self.translation, self.euler_angles)
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    def rendering(self, config_web):
        self.image_id = config_web['image_id']
        self.euler_angles = config_web['euler_angles']  #相机角度
        self.translation = config_web['translation']   #位置  
        print(self.euler_angles, self.translation)
        for i in range(40):
            self.update_pose(self.translation, self.euler_angles)
            color_image = self.get_color_image()
            depth_image = self.get_depth_image() 
        cv2.imwrite(str(self.outputs/self.image_id), color_image)
        print(str(self.outputs/self.image_id))
        np.save(str(self.outputs/(self.image_id[:-4]+'.npy')), depth_image)
        # cv2.imwrite(str(self.outputs/"render_img"/self.image_id), np.uint8(np.around(depth_image)))
        # if not os.path.exists(self.outputs/"normalFOV_images"):
        #     os.mkdir(self.outputs/"normalFOV_images")
        
    def fov_calculate(self, f_mm, sensor_height):
        print("sensor_height: ", sensor_height, f_mm)
        fovy_radians = 2 * math.atan(sensor_height / 2 / f_mm)
        fovy_degrees = math.degrees(fovy_radians)
        
        return fovy_degrees
    
    def generate_pose(self, x, y, z, yaw, pitch, roll):
        output_path = "/home/ubuntu/Documents/code/3DTilesRender-3DTilesRender_dev/euler_gt_pose.txt"
        
        # import pdb; pdb.set_trace()
        x_list = [0]#np.arange(0, 0.05, 0.01)
        y_list = [0]#np.arange(0, 0.05, 0.01)
        z_list = np.arange(0, 1, 0.25)
        yaw_list = np.arange(-1, 1, 0.25)
        pitch_list = np.arange(-1, 1, 0.25)
        num = 0
        with open(output_path, 'w') as f:
            for i in x_list:
                for j in y_list:
                    for k in z_list:
                        for v in yaw_list:
                            for w in pitch_list:
                                infor = str(num) + '.png' + ' ' + str(pitch+w) + ' ' + \
                                    str(roll) + ' ' + str(yaw + v) + ' ' + stsensor_height(x + i) + ' ' + \
                                        str(y + j) + ' ' + str(z + k) + '\n'
                                num += 1
                                f.write(infor)
                    

              
        
if __name__ == "__main__":
    # 卫星象山
    config = {
        "osg": {
        # "model_path": "http://localhost:8088/Scene/Production_6.json"
        "model_path": "http://localhost:8082/tileset.json"
        },
        # "render_camera": [
        #     640,  # width
        #     512, #height
        #     1125, # focal length(pixel/mm)
        #     640, # sensor_width(pixel/mm)
        #     512 # sensor_height(pixel/mm)
        # ],
        #  "render_camera": [
        #     1920,
        #     1080,
        #     950.15,
        #     540.0,
        #     1920,
        #     1080,
        #     1368.55,
        #     1367
        # ],
        #  "render_camera": [
        #     3840,
        #     2160,
        #     1900.3,
        #     1080.0,
        #     3840,
        #     2160,
        #     2737.1,
        #     2734
        # ],
        # "render_camera": [
        #         3840,
        #         2160,
        #         1915.7,
        #         1075.1,
        #         3840,
        #         2160,
        #         2700.0,
        #         2700.0
                 
        # ],
        # "render_camera": [
        #         1920,
        #         1080,
        #         957.85,
        #         537.55,
        #         1920,
        #         1080,
        #         1350.0,
        #         1350.0
        # ],
        # "render_camera": [  # m4t
        #         512,
        #         288,
        #         255.426,
        #     143.3466,
        #     512,
        #     288,
        #     360.0,
        #     360.0
                 
        # ],
        # "render_camera": [  # m4t
        #         960,
        #         540,
        #         478.925,
        #     268.775,
        #     960,
        #     540,
        #     675.0,
        #     675.0
        # ],

        #  "render_camera": [  #m3p
        #         512,
        #         288,
        #         258.3733,
        #     142.373,
        #     512,
        #     288,
        #     361.17,
        #     361.17
                 
        # ],
        #  "render_camera": [   #m3p
        #         960,
        #         540,
        #         484.45,
        #     266.95,
        #     960,
        #     540,
        #     600.2,
        #     600.2
        # ],
        # "render_camera":[
        #     1536,
        #     1024,
        #     768,
        #     512,
        #     1536,
        #     1024,
        #     # 1662.337,
        #     # 1662.337,
        #     2327.27,
        #     2327.27
        # ],
        #    "render_camera": [
        #     1920,
        #     1080,
        #     960,
        #     540,
        #     1920,
        #     1080,
        #     11745,
        #     11745
                 
        # ],  
        # "render_camera": [
        #     1920,
        #     1080,
        #     960,
        #     540,
        #     1920,
        #     1080,
        #     1333.5,
        #     1333
                 
        # ],
        "render_camera": [    #
            1368,    
            912,     
            684,    
            456,
            1368,
            912,          
            912,        
            912   #844
                     
        ],

        #   "render_camera": [
        #     1600,
        #     1200,
        #     800,
        #     600,
        #     1600,
        #     1200,
        #     1931,
        #     1931
        # ],
        #   "render_camera": [
        #     1600,
        #     1200,
        #     800,
        #     600,
        #     1600,
        #     1200,
        #     1931,
        #     1931
        # ],
        #  "render_camera": [
        #     1920,
        #     1080,
        #     4.5,
        #     6.17,
        #     3.47,
        # ],
        #  "render_camera": [
        #     640,
        #     480,
        #     12,
        #     6.4,
        #     4.8
        # ],
        # "render_camera": [
        #     800,  # width
        #     600, #height
        #     1.5648, # width/height
        #     1.143, # sensor_width
        #     0.858 # sensor_height
        # ],
        #         "render_camera": [
        #     4056,  # width
        #     3040, #height,
        #     4.5, # fmm
        #     6.29, # sensor_width
        #     4.71 # sensor_height
        # ],
        "output":"/media/amax/PortableSSD/UrbanScene3D/PolyTech/Zhang/render1/PolyTech_fine/"
    }

    config_web = {
        "image_id":'1.jpg',
        "euler_angles": [],
        "translation":[],
    }
    pose_save_path = "/media/amax/PortableSSD/UrbanScene3D/PolyTech/Zhang/poses/PolyTech_fine.txt"
    renderer = RenderImageProcessor(config)

    # load map 
    with open(pose_save_path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            # Split the data into the quaternion and translation parts
            t_c2w, e_c2w  = np.split(np.array(tokens[1:], dtype=float), [3])
            e_c2w = [e_c2w[1], e_c2w[0], e_c2w[2]]
            # e_c2w, t_c2w = np.split(np.array(tokens[1:], dtype=float), [3])
            config_web['euler_angles']  = e_c2w 
            config_web['translation']  = t_c2w
            config_web['image_id'] = name
            break
    renderer.delay_to_load_map(config_web)
    
    # render
    with open(pose_save_path, 'r') as f:
        for data in tqdm(f.read().rstrip().split('\n')):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            # Split the data into the quaternion and translation parts
            t_c2w, e_c2w  = np.split(np.array(tokens[1:], dtype=float), [3])
            e_c2w = [e_c2w[1], e_c2w[0], e_c2w[2]]
            # e_c2w, t_c2w = np.split(np.array(tokens[1:], dtype=float), [3])
            config_web['euler_angles']  = e_c2w 
            config_web['translation']  = t_c2w
            config_web['image_id'] = name

            renderer.rendering(config_web)
            

    
        
