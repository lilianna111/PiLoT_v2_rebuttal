###给一张图，你返回他在图上的像素坐标点
import cv2

# 定义鼠标回调函数
def get_pixel_coordinates(event, x, y, flags, param):
    """
    event: 鼠标事件（如点击、移动）
    x, y: 鼠标在图像上的当前坐标
    flags, param: 其他参数
    """
    # 监听鼠标左键点击事件 (EVENT_LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击位置坐标 -> X: {x}, Y: {y}")
        
        # 可选：在点击处画一个小圆圈作为视觉反馈
        # (图像, 圆心坐标, 半径, 颜色BGR, 线宽-1为填充)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, f"({x},{y})", (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 刷新窗口显示标记
        cv2.imshow('Image Window', img)

# 1. 读取图片
# 请将 'your_image.jpg' 替换为你的图片路径
image_path = '/home/amax/Documents/datasets/renders/DJI_20251217153719_0002_V/0_0.png' 
img = cv2.imread(image_path)

if img is None:
    print("错误：未找到图片，请检查路径。")
else:
    # 2. 创建一个窗口
    cv2.namedWindow('Image Window')

    # 3. 绑定鼠标回调函数到该窗口
    cv2.setMouseCallback('Image Window', get_pixel_coordinates)

    print("提示：请在图片上点击获取坐标。按 'q' 键退出程序。")
    
    # 4. 显示窗口并等待
    cv2.imshow('Image Window', img)
    
    # 循环等待，直到按下 'q' 键
    while True:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # 5. 销毁所有窗口
    cv2.destroyAllWindows()