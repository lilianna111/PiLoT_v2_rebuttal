import os
import cv2
import numpy as np

# ===== 根据你的 test.py 修改下面这几个路径/名字 =====
ref_dir = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"
name = "2"  # 与 test.py 里保持一致

img_path = os.path.join(ref_dir, f"{name}_bbox_dom.jpg")
npy_path = os.path.join(ref_dir, f"{name}_bbox_dom.npy")
mask_path = os.path.join(ref_dir, f"{name}_bbox_dom_mask.png")  # 可选

# ===== 加载数据 =====
if not os.path.exists(img_path):
    raise FileNotFoundError(f"找不到图像文件: {img_path}")
if not os.path.exists(npy_path):
    raise FileNotFoundError(f"找不到 NPY 文件: {npy_path}")

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
final_npy = np.load(npy_path)  # 期望形状 (H, W, 3)

if img is None:
    raise RuntimeError(f"读取图像失败: {img_path}")

h, w = img.shape[:2]
if final_npy.shape[0] != h or final_npy.shape[1] != w:
    print(f"⚠️ 图像尺寸: {img.shape[:2]}, NPY 尺寸: {final_npy.shape[:2]}")
else:
    print(f"✅ 图像与 NPY 尺寸一致: {img.shape[:2]}")

print(f"NPY 形状: {final_npy.shape}")
if final_npy.ndim == 3 and final_npy.shape[2] >= 3:
    flat = final_npy.reshape(-1, final_npy.shape[2])
    print("=== NPY 各通道统计（用于猜测坐标系）===")
    for c in range(min(3, final_npy.shape[2])):
        ch = flat[:, c]
        finite_mask = np.isfinite(ch)
        if not np.any(finite_mask):
            print(f"通道 {c}: 全部是 NaN/inf")
            continue
        ch = ch[finite_mask]
        print(f"通道 {c}: min={ch.min():.6f}, max={ch.max():.6f}, mean={ch.mean():.6f}")
    print("提示：")
    print(" - 若通道0/1在 100~150, 20~40 左右，可能是 WGS84 (经度/纬度，单位: 度)")
    print(" - 若三个通道都是 1e6 量级（正负几百万），多半是 ECEF (单位: 米)")
    print(" - 若通道0/1是几万到几十万，第三维是几百/几千，可能是投影平面坐标 + 高程")

# ===== 鼠标回调：点击图像返回对应 NPY 信息 =====
def on_mouse(event, x, y, flags, param):
    global img, final_npy
    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= x < w and 0 <= y < h:
            value = final_npy[y, x]
            print(f"[点击] 像素 (x={x}, y={y}) -> NPY 值: {value}")

            # 在图上画一个小圆点 + 文字
            vis = img.copy()
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            txt = f"({x},{y})"
            cv2.putText(vis, txt, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("final_img", vis)

# ===== 打开窗口并等待点击 =====
cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("final_img", on_mouse)

cv2.imshow("final_img", img)
print("在窗口中用鼠标左键点击像素点，终端将打印对应 NPY 的 3D 信息。按 q 或 ESC 退出。")

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in [27, ord('q')]:  # ESC or 'q'
        break

cv2.destroyAllWindows()