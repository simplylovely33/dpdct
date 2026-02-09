import cv2
import numpy as np

# 原始参数
orig_w, orig_h = 1350, 1080
target_w, target_h = 640, 512  # 你实际使用的目标宽

# 原始内参
camera_params = {
    "fx": 802.319,
    "fy": 801.885,
    "cx": 668.286,
    "cy": 547.733,
    "k1": -0.42234,  # 畸变参数通常不需要缩放
    "k2": 0.10654
}

# 1. 计算缩放因子
scale_x = target_w / orig_w
scale_y = target_h / orig_h

# 2. 计算新内参
new_params = camera_params.copy()
new_params["fx"] *= scale_x
new_params["fy"] *= scale_y
new_params["cx"] *= scale_x
new_params["cy"] *= scale_y

print("新的内参:")
print(new_params)


path = "/home/wsco1/yyz/dataset/endo/c3vd/cecum_t1_a/depth/0000_depth.tiff"

# flags=-1 (或 cv2.IMREAD_UNCHANGED) 非常重要，否则会自动转成 8位 BGR
depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

if depth is None:
    print("读取失败，路径不对")
else:
    print(f"数据类型 (dtype): {depth.dtype}")
    print(f"形状 (shape): {depth.shape}")
    print(f"数值范围: Min={depth.min()}, Max={depth.max()}")
    
    # 简单的判断逻辑
    if depth.dtype == np.uint16:
        print("✅ 结论：这是 16位 无符号整数格式 (C3VD 原生格式)")
    elif depth.dtype == np.float32:
        print("⚠️ 结论：这是 32位 浮点数格式")