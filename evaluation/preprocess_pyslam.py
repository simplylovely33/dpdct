import os
import cv2
import argparse
import glob
import numpy as np
import re
from tqdm import tqdm

def extract_number(filename):
    """
    从文件名提取数字用于自然排序
    例如: 'frame_10.png' -> 10
    """
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else -1

def main():
    parser = argparse.ArgumentParser(description="Prepare data for pySLAM")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing RGB images")
    parser.add_argument("--scene_name", type=str, required=True, help="Name of the scene (will be the folder name in output)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for timestamp generation (default: 15)")
    parser.add_argument("--ext", type=str, default="png", help="Image extension (png, jpg, etc.)")
    
    args = parser.parse_args()

    # 1. 准备输出路径
    base_output_dir = "/home/wsco1/yyz/pyslam/data/videos"
    output_dir = os.path.join(base_output_dir, args.scene_name)
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "rgb.mp4")
    times_path = os.path.join(output_dir, "times.txt")
    
    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")

    # 2. 读取并排序图片
    # 支持多种扩展名
    pattern = os.path.join(args.input_dir, f"*.{args.ext}")
    images = glob.glob(pattern)
    
    if len(images) == 0:
        # 尝试大写
        pattern = os.path.join(args.input_dir, f"*.{args.ext.upper()}")
        images = glob.glob(pattern)
        
    if len(images) == 0:
        print(f"Error: No images found in {args.input_dir} with extension {args.ext}")
        return

    # 关键：按数字顺序排序
    images.sort(key=extract_number)
    print(f"Found {len(images)} images.")

    # 3. 读取第一张图获取尺寸
    first_img = cv2.imread(images[0])
    if first_img is None:
        print(f"Error: Could not read image {images[0]}")
        return
    
    height, width, layers = first_img.shape
    print(f"Resolution: {width}x{height}")

    # 4. 初始化 VideoWriter
    # mp4v 是比较通用的编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

    # 5. 处理循环
    print("Processing...")
    with open(times_path, 'w') as f_times:
        for idx, img_path in enumerate(tqdm(images)):
            # A. 写入视频帧
            frame = cv2.imread(img_path)
            # 确保尺寸一致
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video.write(frame)

            # B. 计算并写入时间戳
            # timestamp = frame_index / fps
            timestamp = idx / float(args.fps)
            
            # 格式化为科学计数法: 6.666667e-02
            # {:.6e} 表示保留6位小数的科学计数法
            line = "{:.6e}\n".format(timestamp)
            f_times.write(line)

    # 6. 释放资源
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Done!")
    print(f"Video saved to: {video_path}")
    print(f"Times saved to: {times_path}")

if __name__ == "__main__":
    main()