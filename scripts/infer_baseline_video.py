import os
import sys
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDIMScheduler

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from models.marigold_dc import MarigoldDepthCompletionPipeline
from dataset.endo import EndoscopyDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def tensor_to_pil(tensor):
    """
    [C, H, W] Tensor (0-1) -> PIL Image
    """
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def tensor_to_numpy_depth(tensor):
    """
    [1, H, W] Tensor -> [H, W] Numpy array
    """
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    return tensor.squeeze(0).cpu().numpy()

def main():
    # 1. 配置路径
    config_path = "configs/config.yaml" 
    output_dir = "debug/video_result"
    os.makedirs(output_dir, exist_ok=True)
    video_save_path = os.path.join(output_dir, "marigold_single_frame.mp4")
    
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备模型 (Marigold-DC)
    print("正在加载 Marigold-DC 模型...")
    checkpoint_path = cfg['MARIGOLD_DC']['CHECKPOINT']
    
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        checkpoint_path, 
        prediction_type="depth",
        local_files_only=True # 按照您的要求，强制本地加载
    ).to(device, dtype=torch.float32)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. 准备数据
    print("正在初始化 Dataset...")
    # 注意：为了生成的视频连贯，建议 Dataset 不要 shuffle
    dataset = EndoscopyDataset(cfg, mode='val') 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"开始推理，视频将保存至: {video_save_path}")

    # 视频写入器初始化标记
    writer = None
    fps = 10 # 视频帧率，可按需调整

    # 4. 推理循环
    for i, batch in enumerate(tqdm(dataloader, desc="Processing Frames")):
        # 移除限制：处理所有帧
        # if i >= 10: break 
        
        T = batch['color'].shape[1]
        center_idx = T // 2
        
        # 提取数据
        rgb_tensor = batch['color'][0, center_idx]
        rgb_pil = tensor_to_pil(rgb_tensor)
        
        # Sparse (Inverse Depth)
        sparse_tensor = batch['sparse'][0, center_idx]
        sparse_numpy = tensor_to_numpy_depth(sparse_tensor)
        
        try:
            # 运行 Marigold-DC (输入 Sparse 1/d, 输出 Pred 1/d)
            # 使用 seed 保证一定的确定性，或者去掉 seed 增加随机性
            pred_depth = pipe(
                image=rgb_pil,
                sparse_depth=sparse_numpy,
                num_inference_steps=50, 
                ensemble_size=1,
                processing_resolution=768, 
                seed=2024 
            )
            
            # --- 视频可视化逻辑 (与 infer_video.py 对齐) ---
            # pred_depth 是 numpy array
            d = pred_depth
            
            # 1. Min-Max 归一化
            # 加上 1e-6 防止除以零
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
            
            # 2. 转为 uint8
            d_vis = (d_norm * 255).astype(np.uint8)
            
            # 3. 应用 Inferno Color Map
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            
            # --- 写入视频 ---
            if writer is None:
                # 第一次运行时初始化 writer
                h, w = d_color.shape[:2]
                print(f"初始化视频写入器: {w}x{h} @ {fps}fps")
                writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            # 确保尺寸一致 (防止偶尔尺寸波动导致报错)
            if d_color.shape[0] != h or d_color.shape[1] != w:
                d_color = cv2.resize(d_color, (w, h))
                
            writer.write(d_color)
            
            # 可选：同时也保存单帧图片用于检查
            # cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), d_color)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # 释放资源
    if writer is not None:
        writer.release()
        print(f"视频生成完毕: {video_save_path}")
    else:
        print("未生成任何帧，请检查数据加载。")

if __name__ == "__main__":
    main()