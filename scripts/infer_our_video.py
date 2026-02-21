import os
import sys
import yaml
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger
from diffusers import DDIMScheduler
from einops import rearrange

# 添加路径
sys.path.append(os.getcwd())

from models.video_marigold import VideoMarigoldPipeline
from dataset.endo import EndoscopyDataset
from models.temporal_module import depth_transform

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def generate_chrono_noise(B, T, C, H, W, generator, device, correlation=0.5):
    """
    生成时间相关的噪声 (ChronoDepth Strategy)
    """
    noise_stack = []
    # Frame 0
    noise = torch.randn((B, C, H, W), generator=generator, device=device)
    noise_stack.append(noise)
    
    for t in range(1, T):
        # 混合系数
        alpha = correlation
        beta = (1 - alpha**2) ** 0.5
        
        new_noise = torch.randn((B, C, H, W), generator=generator, device=device)
        prev_noise = noise_stack[-1]
        
        # 简单混合
        curr_noise = alpha * prev_noise + beta * new_noise
        noise_stack.append(curr_noise)
        
    return torch.stack(noise_stack, dim=1) # [B, T, C, H, W]

@torch.no_grad()
def infer_sliding_window(pipe, dataset, window_size=5, stride=2, device="cuda"):
    """
    滑动窗口推理主逻辑
    """
    # 1. 准备全局缓冲区
    if len(dataset) == 0:
        logger.error("Dataset is empty.")
        return

    # 假设我们只处理第一个视频
    sample = dataset._samples[0]
    dataset._lazy_init_video_meta(sample['video_path'])
    video_path = sample['video_path']
    frame_ids = dataset._video_frame_lists[video_path]
    total_frames = len(frame_ids)
    
    # 动态获取真实尺寸
    dummy_img, _, _ = dataset._read_rgb_and_size(video_path, frame_ids[0])
    c, h, w = dummy_img.shape
    logger.info(f"Detected processing resolution: {h}x{w}")
    
    # 累加器
    global_depth_sum = torch.zeros((total_frames, h, w), device=device, dtype=torch.float32)
    global_count = torch.zeros((total_frames, h, w), device=device, dtype=torch.float32)
    
    logger.info(f"Processing video {video_path}: {total_frames} frames.")
    
    # 2. 滑动窗口循环
    windows = []
    for start_idx in range(0, total_frames, stride):
        end_idx = start_idx + window_size
        if end_idx > total_frames:
            start_idx = max(0, total_frames - window_size)
            end_idx = total_frames
            windows.append(range(start_idx, end_idx))
            break
        windows.append(range(start_idx, end_idx))
    
    unique_windows = []
    seen_starts = set()
    for w_rng in windows:
        if w_rng.start not in seen_starts:
            unique_windows.append(w_rng)
            seen_starts.add(w_rng.start)
            
    # 开始推理
    for win_idx, frame_indices in enumerate(tqdm(unique_windows, desc="Sliding Window")):
        # A. 构造 Batch
        rgb_list, sparse_list = [], []
        pose_list = []
        
        real_frame_ids = [frame_ids[i] for i in frame_indices]
        
        for fid in real_frame_ids:
            # Read RGB
            img, _, _ = dataset._read_rgb_and_size(video_path, fid)
            rgb_list.append(img)
            
            # Read Depth (GT/Sparse)
            gt = dataset._read_depth_and_size(video_path, fid, "depth")
            
            # 模拟 Sparse (或者从外部读取)
            mask = (torch.rand_like(gt) > 0.99).float() 
            sparse = gt * mask
            sparse_list.append(sparse) 
            
            # Pose
            if fid in dataset._pose_cache[video_path]:
                pose_list.append(dataset._pose_cache[video_path][fid])
            else:
                pose_list.append(torch.eye(4))
                
        # Stack & Batch
        rgb_batch = torch.stack(rgb_list).unsqueeze(0).to(device) 
        sparse_batch = torch.stack(sparse_list).unsqueeze(0).to(device)
        pose_batch = torch.stack(pose_list).unsqueeze(0).to(device)
        
        # Intrinsics
        K_raw = dataset._read_intrinsics(video_path, h, w).to(device)
        K_batch = K_raw.unsqueeze(0).unsqueeze(0).repeat(1, len(frame_indices), 1, 1)
        
        # 尺寸校验
        B, T, _, current_h, current_w = rgb_batch.shape
        if current_h != h or current_w != w:
             # 如果出现尺寸不一致，这里简单处理为更新 h,w (可能会导致 buffer 错位，但在 dataset 一致的情况下不会发生)
             logger.warning(f"Batch dimension mismatch! Expected {h}x{w}, got {current_h}x{current_w}")
             h, w = current_h, current_w

        # B. 预处理
        mask_batch = (sparse_batch > 1e-4).float()
        sparse_norm = depth_transform(sparse_batch)
        sparse_norm = sparse_norm * mask_batch + (-1.0) * (1 - mask_batch)
        
        # C. Encode RGB
        rgb_flat = rgb_batch.view(B*T, 3, h, w)
        latents = pipe.encode_rgb(rgb_flat) 
        
        # Resize Sparse
        latent_h, latent_w = latents.shape[-2:]
        sparse_small = torch.nn.functional.interpolate(sparse_norm.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
        mask_small = torch.nn.functional.interpolate(mask_batch.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
        
        # D. 生成噪声
        noise = generate_chrono_noise(B, T, 4, latent_h, latent_w, None, device, correlation=0.6)
        noise = rearrange(noise, 'b t c h w -> (b t) c h w')
        
        latents_input = latents * pipe.scheduler.init_noise_sigma
        
        # E. 扩散循环
        pipe.scheduler.set_timesteps(50)
        
        # Empty Text
        empty_text_embed = pipe._get_empty_text_embedding(device, pipe.unet.dtype)
        encoder_hidden_states = empty_text_embed.repeat(B*T, 1, 1)

        for t in pipe.scheduler.timesteps:
            latent_model_input = pipe.scheduler.scale_model_input(latents_input, t)
            unet_input = torch.cat([latents, latent_model_input, sparse_small, mask_small], dim=1)
            
            # [关键修复] 将时间步 t 移动到 device
            t_batch = t.unsqueeze(0).repeat(B*T).to(device)

            noise_pred = pipe.unet(
                unet_input,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs={"num_frames": T}
            ).sample
            
            latents_input = pipe.scheduler.step(noise_pred, t, latents_input).prev_sample
            
        # F. Decode
        decoded = pipe.vae.decode(latents_input / pipe.vae.config.scaling_factor).sample 
        
        log_min, log_max = -6.9, -1.6
        decoded_log = (decoded.mean(dim=1, keepdim=True) + 1.0) * 0.5 * (log_max - log_min) + log_min
        decoded_metric = torch.exp(decoded_log) 
        
        # G. 写入全局 Buffer
        window_weights = torch.hamming_window(T, device=device).reshape(1, T, 1, 1, 1)
        decoded_metric = decoded_metric.view(B, T, 1, h, w)
        
        for i, global_idx in enumerate(frame_indices):
            w_val = window_weights[0, i, 0, 0, 0]
            if decoded_metric.shape[-1] != w:
                 frame_pred = torch.nn.functional.interpolate(decoded_metric[:, i], size=(h, w), mode='bilinear', align_corners=False)
            else:
                 frame_pred = decoded_metric[:, i]
                 
            global_depth_sum[global_idx] += frame_pred[0, 0] * w_val
            global_count[global_idx] += w_val
            
    # 3. 保存
    logger.info("Fusing windows...")
    final_depths = global_depth_sum / (global_count + 1e-6)
    
    save_dir = "debug/video_result"
    os.makedirs(save_dir, exist_ok=True)
    
    fps = 10
    out_path = os.path.join(save_dir, "depth_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for i in range(total_frames):
        d = final_depths[i].cpu().numpy()
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d_vis = (d_norm * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
        writer.write(d_color)
        
    writer.release()
    logger.info(f"Saved video to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = "cuda"
    
    # Init Model
    pipe = VideoMarigoldPipeline.from_pretrained(
        cfg['MARIGOLD_DC']['CHECKPOINT'],
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # 插入组件
    pipe.setup_video_model(
        num_frames=cfg['DATA']['CONTEXT_SIZE']
    )
    
    # 移动到 GPU
    pipe = pipe.to(device)
    
    # Load Weights
    logger.info(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint)
    pipe.unet.load_state_dict(state_dict['unet_state'], strict=False)
    
    dataset = EndoscopyDataset(cfg, mode='val')
    
    infer_sliding_window(pipe, dataset, window_size=5, stride=2, device=device)

if __name__ == "__main__":
    main()