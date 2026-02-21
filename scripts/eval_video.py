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

# 引入您添加的对齐评估工具
from utils.alignment import compute_metrics

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
        alpha = correlation
        beta = (1 - alpha**2) ** 0.5
        new_noise = torch.randn((B, C, H, W), generator=generator, device=device)
        prev_noise = noise_stack[-1]
        curr_noise = alpha * prev_noise + beta * new_noise
        noise_stack.append(curr_noise)
        
    return torch.stack(noise_stack, dim=1) # [B, T, C, H, W]

@torch.no_grad()
def evaluate_dataset(pipe, dataset, device="cuda", output_dir="TaoTie_eval_results"):
    if len(dataset) == 0:
        logger.error("Dataset is empty.")
        return
        
    # 按视频对样本索引进行分组
    video_to_indices = {}
    for idx, sample in enumerate(dataset._samples):
        vpath = sample["video_path"]
        if vpath not in video_to_indices:
            video_to_indices[vpath] = []
        video_to_indices[vpath].append(idx)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 逐个视频进行评估
    for video_path, indices in video_to_indices.items():
        logger.info(f"Evaluating video: {video_path}")
        
        video_name = os.path.basename(video_path.strip('/'))
        
        # 获取视频元数据
        kf_list = dataset._video_frame_lists[video_path]
        num_kfs = len(kf_list)
        context_len = dataset.context_len
        
        # 通过读取第一个 batch 来获取图像的高和宽
        first_batch = dataset[indices[0]]
        _, h, w = first_batch['color'].shape[-3:]
        
        # 初始化全局 Buffer
        global_pred_sum = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_count = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_gt = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_sparse = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        
        logger.info(f"Processing {num_kfs} keyframes with context length {context_len}...")
        
        # 滑动窗口推理
        for i, ds_idx in enumerate(tqdm(indices, desc="Inference")):
            batch = dataset[ds_idx]
            
            # 增加 Batch 维度并移动到设备 [1, T, C, H, W]
            colors = batch['color'].unsqueeze(0).to(device)
            sparses = batch['sparse'].unsqueeze(0).to(device)
            depths_gt = batch['depth'].unsqueeze(0).to(device) 
            poses = batch['pose'].unsqueeze(0).to(device)
            intrinsics = batch['intrinsic'].unsqueeze(0).to(device)
            
            B, T = colors.shape[:2]
            
            # 推理窗口在全局关键帧列表中的索引位置，严格对齐 endo.py 逻辑
            half = context_len // 2
            start_idx = max(0, min(i - half, num_kfs - context_len))
            if num_kfs < context_len: start_idx = 0
            
            window_indices = []
            for j in range(context_len):
                idx = start_idx + j
                if idx >= num_kfs: idx = num_kfs - 1 # endo.py 里面的 padding 逻辑
                window_indices.append(idx)
                
            # 将 GT 和 Sparse 写入全局 Buffer (每个全局帧只存一次即可)
            for local_idx, global_idx in enumerate(window_indices):
                if global_count[global_idx].max() == 0:
                    global_gt[global_idx] = depths_gt[0, local_idx, 0] # 提取单通道 [H, W]
                    global_sparse[global_idx] = sparses[0, local_idx, 0]
            
            # 数据预处理
            mask_batch = (sparses > 1e-4).float()
            sparse_norm = depth_transform(sparses)
            sparse_norm = sparse_norm * mask_batch + (-1.0) * (1 - mask_batch)
            
            # 编码 RGB
            rgb_flat = colors.view(B*T, 3, h, w)
            latents = pipe.encode_rgb(rgb_flat)
            latent_h, latent_w = latents.shape[-2:]
            
            sparse_small = torch.nn.functional.interpolate(sparse_norm.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
            mask_small = torch.nn.functional.interpolate(mask_batch.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
            
            # 初始噪声
            noise = generate_chrono_noise(B, T, 4, latent_h, latent_w, None, device, correlation=0.6)
            noise = rearrange(noise, 'b t c h w -> (b t) c h w')
            
            latents_input = latents * pipe.scheduler.init_noise_sigma
            pipe.scheduler.set_timesteps(50)
            
            cam_emb = pipe.encode_camera(intrinsics, poses)
            empty_text_embed = pipe._get_empty_text_embedding(device, pipe.unet.dtype)
            encoder_hidden_states = empty_text_embed.repeat(B*T, 1, 1)
            
            # 去噪循环
            for t in pipe.scheduler.timesteps:
                latent_model_input = pipe.scheduler.scale_model_input(latents_input, t)
                unet_input = torch.cat([latents, latent_model_input, sparse_small, mask_small], dim=1)
                
                t_batch = t.unsqueeze(0).repeat(B*T).to(device)
                
                noise_pred = pipe.unet(
                    unet_input, t_batch,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs={"num_frames": T}
                ).sample
                
                latents_input = pipe.scheduler.step(noise_pred, t, latents_input).prev_sample
                
            # 解码与反归一化
            decoded = pipe.vae.decode(latents_input / pipe.vae.config.scaling_factor).sample
            log_min, log_max = -6.9, -1.6
            decoded_log = (decoded.mean(dim=1, keepdim=True) + 1.0) * 0.5 * (log_max - log_min) + log_min
            decoded_metric = torch.exp(decoded_log).view(B, T, 1, h, w)
            
            # 高斯融合
            window_weights = torch.hamming_window(T, device=device).reshape(1, T, 1, 1, 1)
            for local_idx, global_idx in enumerate(window_indices):
                w_val = window_weights[0, local_idx, 0, 0, 0]
                frame_pred = decoded_metric[:, local_idx]
                global_pred_sum[global_idx] += frame_pred[0, 0] * w_val
                global_count[global_idx] += w_val
                
        # 计算指标与生成视频
        logger.info("Computing Metrics & Generating Video...")
        final_preds = global_pred_sum / (global_count + 1e-6)
        
        video_out_path = os.path.join(output_dir, f"{video_name}_eval.mp4")
        writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w * 3, h))
        
        mae_list, rmse_list = [], []
        
        for i in range(num_kfs):
            pred = final_preds[i]
            gt = global_gt[i] 
            sparse = global_sparse[i]
            
            # 基于最小二乘法的尺度和偏移对齐
            metrics = compute_metrics(pred, gt, align=True)
            mae_list.append(metrics['mae'])
            rmse_list.append(metrics['rmse'])
            pred_aligned = metrics['pred_aligned']
            
            # 可视化范围基于真值固定
            valid_gt = gt[gt > 0]
            v_min, v_max = (valid_gt.min().item(), valid_gt.max().item()) if len(valid_gt) > 0 else (0, 1)
            
            def to_color(d):
                norm = (d - v_min) / (v_max - v_min + 1e-6)
                norm = torch.clamp(norm, 0, 1).cpu().numpy()
                vis = (norm * 255).astype(np.uint8)
                return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
                
            vis_gt = to_color(gt)
            vis_pred = to_color(pred_aligned)
            
            vis_sparse = np.zeros_like(vis_gt)
            sparse_np = sparse.cpu().numpy()
            valid_sparse = sparse_np > 0
            vis_sparse[valid_sparse] = (0, 255, 0) # 绿色点突出稀疏深度
            
            # 从左至右：输入稀疏点 | 真实深度 | 对齐后的预测深度
            row = np.hstack([vis_sparse, vis_gt, vis_pred])
            cv2.putText(row, f"MAE: {metrics['mae']:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(row)
            
        writer.release()
        
        mean_mae = np.mean(mae_list)
        mean_rmse = np.mean(rmse_list)
        logger.info(f"Video {video_name} Evaluation Complete.")
        logger.info(f"Mean MAE:  {mean_mae:.5f}")
        logger.info(f"Mean RMSE: {mean_rmse:.5f}")
        
        # 将结果写入文本文档
        with open(os.path.join(output_dir, f"{video_name}_metrics.txt"), "w") as f:
            f.write(f"Evaluated Keyframes: {num_kfs}\n")
            f.write(f"Mean MAE:  {mean_mae:.5f}\n")
            f.write(f"Mean RMSE: {mean_rmse:.5f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = "cuda"
    
    # 1. 初始化模型，仅读取本地文件
    pipe = VideoMarigoldPipeline.from_pretrained(
        cfg['MARIGOLD_DC']['CHECKPOINT'],
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    pipe.setup_video_model(
        num_frames=cfg['DATA']['CONTEXT_SIZE'],
        camera_embed_dim=cfg['DATA']['CAMERA_EMBED_DIM']
    )
    pipe = pipe.to(device)
    
    logger.info(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint)
    pipe.unet.load_state_dict(state_dict['unet_state'], strict=False)
    pipe.pose_encoder.load_state_dict(state_dict['pose_encoder'])
    
    # 2. 实例化测试集 (自动解析所有的 map.json 并执行投影)
    test_dataset = EndoscopyDataset(cfg, mode='test')
    
    # 3. 运行完整评估流程
    evaluate_dataset(pipe, test_dataset, device=device)

if __name__ == "__main__":
    main()