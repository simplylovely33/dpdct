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

sys.path.append(os.getcwd())

from models.video_marigold import VideoMarigoldPipeline
from dataset.endo import EndoscopyDataset
from models.temporal_module import depth_transform
from utils.alignment import compute_global_metrics

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def generate_chrono_noise(B, T, C, H, W, generator, device, correlation=0.5):
    noise_stack = []
    noise = torch.randn((B, C, H, W), generator=generator, device=device)
    noise_stack.append(noise)
    
    for t in range(1, T):
        alpha = correlation
        beta = (1 - alpha**2) ** 0.5
        new_noise = torch.randn((B, C, H, W), generator=generator, device=device)
        prev_noise = noise_stack[-1]
        curr_noise = alpha * prev_noise + beta * new_noise
        noise_stack.append(curr_noise)
        
    return torch.stack(noise_stack, dim=1) 

@torch.no_grad()
def evaluate_dataset(pipe, dataset, device="cuda", output_dir="TaoTie_eval_results"):
    if len(dataset) == 0:
        logger.error("Dataset is empty.")
        return
        
    video_to_indices = {}
    for idx, sample in enumerate(dataset._samples):
        vpath = sample["video_path"]
        if vpath not in video_to_indices:
            video_to_indices[vpath] = []
        video_to_indices[vpath].append(idx)
        
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path, indices in video_to_indices.items():
        logger.info(f"Evaluating video: {video_path}")
        video_name = os.path.basename(video_path.strip('/'))
        
        kf_list = dataset._video_frame_lists[video_path]
        num_kfs = len(kf_list)
        context_len = dataset.context_len
        
        first_batch = dataset[indices[0]]
        _, h, w = first_batch['color'].shape[-3:]
        
        # 全局 Buffer
        global_pred_sum = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_count = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_gt = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        global_sparse = torch.zeros((num_kfs, h, w), device=device, dtype=torch.float32)
        
        logger.info(f"Processing {num_kfs} keyframes with context length {context_len}...")
        
        # 1. 滑动窗口推理
        for i, ds_idx in enumerate(tqdm(indices, desc="Inference")):
            batch = dataset[ds_idx]
            
            colors = batch['color'].unsqueeze(0).to(device)
            sparses = batch['sparse'].unsqueeze(0).to(device)
            depths_gt = batch['depth'].unsqueeze(0).to(device) 
            
            B, T = colors.shape[:2]
            
            half = context_len // 2
            start_idx = max(0, min(i - half, num_kfs - context_len))
            if num_kfs < context_len: start_idx = 0
            
            window_indices = []
            for j in range(context_len):
                idx = start_idx + j
                if idx >= num_kfs: idx = num_kfs - 1 
                window_indices.append(idx)
                
            for local_idx, global_idx in enumerate(window_indices):
                if global_count[global_idx].max() == 0:
                    global_gt[global_idx] = depths_gt[0, local_idx, 0] 
                    global_sparse[global_idx] = sparses[0, local_idx, 0]
            
            mask_batch = (sparses > 1e-4).float()
            sparse_norm = depth_transform(sparses)
            sparse_norm = sparse_norm * mask_batch + (-1.0) * (1 - mask_batch)
            
            rgb_flat = colors.view(B*T, 3, h, w)
            latents = pipe.encode_rgb(rgb_flat)
            latent_h, latent_w = latents.shape[-2:]
            
            sparse_small = torch.nn.functional.interpolate(sparse_norm.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
            mask_small = torch.nn.functional.interpolate(mask_batch.view(B*T, 1, h, w), size=(latent_h, latent_w), mode='nearest')
            
            noise = generate_chrono_noise(B, T, 4, latent_h, latent_w, None, device, correlation=0.6)
            noise = rearrange(noise, 'b t c h w -> (b t) c h w')
            
            latents_input = latents * pipe.scheduler.init_noise_sigma
            pipe.scheduler.set_timesteps(50)
            
            # 移除了 CameraPoseEncoder 的注入，仅使用 Text Embedding
            empty_text_embed = pipe._get_empty_text_embedding(device, pipe.unet.dtype)
            encoder_hidden_states = empty_text_embed.repeat(B*T, 1, 1)
            
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
                
            decoded = pipe.vae.decode(latents_input / pipe.vae.config.scaling_factor).sample
            
            # 注意：如果您基于 [0, 1] 重新训练了模型，此处的反 Log 变换需要修改为线性映射
            # 目前保留 Marigold 默认的反归一化
            log_min, log_max = -6.9, -1.6
            decoded_log = (decoded.mean(dim=1, keepdim=True) + 1.0) * 0.5 * (log_max - log_min) + log_min
            decoded_metric = torch.exp(decoded_log).view(B, T, 1, h, w)
            
            window_weights = torch.hamming_window(T, device=device).reshape(1, T, 1, 1, 1)
            for local_idx, global_idx in enumerate(window_indices):
                w_val = window_weights[0, local_idx, 0, 0, 0]
                frame_pred = decoded_metric[:, local_idx]
                global_pred_sum[global_idx] += frame_pred[0, 0] * w_val
                global_count[global_idx] += w_val
                
        # 2. 全局对齐与指标计算
        logger.info("Performing Global Spatio-Temporal Alignment...")
        final_preds = global_pred_sum / (global_count + 1e-6)
        
        # 传入整个视频的 Pred 和 GT 求解一次性的 Scale 和 Shift
        metrics = compute_global_metrics(final_preds, global_gt)
        global_mae = metrics['mae']
        global_rmse = metrics['rmse']
        aligned_preds = metrics['pred_aligned']
        scale, shift = metrics['scale'], metrics['shift']
        
        logger.info(f"Global Alignment -> Scale: {scale:.4f}, Shift: {shift:.4f}")
        
        # 3. 可视化视频生成
        video_out_path = os.path.join(output_dir, f"{video_name}_eval.mp4")
        writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w * 3, h))
        
        # 全局统一 ColorMap 的极值，防止视频闪烁
        valid_gt_video = global_gt[global_gt > 0]
        v_min, v_max = (valid_gt_video.min().item(), valid_gt_video.max().item()) if len(valid_gt_video) > 0 else (0, 1)
        
        def to_color(d):
            norm = (d - v_min) / (v_max - v_min + 1e-6)
            norm = torch.clamp(norm, 0, 1).cpu().numpy()
            vis = (norm * 255).astype(np.uint8)
            return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        
        for i in range(num_kfs):
            pred_aligned = aligned_preds[i]
            gt = global_gt[i] 
            sparse = global_sparse[i]
            
            vis_gt = to_color(gt)
            vis_pred = to_color(pred_aligned)
            
            vis_sparse = np.zeros_like(vis_gt)
            sparse_np = sparse.cpu().numpy()
            valid_sparse = sparse_np > 0
            vis_sparse[valid_sparse] = (0, 255, 0) 
            
            row = np.hstack([vis_sparse, vis_gt, vis_pred])
            
            # 写入全局评估数值
            cv2.putText(row, f"Video MAE: {global_mae:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(row)
            
        writer.release()
        
        logger.info(f"Video {video_name} Evaluation Complete.")
        logger.info(f"Global MAE:  {global_mae:.5f}")
        logger.info(f"Global RMSE: {global_rmse:.5f}")
        
        with open(os.path.join(output_dir, f"{video_name}_metrics.txt"), "w") as f:
            f.write(f"Evaluated Keyframes: {num_kfs}\n")
            f.write(f"Global Scale: {scale:.5f}\n")
            f.write(f"Global Shift: {shift:.5f}\n")
            f.write(f"Global MAE:  {global_mae:.5f}\n")
            f.write(f"Global RMSE: {global_rmse:.5f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = "cuda"
    
    pipe = VideoMarigoldPipeline.from_pretrained(
        cfg['MARIGOLD_DC']['CHECKPOINT'],
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # 移除了 camera_embed_dim 参数
    pipe.setup_video_model(num_frames=cfg['DATA']['CONTEXT_SIZE'])
    pipe = pipe.to(device)
    
    logger.info(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint)
    pipe.unet.load_state_dict(state_dict['unet_state'], strict=False)
    # 不再加载 pose_encoder 的权重
    
    test_dataset = EndoscopyDataset(cfg, mode='test')
    evaluate_dataset(pipe, test_dataset, device=device)

if __name__ == "__main__":
    main()