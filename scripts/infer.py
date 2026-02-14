import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from loguru import logger
from einops import rearrange
import utils.toolkit as utils
from models.temporal_module import depth_transform 



class DepthMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.depth_errors = {
            'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []
        }
        self.count = 0

    def compute_errors(self, gt, pred):
        """
        计算单张图的指标
        gt, pred: [H, W] numpy arrays or torch tensors
        """
        if isinstance(gt, torch.Tensor): gt = gt.cpu().numpy()
        if isinstance(pred, torch.Tensor): pred = pred.cpu().numpy()

        # 1. 有效掩码 (只在 GT 有值的地方算)
        mask = (gt > 1e-3) & (gt < 80.0) # 假设最大深度 80m，根据你的场景调整
        gt = gt[mask]
        pred = pred[mask]

        if len(gt) == 0: return # 避免空数据报错

        # 2. 尺度对齐 (可选，如果你追求绝对深度，可以注释掉这一段)
        # scale, shift = np.polyfit(pred, gt, 1)
        # pred = pred * scale + shift
        
        # 3. 防止除零和负数
        pred = np.clip(pred, 1e-3, 80.0) 

        # 4. 计算指标
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        # 记录
        self.depth_errors['abs_rel'].append(abs_rel)
        self.depth_errors['sq_rel'].append(sq_rel)
        self.depth_errors['rmse'].append(rmse)
        self.depth_errors['rmse_log'].append(rmse_log)
        self.depth_errors['a1'].append(a1)
        self.depth_errors['a2'].append(a2)
        self.depth_errors['a3'].append(a3)
        self.count += 1

    def print_avg(self, logger=None):
        avg_metrics = {k: np.mean(v) for k, v in self.depth_errors.items()}
        msg = (f"  AbsRel: {avg_metrics['abs_rel']:.4f}\n"
               f"  SqRel : {avg_metrics['sq_rel']:.4f}\n"
               f"  RMSE  : {avg_metrics['rmse']:.4f}\n"
               f"  d < 1.25   : {avg_metrics['a1']:.3f}")
        if logger:
            logger.info(f"Validation Metrics:\n{msg}")
        else:
            print(msg)
        return avg_metrics
    

def process_depth_for_vis(depth_map):
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
        
    # 1. 过滤无效值
    valid_mask = depth_map > 1e-4
    if valid_mask.sum() == 0:
        return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    valid_data = depth_map[valid_mask]

    # --- 关键修改：使用分位数进行鲁棒归一化 ---
    # 找出 2% 和 98% 的分位点，忽略极亮和极暗的噪点
    p_min = np.percentile(valid_data, 2)
    p_max = np.percentile(valid_data, 98)
    
    # 打印一下数值范围，帮你诊断模型到底输出了什么
    # print(f"Depth Range: {p_min:.4f} ~ {p_max:.4f}") 

    # 截断并归一化到 0~1
    norm = np.clip((depth_map - p_min) / (p_max - p_min + 1e-8), 0, 1)
    
    # 转 uint8 并上色
    norm_uint8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_INFERNO)
    
    # 背景置黑
    colored[~valid_mask] = 0
    return colored

def process_rgb_for_vis(rgb_tensor):
    """
    RGB Tensor [3, H, W] -> Image [H, W, 3] BGR for OpenCV
    Input is assumed to be [0, 1]
    """
    img = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def colorize_depth(depth_map):
    """将单通道深度图转为彩色热力图以便观察"""
    # depth_map: [H, W] numpy array
    # 归一化到 0-255
    min_v, max_v = depth_map.min(), depth_map.max()
    if max_v - min_v > 1e-6:
        norm = (depth_map - min_v) / (max_v - min_v)
    else:
        norm = np.zeros_like(depth_map)
    
    norm = (norm * 255).astype(np.uint8)
    # 使用 INFERNO 或 MAGMA 色图
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return colored


@torch.no_grad()
def run_inference_preview(pipe, camera_encoder, dataloader, device, dtype, generator, epoch, num_batches=5):
    """
    运行一次推理并保存可视化结果 (适配 10 通道 Log+Mask 输入)
    """
    pipe.unet.eval()
    pipe.vae.eval()
    camera_encoder.eval()

    save_dir, save_name = os.path.join('./debug/val', f'epoch{epoch}'), 'preview'
    os.makedirs(save_dir, exist_ok=True)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches: break
        
        # 1. 准备数据
        rgb = batch['color'].to(device, dtype=dtype)
        sparse_metric = batch['sparse'].to(device, dtype=dtype) # 原始 Metric 深度 [B, T, 1, H, W]
        gt_depth = batch['depth'].to(device, dtype=dtype)
        pose = batch['pose'].to(device, dtype=dtype)
        intrinsic = batch['intrinsic'].to(device, dtype=dtype)

        B, T, C, H, W = rgb.shape
        intr_flat, pose_flat = utils.normalize_flatten_camrea_info(intrinsic, pose, W, H)
        camera_embedding = camera_encoder(intr_flat, pose_flat)

        # --- Step 1: 编码 RGB ---
        rgb_latents = pipe.encode_batch_rgb(rgb, generator=generator) # [B, T, 4, h, w]
        latent_h, latent_w = rgb_latents.shape[-2:]

        # --- 【关键修改】Step 1.5: 准备 Sparse Latent (Log + Mask) ---
        # 必须与训练时的 prepare_unet_input 逻辑保持完全一致
        
        # A. 生成 Mask
        valid_mask = (sparse_metric > 1e-4).float()
        
        # B. Log 变换 + 归一化 (使用统一的 depth_transform)
        sparse_norm = depth_transform(sparse_metric) # [-1, 1]
        
        # C. 应用 Mask (无效区域填 -1 或 0)
        sparse_norm = sparse_norm * valid_mask + (-1.0) * (1 - valid_mask)

        # D. Flatten & Resize
        # [B, T, 1, H, W] -> [B*T, 1, H, W]
        sparse_flat = rearrange(sparse_norm, 'b t c h w -> (b t) c h w')
        mask_flat = rearrange(valid_mask, 'b t c h w -> (b t) c h w')
        
        # Resize 到 Latent 大小 (nearest 插值)
        sparse_resized = F.interpolate(sparse_flat, size=(latent_h, latent_w), mode='nearest')
        mask_resized = F.interpolate(mask_flat, size=(latent_h, latent_w), mode='nearest')

        # --- Step 2: 初始噪声 ---
        # 注意: generate_correlated_noise 内部如果用了 warping，暂时可能还是基于旧逻辑
        # 但它主要影响初始噪声分布，暂时不改也没关系，或者确保它用的是 Metric Depth
        init_noise = pipe.generate_correlated_noise(
            batch_size=B, num_frames=T, latent_shape=(4, latent_h, latent_w),
            poses=pose, intrinsics=intrinsic, sparse_depths=sparse_metric, 
            generator=generator, correlation_strength=0.95
        )
        latents = rearrange(init_noise, 'b t c h w -> (b t) c h w') * pipe.scheduler.init_noise_sigma

        # --- Step 3: 去噪循环 ---
        pipe.scheduler.set_timesteps(50) 
        
        empty_text_embed = pipe.empty_text_embedding.to(device, dtype=dtype)
        batch_text_embed = empty_text_embed.repeat(B*T, 1, 1)

        for t in tqdm(pipe.scheduler.timesteps, desc=f"Infer Batch {batch_idx}", leave=False):
            # 1. 扩展 Latents
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            
            # 2. 拼接输入 (10通道)
            rgb_latents_flat = rearrange(rgb_latents, 'b t c h w -> (b t) c h w')
            
            # Order: RGB(4) + Noise(4) + Sparse(1) + Mask(1)
            unet_input = torch.cat([
                rgb_latents_flat, 
                latent_model_input, 
                sparse_resized, 
                mask_resized
            ], dim=1)
            
            # 3. 构造 Timestep
            t_batch = torch.tensor([t], device=device, dtype=dtype).repeat(B*T)
            
            # 4. 预测
            noise_pred = pipe.unet(
                unet_input,
                t_batch,
                encoder_hidden_states=batch_text_embed, 
                return_dict=False,
                cross_attention_kwargs={"camera_emb": camera_embedding}
            )[0]
            
            # 5. Step
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # --- Step 5: 解码 (这里假设你是预测 Latent) ---
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents).sample
        
        # 如果训练时 Target 也是 depth_transform 后的 [-1, 1] 图像
        # 那么这里 decode 出来的 image 也是 [-1, 1] 的 Log Depth
        # 我们需要把它还原回 Metric Depth 才能做可视化
        
        # 还原逻辑 (Log Inverse):
        # norm = (log_d - log_min) / (log_max - log_min) * 2 - 1
        # -> log_d = (norm + 1) / 2 * (log_max - log_min) + log_min
        # -> d = exp(log_d)
        
        # 简单可视化：直接取均值 (虽然 Latent 空间均值不完全等于 Pixel 均值，但作为预览够了)
        pred_depth_log_norm = torch.mean(image, dim=1, keepdim=True) 
        
        # 这里为了简单预览，我们直接把 Log Depth 画出来看纹理是否正确
        # 因为 Log Depth 本身就是最好的可视化形式（符合人眼感知）
        pred_depth = pred_depth_log_norm 

        logger.debug(f"[Batch {batch_idx}] Pred (LogNorm) Stats: Min={pred_depth.min():.4f}, Max={pred_depth.max():.4f}")
        
        pred_depth = rearrange(pred_depth, '(b t) c h w -> b t c h w', b=B)
        
        # --- 可视化拼接 ---
        video_idx = 0
        row_rgb, row_gt, row_sparse, row_pred = [], [], [], []
        
        for t in range(T):
            target_h, target_w = rgb.shape[-2], rgb.shape[-1]
            
            img_vis = process_rgb_for_vis(rgb[video_idx, t])
            # GT 是 Metric 的，处理一下方便对比
            gt_vis = process_depth_for_vis(gt_depth[video_idx, t, 0])
            # Sparse 也是 Metric 的
            sp_vis = process_depth_for_vis(sparse_metric[video_idx, t, 0])
            
            # Pred 是 Log Norm [-1, 1] 的，我们简单归一化到 0-1 做图
            # 也可以在这里写一个 inverse transform 变回 metric
            pred_vis_raw = pred_depth[video_idx, t, 0].float().cpu().numpy()
            # 简单 min-max 归一化用于可视化
            pred_vis_norm = (pred_vis_raw - pred_vis_raw.min()) / (pred_vis_raw.max() - pred_vis_raw.min() + 1e-6)
            pred_vis = (pred_vis_norm * 255).astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_INFERNO)
            
            if pred_vis.shape[:2] != (target_h, target_w):
                pred_vis = cv2.resize(pred_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            row_rgb.append(img_vis)
            row_gt.append(gt_vis)
            row_sparse.append(sp_vis)
            row_pred.append(pred_vis)
        
        final_grid = np.vstack([
            np.hstack(row_rgb),
            np.hstack(row_gt),
            np.hstack(row_sparse),
            np.hstack(row_pred)
        ])
        
        current_save_name = f"epoch{epoch}_{save_name}_{batch_idx}.jpg"
        save_path = os.path.join(save_dir, current_save_name)
        cv2.imwrite(save_path, final_grid)
        logger.info(f"Saved preview to {save_path}")

        del batch, rgb, sparse_metric, gt_depth, pose, intrinsic
        del rgb_latents, init_noise, latents, unet_input, noise_pred
        del image, pred_depth
        if 'rgb_latents_flat' in locals(): del rgb_latents_flat
        
        torch.cuda.empty_cache()