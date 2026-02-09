import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from diffusers import DDIMScheduler

from models.temporal_module import TemporalAttention
from models.marigold_dc import MarigoldDepthCompletionPipeline


class SpatioTemporalMarigoldPipeline(MarigoldDepthCompletionPipeline):
    def warp_latents(self, prev_latent, depth_curr, pose_curr_to_prev, K, output_shape):
        B, C, h, w = prev_latent.shape
        
        # 1. 下采样当前帧深度 (Max Pool)
        scale_factor = depth_curr.shape[-1] // w
        depth_resized = F.max_pool2d(depth_curr, kernel_size=scale_factor, stride=scale_factor)
        
        # 2. 构建当前帧网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=prev_latent.device),
            torch.arange(w, device=prev_latent.device),
            indexing='ij'
        )
        pixel_coords = torch.stack(
            [x_grid.flatten(), y_grid.flatten(), torch.ones_like(x_grid.flatten())], dim=0
        ).float()
        pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1) 

        # 3. 调整内参
        K_latent = K.clone()
        K_latent[:, :2, :] /= scale_factor
        K_inv = torch.inverse(K_latent)

        # 4. 反投影 (Curr Pixel -> Curr Cam)
        depth_flat = depth_resized.reshape(B, 1, -1) # fix view to reshape for safety
        cam_points_curr = K_inv.bmm(pixel_coords) * depth_flat
        cam_points_curr_homo = torch.cat(
            [cam_points_curr, torch.ones((B, 1, h*w), device=prev_latent.device)], dim=1)

        # 5. 变换 (Curr Cam -> Prev Cam)
        cam_points_prev = pose_curr_to_prev.bmm(cam_points_curr_homo)

        # 6. 投影 (Prev Cam -> Prev Pixel)
        proj_points_prev = K_latent.bmm(cam_points_prev[:, :3, :])
        z_prev = proj_points_prev[:, 2:3, :]
        eps = 1e-6
        u_prev = proj_points_prev[:, 0, :] / (z_prev.squeeze(1) + eps)
        v_prev = proj_points_prev[:, 1, :] / (z_prev.squeeze(1) + eps)

        # 7. Grid Sample
        u_norm = 2.0 * u_prev / (w - 1) - 1.0
        v_norm = 2.0 * v_prev / (h - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).view(B, h, w, 2)

        warped_latent = F.grid_sample(
            prev_latent, 
            grid.to(prev_latent.dtype),
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )

        # 8. Mask
        in_bound = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        valid_z = z_prev.view(B, h, w) > 0.01
        has_depth = depth_resized.squeeze(1) > 0
        valid_mask = (in_bound.view(B, h, w) & valid_z & has_depth).float().unsqueeze(1)
        return warped_latent, valid_mask

    def encode_batch_rgb(self, rgb_batch, generator=None):
        B, T, C, H, W = rgb_batch.shape
        images = 2.0 * rgb_batch - 1.0
        images_flat = images.view(B * T, C, H, W).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latents_dist = self.vae.encode(images_flat).latent_dist
            latents = latents_dist.sample(generator) * self.vae.config.scaling_factor
        h, w = latents.shape[-2:]
        return latents.view(B, T, 4, h, w)

    def encode_batch_depth(self, depth_batch, generator=None, max_depth=10.0):
        B, T, C, H, W = depth_batch.shape
        depth_inv = 1.0 / (depth_batch + 1e-6)
        mask_invalid = depth_batch < 1e-6
        depth_inv[mask_invalid] = 0.0
        
        depth_flat = depth_inv.view(B, T, -1)
        
        d_min = depth_flat.min(dim=-1, keepdim=True)[0].view(B, T, 1, 1, 1)
        d_max = depth_flat.max(dim=-1, keepdim=True)[0].view(B, T, 1, 1, 1)
        
        scale = d_max - d_min + 1e-6
        depth_01 = (depth_inv - d_min) / scale
        depth_3c = depth_01.repeat(1, 1, 3, 1, 1)
        images = 2.0 * depth_3c - 1.0 
        
        images_flat = images.view(B * T, 3, H, W).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latents_dist = self.vae.encode(images_flat).latent_dist
            latents = latents_dist.sample(generator) * self.vae.config.scaling_factor
            
        h, w = latents.shape[-2:]
        return latents.view(B, T, 4, h, w)
    
    def generate_correlated_noise(
            self, 
            batch_size, 
            num_frames, 
            latent_shape, 
            poses, 
            intrinsics, 
            sparse_depths, 
            generator=None, 
            correlation_strength=0.9,
            debug=None
        ):
        C, h, w = latent_shape
        noise_stack = []
        
        # Frame 0
        current_noise = torch.randn(
            (batch_size, C, h, w), generator=generator, device=self.device, dtype=self.dtype
        )
        noise_stack.append(current_noise)
        
        for t in range(1, num_frames):
            prev_noise = noise_stack[-1]
            pose_prev, pose_curr = poses[:, t-1], poses[:, t]
            pose_curr_to_prev = torch.inverse(pose_prev) @ pose_curr
            
            K = intrinsics[:, t]
            depth_curr = sparse_depths[:, t]
            
            warped_noise, valid_mask = self.warp_latents(
                prev_noise, depth_curr, pose_curr_to_prev, K, (h, w)
            )
            
            new_noise = torch.randn(
                (batch_size, C, h, w), generator=generator, device=self.device, dtype=self.dtype
            )
            
            alpha = correlation_strength
            beta = (1 - alpha**2) ** 0.5
            correlated_part = alpha * warped_noise + beta * new_noise
            
            final_noise = valid_mask * correlated_part + (1 - valid_mask) * new_noise
            noise_stack.append(final_noise)
        
        final_stack = torch.stack(noise_stack, dim=1)

        if debug is not None:
            save_debug_path = "/home/wsco1/yyz/ssh/dpdct/debug"
            os.makedirs(save_debug_path, exist_ok=True)
            noise_vis = final_stack[0, :, 0].float().cpu().numpy() # [T, h, w]
            sparse_vis = sparse_depths[0, :, 0].float().cpu().numpy()
            H, W = sparse_vis.shape[1], sparse_vis.shape[2]
            vis_frames = []
            for t in range(num_frames):
                d_t = sparse_vis[t]
                d_max = d_t.max()
                if d_max > 0:
                    d_img = (d_t / d_max * 255.0).astype(np.uint8)
                else:
                    d_img = np.zeros_like(d_t, dtype=np.uint8)
                
                d_img_color = cv2.applyColorMap(d_img, cv2.COLORMAP_INFERNO)
                n_t = noise_vis[t]
                n_t = (n_t - n_t.min()) / (n_t.max() - n_t.min()) * 255.0
                n_t = n_t.astype(np.uint8)
                n_t = cv2.resize(n_t, (W, H), interpolation=cv2.INTER_NEAREST)
                n_t_color = cv2.applyColorMap(n_t, cv2.COLORMAP_JET)

                concat = np.hstack([d_img_color, n_t_color])

                cv2.putText(concat, f"Frame {t}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(concat, "Sparse Depth", (20, H-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(concat, "Correlated Noise", (W+20, H-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                vis_frames.append(concat)
            final_vis = np.hstack(vis_frames)
            save_file = os.path.join(save_debug_path, "vis_noise_step.jpg")
            cv2.imwrite(save_file, final_vis)
            #logger.debug(f"Saved visualization to {save_file}")

        return final_stack
    
    def inject_temporal_layers(self, num_frames, camera_embed_dim=256):
        """
        核心手术：
        1. 修改 U-Net 输入层以接受 Sparse Depth。
        2. 遍历 U-Net，在每个 Spatial Transformer 后插入 Temporal Attention。
        """
        self.num_frames = num_frames
        
        # --- 步骤 1: 修改输入层 (conv_in) ---
        # 原始 Marigold 输入: 8 通道 (4 RGB Latent + 4 Noisy Depth Latent)
        # 新的目标输入: 9 通道 ( + 1 Sparse Depth Mask)
        # 注意: 如果你选择把 Sparse Encode 成 Latent，那就是 4+4+4=12 通道。
        # 这里我们采用更轻量的方案: 1 通道 Sparse (Resize 后直接拼)，这样参数量增加最少。
        
        old_conv = self.unet.conv_in
        old_weights = old_conv.weight.data # [320, 8, 3, 3]
        old_bias = old_conv.bias.data
        
        # 定义新卷积: 输入 9 通道
        new_in_channels = 9 
        new_conv = nn.Conv2d(
            new_in_channels, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            padding=old_conv.padding
        )
        
        # 【关键】零初始化策略 (Zero Initialization)
        # 前 8 个通道复制原有权重，第 9 个通道(Sparse)初始化为 0。
        # 这样训练初期模型行为与预训练模型完全一致，避免 Loss 爆炸。
        with torch.no_grad():
            # 权重部分是切片赋值，PyTorch 允许直接用 Tensor 赋值给 Parameter 的切片
            new_conv.weight[:, :8] = old_weights
            new_conv.weight[:, 8:] = 0.0 
            
            # bias 部分是整体替换，必须处理类型
            if old_bias is not None:
                # 方案 A (推荐): 将 Tensor 包装成 Parameter
                new_conv.bias = torch.nn.Parameter(old_bias)
            
        # 替换层
        new_conv = new_conv.to(self.unet.device)
        self.unet.conv_in = new_conv
        logger.info(f"Modified UNet input layer: {8} -> {new_in_channels} channels.")
        
        # --- 步骤 2: 冻结与解冻策略 ---
        # 默认先冻结所有
        self.unet.requires_grad_(False)
        
        # 解冻输入层 (必须训练!)
        self.unet.conv_in.requires_grad_(True)
        
        # --- 步骤 3: 插入 Temporal Attention ---
        temporal_layers = {}
        
        # Wrapper 类保持不变 (SpatioTemporalBlock)
        class SpatioTemporalBlock(nn.Module):
            def __init__(self, spatial_block, temporal_block, frames):
                super().__init__()
                self.spatial = spatial_block
                self.temporal = temporal_block
                self.frames = frames
                
            def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
                # 1. 获取 cross_attention_kwargs 引用
                # 注意：这里我们要操作的是引用，或者深拷贝一份修改
                cross_attention_kwargs = kwargs.get("cross_attention_kwargs", None)
                camera_emb = None

                # 2. 【关键步骤】拦截并移除 camera_emb
                if cross_attention_kwargs is not None:
                    # 这里的 pop 会把 camera_emb 取出来，同时从字典里删除它
                    # 这样传给 self.spatial 时，里面就不包含这个键了，AttnProcessor 就不会抱怨了
                    camera_emb = cross_attention_kwargs.pop("camera_emb", None)

                # 3. 执行 Spatial Block (原始 Transformer)
                # 此时 cross_attention_kwargs 已经干净了
                output = self.spatial(
                    hidden_states, 
                    encoder_hidden_states=encoder_hidden_states, 
                    **kwargs
                )

                # 处理输出
                if isinstance(output, tuple):
                    hidden_states = output[0]
                elif hasattr(output, 'sample'):
                    hidden_states = output.sample
                else:
                    hidden_states = output

                # 4. 执行 Temporal Block (注入刚才拦截下来的 camera_emb)
                if camera_emb is not None:
                    # 恢复一下 camera_emb 进 kwargs (可选，如果后面层还需要用到的话)
                    # 但通常 Temporal Block 是自包含的，这里直接传给 temporal 即可
                    hidden_states = self.temporal(hidden_states, self.frames, camera_emb)
                    
                    # 【重要】如果后续层共享这个 kwargs 对象，最好把 camera_emb 放回去
                    # cross_attention_kwargs['camera_emb'] = camera_emb 
                else:
                    # 如果没拿到，说明可能在推理或者忘记传了
                    # 视情况决定是否报错或静默失败
                    # hidden_states = self.temporal(hidden_states, self.frames, None)
                    pass

                return (hidden_states,)
                
        # 寻找并替换 Transformer 模块
        modules_to_replace = []
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'Transformer2DModel':
                modules_to_replace.append((name, module))

        self.temporal_modules = nn.ModuleList()

        for name, original_module in modules_to_replace:
            # 获取通道数
            if hasattr(original_module, 'in_channels'):
                dim = original_module.in_channels
            elif hasattr(original_module, 'config'):
                dim = original_module.config.in_channels
            else:
                dim = original_module.norm.weight.shape[0]

            # 创建 Temporal Layer
            temporal_layer = TemporalAttention(channel_dim=dim, camera_embed_dim=camera_embed_dim)
            temporal_layer = temporal_layer.to(device=self.unet.device, dtype=self.unet.dtype)
            
            # 【关键】Temporal Layer 必须训练
            temporal_layer.requires_grad_(True) 
            
            self.temporal_modules.append(temporal_layer)
            
            # 替换
            hybrid_block = SpatioTemporalBlock(original_module, temporal_layer, num_frames)
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = self.unet.get_submodule(parent_name)
            setattr(parent_module, child_name, hybrid_block)
            
        logger.info(f"Successfully injected {len(self.temporal_modules)} Temporal Attention layers.")
        
        # --- 步骤 4: 收集所有需要优化的参数 ---
        # 我们需要训练: 1. conv_in  2. temporal_modules
        # 可选: 3. spatial layers (建议设置小 LR 或 LoRA，这里暂不包含)
        
        params_to_optimize = [
            {'params': self.unet.conv_in.parameters(), 'lr': 1e-4}, # 输入层可以大一点
            {'params': self.temporal_modules.parameters(), 'lr': 1e-4}
        ]
        
        return params_to_optimize
    
    def train_step(self, rgb_latents, noise_stack, sparse_depth_mask=None):
        """
        执行单步训练：加噪 -> 预测噪声 -> 计算 Loss
        Args:
            rgb_latents: [B, T, 4, h, w] (Conditioning)
            noise_stack: [B, T, 4, h, w] (Target Noise / Initial Noise)
            sparse_depth_mask: Optional, 如果你有 sparse depth 的 mask
        """
        B, T, C, h, w = rgb_latents.shape
        device = rgb_latents.device
        
        # 1. 准备输入
        # 将 Batch 和 Time 维度合并，因为 U-Net 只吃 [N, C, H, W]
        # 注意：这里我们把 noise_stack 当作我们想要模型去预测的目标，或者作为加噪的源头
        # 标准 Diffusion 训练流程：
        #   x_0 (GT Depth Latent) -> 加噪 -> x_t
        #   model(x_t, t, condition) -> pred_noise
        #   Loss = MSE(pred_noise, noise)
        
        # 但 Marigold-DC 是 Depth Completion，情况特殊：
        # 它没有 GT Depth Latent (除非你有 Dense GT)。
        # 如果你有 Dense GT Depth (batch['gt_depth']):
        #    你需要把 GT Depth 编码成 Latent 作为 x_0。
        #    然后把 noise_stack 作为噪声加到 x_0 上。
        
        # 假设：你的 Dataset 提供了 Dense GT Depth (batch['gt_depth'])
        # 那么你需要先编码 GT Depth。如果目前只有 Sparse，训练会很难。
        # 既然是 Debug 阶段，我们先假设我们就在训练 "Denoising" 这一步。
        
        # 暂时策略：如果没有 GT Depth，我们无法做标准的 Diffusion 训练。
        # 这里的 noise_stack 是 "Correlated Noise"，它是作为 x_T (初始状态) 输入的。
        
        # 让我们回退一步：为了训练 Temporal Layer，我们需要让模型学会“去噪”。
        # 我们必须有 GT Depth Latent (x_0)。
        # 请确认你的 Dataset 是否返回了 Dense Depth？是的，之前的代码里有 gt_depth_list。
        
        raise NotImplementedError("需要传入 GT Depth Latent 来计算 Loss。请在主函数中编码 GT Depth。")

