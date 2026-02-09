import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter:
    def __init__(self, momentum=0.95):
        self.val = 0
        self.avg = 0
        self.momentum = momentum
        self.initialized = False

    def update(self, val):
        if not self.initialized:
            self.avg = val
            self.initialized = True
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class GeometricConsistencyLoss(nn.Module):
    def __init__(self, vae, scheduler, sample_pairs=None, scaling_factor=0.18215):
        """
        几何一致性损失模块
        
        Args:
            vae: 预训练的 VAE 模型 (用于将 Latent 解码回 Pixel 空间)
            scheduler: 噪声调度器 (用于从 Latent 和 Noise 反推 x0)
            sample_pairs (int, optional): 如果非 None，每次只随机计算 N 对几何损失，防止 OOM。
            scaling_factor (float): VAE 的缩放因子，SD 默认为 0.18215
        """
        super().__init__()
        self.vae = vae
        self.scheduler = scheduler
        self.sample_pairs = sample_pairs
        self.scaling_factor = scaling_factor
        
        # 冻结 VAE 参数，但允许梯度回传 (PyTorch 特性: 即使 require_grad=False，
        # 只要输入有梯度，操作也是可微的)
        self.vae.requires_grad_(False) 

    def get_prediction_x0(self, noisy_latents, model_pred, timesteps):
        """反推 Clean Latent (x0)"""
        # 确保 alpha 在正确的设备上
        alphas_cumprod = self.scheduler.alphas_cumprod.to(noisy_latents.device)
        
        # 取出对应 timestep 的 alpha
        # timesteps: [N]
        alpha_prod_t = alphas_cumprod[timesteps]
        
        # Reshape for broadcasting: [N, 1, 1, 1]
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        # 公式: x0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        # 注意：这里假设 model_pred 是 epsilon。如果是 v-pred，公式需调整。
        pred_original_sample = (noisy_latents - beta_prod_t ** 0.5 * model_pred) / (alpha_prod_t ** 0.5)
        
        return pred_original_sample

    def warp_frame(self, depth_src, pose_src, pose_tgt, intrinsic):
        """
        基础 Warping 函数: 将 Source 视角的 Depth 投影到 Target 视角
        depth_src: [N, 1, H, W]
        pose: [N, 4, 4]
        intrinsic: [N, 3, 3]
        """
        N, _, H, W = depth_src.shape
        device = depth_src.device
        
        # 1. 像素坐标构建
        y_range = torch.arange(H, device=device).float()
        x_range = torch.arange(W, device=device).float()
        Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # [N, 3, H*W]
        pixel_coords = torch.stack([X, Y, torch.ones_like(X)], dim=0).flatten(1).unsqueeze(0).repeat(N, 1, 1)
        
        # 2. 反投影 Pixel -> Cam
        K_inv = torch.inverse(intrinsic)
        cam_coords = torch.matmul(K_inv, pixel_coords) 
        cam_coords = cam_coords * depth_src.flatten(2) # 乘以深度
        
        # 转齐次坐标 [N, 4, H*W]
        ones = torch.ones((N, 1, H*W), device=device)
        cam_coords_homo = torch.cat([cam_coords, ones], dim=1)
        
        # 3. 坐标变换 Src -> Tgt
        # T_rel = T_tgt^-1 @ T_src
        T_rel = torch.matmul(torch.inverse(pose_tgt), pose_src)
        tgt_coords_homo = torch.matmul(T_rel, cam_coords_homo)
        
        tgt_coords = tgt_coords_homo[:, :3, :] # 去掉齐次项
        
        # 4. 投影 Cam -> Pixel
        proj_coords = torch.matmul(intrinsic, tgt_coords)
        Z = proj_coords[:, 2:3, :] + 1e-6
        X_proj = proj_coords[:, 0:1, :] / Z
        Y_proj = proj_coords[:, 1:2, :] / Z
        
        # 5. 归一化 [-1, 1] 供 grid_sample 使用
        X_norm = 2 * X_proj / (W - 1) - 1
        Y_norm = 2 * Y_proj / (H - 1) - 1
        
        grid = torch.cat([X_norm, Y_norm], dim=1).permute(0, 2, 1).view(N, H, W, 2)
        projected_depth = Z.view(N, 1, H, W)
        
        return grid, projected_depth

    def forward(self, noisy_latents, model_pred, timesteps, pose, intrinsic, batch_size, num_frames):
        """
        Args:
            noisy_latents: [B*T, 4, h, w] (flattened)
            model_pred:    [B*T, 4, h, w] (flattened noise/velocity)
            timesteps:     [B*T]
            pose:          [B, T, 4, 4] 
            intrinsic:     [B, T, 3, 3]
            batch_size:    int (B)
            num_frames:    int (T)
        """
        # --- 配置：分辨率缩放比例 ---
        # 0.5 表示长宽各变为一半，显存占用约为原来的 1/4
        # 建议：如果显存非常紧张，可以用 0.5；如果还好，可以用 0.75
        downscale_ratio = 0.5 
        
        # 1. 反推 Latent x0 [B*T, 4, h, w]
        pred_latents = self.get_prediction_x0(noisy_latents, model_pred, timesteps)
        
        # 2. 准备配对索引 (Sampling Strategy)
        valid_pairs = []
        for b in range(batch_size):
            for t in range(num_frames - 1):
                curr_idx = b * num_frames + t
                next_idx = b * num_frames + (t + 1)
                valid_pairs.append((curr_idx, next_idx, b, t, t+1))
        
        if not valid_pairs:
            return torch.tensor(0.0, device=noisy_latents.device, requires_grad=True)

        if self.sample_pairs is not None and len(valid_pairs) > self.sample_pairs:
            import random
            selected_pairs = random.sample(valid_pairs, self.sample_pairs)
        else:
            selected_pairs = valid_pairs
            
        # 3. 提取需要计算的子集
        indices_src = [p[0] for p in selected_pairs]
        indices_tgt = [p[1] for p in selected_pairs]
        
        unique_indices = list(set(indices_src + indices_tgt))
        idx_map = {idx: i for i, idx in enumerate(unique_indices)}
        
        latents_subset = pred_latents[unique_indices] # [K, 4, h, w]
        
        # === 【核心修改 1】: Downsample Latents ===
        # 在 Decode 之前缩小 Latents，这样 VAE 就会输出小图
        # align_corners=False 也就是标准的线性插值
        if downscale_ratio != 1.0:
            latents_subset = F.interpolate(
                latents_subset, scale_factor=downscale_ratio, mode='bilinear', align_corners=False
            )
        
        # === VAE DECODE ===
        latents_subset = latents_subset / self.scaling_factor
        
        # 此时 decode 出来的 pixels_subset 已经是小图了 (例如 256x256)
        # 依然建议保留切片解码 (decode_latents_sliced) 以防万一，或者直接 decode
        # 如果你之前用了 slice 这里也可以用，如果没有就直接调 vae
        pixels_subset = self.vae.decode(latents_subset).sample 
        #pixels_subset = self.decode_latents_sliced(latents_subset) # 推荐保留切片逻辑
        
        depth_subset = pixels_subset.mean(dim=1, keepdim=True) # [K, 1, H_small, W_small]
        
        # 4. 准备 Warping 数据
        mapped_src_idx = [idx_map[i] for i in indices_src]
        mapped_tgt_idx = [idx_map[i] for i in indices_tgt]
        
        depth_src_batch = depth_subset[mapped_src_idx] 
        depth_tgt_batch = depth_subset[mapped_tgt_idx] 
        
        pose_src_list = []
        pose_tgt_list = []
        intr_src_list = []
        
        for _, _, b, t_src, t_tgt in selected_pairs:
            pose_src_list.append(pose[b, t_src])
            pose_tgt_list.append(pose[b, t_tgt])
            intr_src_list.append(intrinsic[b, t_src])
            
        pose_src_batch = torch.stack(pose_src_list, dim=0)
        pose_tgt_batch = torch.stack(pose_tgt_list, dim=0)
        intr_src_batch = torch.stack(intr_src_list, dim=0)

        # === 【核心修改 2】: Rescale Intrinsics ===
        # 图像缩小了，内参矩阵中的 fx, fy, cx, cy 也要相应缩小
        if downscale_ratio != 1.0:
            # Clone 一份，避免修改原始数据影响其他计算
            intr_src_batch = intr_src_batch.clone()
            # 内参矩阵前两行 (x, y 相关) 乘以缩放比例
            intr_src_batch[:, :2, :] *= downscale_ratio
        
        # 5. 执行 Warping (现在是在低分辨率下进行的)
        grid, projected_depth = self.warp_frame(depth_src_batch, pose_src_batch, pose_tgt_batch, intr_src_batch)
        
        sampled_depth_tgt = F.grid_sample(depth_tgt_batch, grid, padding_mode="border", align_corners=True)
        
        # 6. 计算 Loss
        mask = (grid.abs() <= 1).all(dim=-1, keepdim=True).float()
        
        diff = torch.abs(projected_depth - sampled_depth_tgt) * mask.permute(0, 3, 1, 2)
        loss = diff.mean()
        
        return loss