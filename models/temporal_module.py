import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CameraPoseEncoder(nn.Module):
    def __init__(self, input_dim=21, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

    def forward(self, intrinsic, pose):
        x = torch.cat([intrinsic, pose], dim=-1) # [B, T, 21]
        emb = self.net(x) # [B, T, embed_dim]
        return emb
    

class TemporalAttention(nn.Module):
    def __init__(self, channel_dim, camera_embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.channel_dim = channel_dim
        self.norm = nn.GroupNorm(32, channel_dim)
        self.camera_proj = nn.Linear(camera_embed_dim, channel_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=channel_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.proj_out = nn.Linear(channel_dim, channel_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, hidden_states, num_frames, camera_emb):
        identity = hidden_states
        b_t, c, h, w = hidden_states.shape
        B = b_t // num_frames
        # Pose/Intrinsics Embedding
        hidden_states = self.norm(hidden_states)
        hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', t=num_frames)
        cam_feat = self.camera_proj(camera_emb)
        hidden_states = hidden_states + cam_feat.unsqueeze(-1).unsqueeze(-1)

        hidden_states = rearrange(hidden_states, 'b t c h w -> (b h w) t c', t=num_frames)
        hidden_states, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, '(b h w) t c -> (b t) c h w', h=h, w=w)
        return identity + hidden_states
    


def depth_transform(depth_map, min_val=0.5, max_val=10.0):
    """
    统一的深度预处理：Log 变换 + 线性归一化到 [-1, 1]
    Args:
        depth_map: [B, ..., H, W] Metric Depth (单位: 米)
        min_val: 场景最小深度 (例如 1mm)
        max_val: 场景最大深度 (例如 20cm)
    Returns:
        norm_depth: [-1, 1]
    """
    # 1. 截断防止 Log 报错或数值过大
    depth_clamped = torch.clamp(depth_map, min=min_val, max=max_val)
    
    # 2. Log 变换 (将指数分布拉伸为线性分布)
    depth_log = torch.log(depth_clamped)
    
    # 3. 归一化到 [0, 1]
    log_min = torch.log(torch.tensor(min_val, device=depth_map.device))
    log_max = torch.log(torch.tensor(max_val, device=depth_map.device))
    depth_01 = (depth_log - log_min) / (log_max - log_min)
    
    # 4. 映射到 [-1, 1] (适配 SD Latent 分布)
    depth_norm = depth_01 * 2.0 - 1.0
    
    return depth_norm

def prepare_unet_input(color_latents, noisy_depth_latents, sparse_depth_metric, target_h, target_w):
    """
    输入:
        sparse_depth_metric: 原始 Metric 深度 (米)
    输出:
        unet_input: 10 通道 tensor
    """
    B, T, _, h, w = color_latents.shape
    
    # --- 1. 生成 Mask (非常重要) ---
    # 0 表示无效，1 表示有效
    valid_mask = (sparse_depth_metric > 1e-4).float() # 阈值稍微宽一点防止浮点误差
    
    # --- 2. 统一变换 ---
    # 调用上面定义的函数，保证和 GT 处理逻辑完全一致
    sparse_norm = depth_transform(sparse_depth_metric)
    
    # --- 3. Mask 过滤 ---
    # 无效区域必须置为 0 (或者 -1，但在 Mask 存在的情况下 0 比较好理解)
    # 注意：depth_transform 输出是 [-1, 1]，无效区域如果也是 -1 (即最小深度) 会有歧义
    # 所以有了 Mask 通道后，这里填 0 (也就是 Log 分布的中间值) 或者 -1 都可以
    # 建议填 -1 (代表最近距离)，因为 0 代表中间距离。
    sparse_norm = sparse_norm * valid_mask + (-1.0) * (1 - valid_mask)
    
    # --- 4. 调整尺寸 ---
    sparse_flat = rearrange(sparse_norm, 'b t c h w -> (b t) c h w')
    mask_flat = rearrange(valid_mask, 'b t c h w -> (b t) c h w')
    
    # 使用 nearest 插值保持稀疏性
    sparse_resized = F.interpolate(sparse_flat, size=(h, w), mode='nearest')
    mask_resized = F.interpolate(mask_flat, size=(h, w), mode='nearest')
    
    # --- 5. 拼接 (10通道) ---
    color_flat = rearrange(color_latents, 'b t c h w -> (b t) c h w')
    noise_flat = rearrange(noisy_depth_latents, 'b t c h w -> (b t) c h w')
    
    # 顺序: RGB(4) + Noise(4) + Sparse(1) + Mask(1)
    unet_input = torch.cat([color_flat, noise_flat, sparse_resized, mask_resized], dim=1)
    
    return unet_input