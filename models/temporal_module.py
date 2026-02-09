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
    

def prepare_unet_input(color_latents, noisy_depth_latents, sparse_depth_metric, target_h, target_w):
    """
    Args:
        color_latents: [B, T, 4, h, w]
        noisy_depth_latents: [B, T, 4, h, w]
        sparse_depth_metric: [B, T, 1, H, W] (原始分辨率, Metric)
    Returns:
        unet_input: [B*T, 9, h, w]
    """
    B, T, _, h, w = color_latents.shape
    
    # 1. Sparse: Metric -> Inverse
    sparse_inv = 1.0 / (sparse_depth_metric + 1e-6)
    sparse_inv[sparse_depth_metric < 1e-6] = 0.0 # 无效点设为0
    
    # 2. 展平并 Resize 到 Latent 大小
    # [B, T, 1, H, W] -> [B*T, 1, H, W]
    sparse_flat = rearrange(sparse_inv, 'b t c h w -> (b t) c h w')
    
    # 使用 nearest 插值，因为这是稀疏点，bilinear 会导致点扩散变糊
    sparse_resized = F.interpolate(sparse_flat, size=(h, w), mode='nearest')
    
    # 3. 归一化 (可选，但推荐)
    # 因为 Latent 通常在 -1~1 或 -4~4 之间，而 1/depth 可能很大
    # 简单的做法是把 sparse 也映射到类似范围，或者 trust 卷积层去学 scale
    # 这里不做额外处理，依赖 conv_in 的学习能力
    
    # 4. 拼接
    color_flat = rearrange(color_latents, 'b t c h w -> (b t) c h w')
    noise_flat = rearrange(noisy_depth_latents, 'b t c h w -> (b t) c h w')
    
    # Concatenate: 4 + 4 + 1 = 9 Channels
    unet_input = torch.cat([color_flat, noise_flat, sparse_resized], dim=1)
    
    return unet_input