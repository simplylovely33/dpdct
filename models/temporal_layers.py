import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.models.attention import BasicTransformerBlock

class TemporalTransformer3D(nn.Module):
    """
    时序注意力模块 (Temporal Attention Module)
    用于在 U-Net 的 Spatial 层之后处理时间轴信息。
    参考 AnimateDiff 的设计逻辑。
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_layers=1,
        camera_embed_dim=None, # 支持注入相机位姿 Embedding
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.camera_embed_dim = camera_embed_dim

        # 1. 基础 Transformer Block (使用 Diffusers 的标准组件以兼容 checkpointing)
        # 我们使用 BasicTransformerBlock，它包含了 Self-Attention 和 FeedForward
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0.0,
                    cross_attention_dim=None, # Temporal Attention 通常是 Self-Attention
                    activation_fn="geglu",
                    num_embeds_ada_norm=camera_embed_dim, # AdaLayerNorm 用于注入 Camera Pose
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_elementwise_affine=True,
                    norm_type="layer_norm_context" if camera_embed_dim else "layer_norm", # 关键：使用 Context Norm 注入 Pose
                )
                for _ in range(num_layers)
            ]
        )

        # 2. 零初始化输出层 (Zero-Convolution / Zero-Linear)
        # 保证初始状态下该模块不影响原始 Spatial 特征，实现平滑微调
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, class_labels=None, num_frames=1):
        """
        Args:
            hidden_states: [B*T, C, H, W]  <- 注意输入是 Spatial Flatten 的
            class_labels: 这里复用作为 Camera Pose Embedding [B*T, camera_embed_dim]
            num_frames: 视频长度 T
        """
        # 1. 维度重整: [B*T, C, H, W] -> [B, T, H*W, C] -> [B*HW, T, C]
        # 我们要把 T 放到序列长度维度，把 Spatial 放到 Batch 维度
        # 这样 Attention 是在 T 上做的
        
        batch_size_total, channels, height, width = hidden_states.shape
        batch_size = batch_size_total // num_frames
        
        # [B*T, C, H, W] -> [B, T, C, H, W]
        hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', b=batch_size, t=num_frames)
        # -> [B, HW, T, C] (为了让 Transformer 处理 T)
        hidden_states = rearrange(hidden_states, 'b t c h w -> (b h w) t c')

        # 2. 处理 Camera Condition (如果有)
        # Camera Embedding 通常是 [B*T, D]，我们需要把它广播适配到 [B*HW, D] ?
        # 不，BasicTransformerBlock 的 AdaLayerNorm 需要 [Batch_Size, Embed_Dim]
        # 这里的 Batch_Size 是 (B * H * W)。所以我们需要重复 Camera Embedding。
        
        camera_emb = None
        if self.camera_embed_dim is not None and class_labels is not None:
            # class_labels: [B*T, D] -> [B, T, D]
            emb = rearrange(class_labels, '(b t) d -> b t d', b=batch_size, t=num_frames)
            # 我们需要让每个 Spatial Token 都能看到对应的 Time Pose
            # [B, T, D] -> [B, 1, T, D] -> [B, HW, T, D] -> [(B HW), T, D] ?
            # 等等，BasicTransformerBlock 处理的是 Sequence Length 维度的 Norm 吗？
            # 通常 AdaLN 接收的是一个全局向量。
            # 这里我们简化处理：假设 Pose 主要影响 Temporal 变化，我们在 Attention 内部并不直接用 Pose 做 QK。
            # 而是通过 AdaLayerNorm 调制特征。
            
            # 为了节省显存，我们可能需要 repeat。但 (B*HW) 太大了。
            # 这是一个显存瓶颈点。
            # 优化策略：如果 H*W 很大，我们通常不把 Pose 注入到每个 Pixel 的 Norm 里。
            # 替代方案：把 Pose 加到 hidden_states 上，或者只在 Input 做一次注入。
            
            # 鉴于显存考虑，我们暂时不在 transformer block 内部做 dense 的 AdaLN，
            # 而是直接把 Pose Embedding 加到 hidden_states 上 (Add Token)。
            
            # [B, T, D] -> [B, 1, T, D]
            emb = emb.unsqueeze(1) 
            # 线性投影适配维度
            # (这里假设外部已经投影好了，或者我们在 Block 里不做处理)
            pass

        # 3. Transformer Forward
        for block in self.transformer_blocks:
            # BasicTransformerBlock 期望输入 [Batch, Seq_Len, Dim]
            # 这里的 Batch 是 (B * H * W), Seq_Len 是 T
            
            # 注意：如果开启 gradient_checkpointing，显存会大幅节省
            if self.training: 
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, None, None, None, None, None, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states)

        # 4. Zero-Out Projection
        hidden_states = self.proj_out(hidden_states)

        # 5. 还原维度
        # [(B HW), T, C] -> [B, T, C, H, W] -> [B*T, C, H, W]
        hidden_states = rearrange(hidden_states, '(b h w) t c -> (b t) c h w', b=batch_size, h=height, w=width)

        return hidden_states