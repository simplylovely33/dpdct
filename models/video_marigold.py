import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any, Union

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# 继承基础 Pipeline
from models.marigold_dc import MarigoldDepthCompletionPipeline
from models.temporal_layers import TemporalTransformer3D

class VideoMarigoldPipeline(MarigoldDepthCompletionPipeline):
    """
    支持时序一致性的 Video Marigold Pipeline。
    纯净版：移除了 CameraPoseEncoder，完全依靠 10通道 Sparse 输入和 Temporal Attention 提供时空约束。
    """
    
    def setup_video_model(self, num_frames=5):
        """
        初始化模型结构：修改输入层，注入时序层，冻结 Spatial 层。
        必须在加载预训练权重后调用。
        """
        self.num_frames = num_frames
        
        # --- 1. 修改输入层 (10 通道) ---
        old_conv = self.unet.conv_in
        if old_conv.in_channels != 10:
            logger.info(f"Modifying conv_in: {old_conv.in_channels} -> 10 channels")
            new_conv = nn.Conv2d(
                10, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                padding=old_conv.padding
            )
            # 零初始化策略
            with torch.no_grad():
                new_conv.weight[:, :8] = old_conv.weight # RGB(4) + Noise(4)
                new_conv.weight[:, 8:] = 0.0             # Sparse(1) + Mask(1)
                new_conv.bias = old_conv.bias
            
            self.unet.conv_in = new_conv
        
        # --- 2. 注入 Temporal Layers ---
        logger.info("Injecting Temporal Layers...")
        
        modules_to_patch = {}
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == "Transformer2DModel":
                dim = module.config.in_channels if hasattr(module.config, "in_channels") else module.norm.weight.shape[0]
                heads = module.config.num_attention_heads
                head_dim = module.config.attention_head_dim
                
                temporal_layer = TemporalTransformer3D(
                    dim=dim,
                    num_attention_heads=heads,
                    attention_head_dim=head_dim,
                    num_layers=1, 
                    camera_embed_dim=None 
                )
                
                module.temporal_layer = temporal_layer
                
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                
                def new_forward(self_module, hidden_states, encoder_hidden_states=None, *args, **kwargs):
                    # 分离 num_frames，不传给 spatial layer
                    cross_attention_kwargs = kwargs.get("cross_attention_kwargs", {})
                    T = 1
                    spatial_kwargs = kwargs.copy()
                    
                    if cross_attention_kwargs is not None:
                        spatial_cross_attn_kwargs = cross_attention_kwargs.copy()
                        if "num_frames" in spatial_cross_attn_kwargs:
                            T = spatial_cross_attn_kwargs.pop("num_frames")
                        spatial_kwargs["cross_attention_kwargs"] = spatial_cross_attn_kwargs
                    
                    # 1. 先跑 Spatial
                    spatial_out = self_module._original_forward(hidden_states, encoder_hidden_states, *args, **spatial_kwargs)
                    
                    if isinstance(spatial_out, tuple):
                        x = spatial_out[0]
                    elif hasattr(spatial_out, "sample"):
                        x = spatial_out.sample
                    else:
                        x = spatial_out
                        
                    # 2. 再跑 Temporal
                    temporal_out = self_module.temporal_layer(x, num_frames=T)
                    output = x + temporal_out
                    
                    # 保持原有返回结构
                    if isinstance(spatial_out, tuple):
                        return (output,) + spatial_out[1:]
                    elif hasattr(spatial_out, "sample"):
                        spatial_out.sample = output
                        return spatial_out
                    else:
                        return output

                import types
                module.forward = types.MethodType(new_forward, module)
                modules_to_patch[name] = module

        logger.info(f"Patched {len(modules_to_patch)} Spatial Transformers with Temporal Layers.")

        # --- 3. 冻结与解冻 ---
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, module in self.unet.named_modules():
            if hasattr(module, "temporal_layer"):
                module.temporal_layer.requires_grad_(True)

    def to(self, *args, **kwargs):
        # 移除了 pose_encoder 的显式设备转移
        super().to(*args, **kwargs)
        return self

    # --- 辅助函数 ---
    def _get_empty_text_embedding(self, device, dtype):
        if self.empty_text_embedding is None:
            logger.info("Initializing empty text embedding...")
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    "", 
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]
        
        return self.empty_text_embedding.to(device, dtype=dtype)

    def train_step(self, batch, generator=None):
        """
        统一的训练步逻辑
        """
        device = self.unet.device
        dtype = self.unet.dtype 
        
        # 1. 准备数据
        rgb = batch['color'].to(device, dtype=dtype) 
        sparse = batch['sparse'].to(device, dtype=dtype)
        gt_depth = batch['depth'].to(device, dtype=dtype)
        # 不再强制需要 Pose 和 K 用于前向传播，但后续加 Geometric Loss 还会用到
        
        B, T = rgb.shape[:2]
        
        # 2. VAE Encoding (RGB)
        rgb_flat = rgb.view(B*T, 3, *rgb.shape[-2:])
        latents = self.encode_rgb(rgb_flat) 
        
        # 3. VAE Encoding (GT Depth)
        from models.temporal_module import depth_transform
        gt_norm = depth_transform(gt_depth) 
        gt_3c = gt_norm.repeat(1, 1, 3, 1, 1).view(B*T, 3, *rgb.shape[-2:])
        depth_latents = self.encode_rgb(gt_3c) 
        
        # 4. Prepare Sparse Condition
        mask = (sparse > 1e-4).float() 
        sparse_norm = depth_transform(sparse)
        sparse_norm = sparse_norm * mask + (-1.0) * (1 - mask)
        
        h, w = latents.shape[-2:]
        sparse_small = torch.nn.functional.interpolate(
            sparse_norm.view(B*T, 1, *rgb.shape[-2:]), size=(h, w), mode='nearest'
        )
        mask_small = torch.nn.functional.interpolate(
            mask.view(B*T, 1, *rgb.shape[-2:]), size=(h, w), mode='nearest'
        )
        
        # 5. Add Noise
        noise = torch.randn_like(depth_latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B*T,), device=device).long()
        noisy_latents = self.scheduler.add_noise(depth_latents, noise, timesteps)
        
        # 6. UNet Input (10 通道)
        unet_input = torch.cat([latents, noisy_latents, sparse_small, mask_small], dim=1)
        
        # 7. Forward
        empty_text_embed = self._get_empty_text_embedding(device, dtype)
        encoder_hidden_states = empty_text_embed.repeat(B*T, 1, 1) 
        
        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"num_frames": T}
        ).sample
        
        return noise_pred, noise, depth_latents
    
    def encode_rgb(self, rgb):
        x = 2.0 * rgb - 1.0
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor