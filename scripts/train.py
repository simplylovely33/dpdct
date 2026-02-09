import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

import utils.toolkit as utils

from loguru import logger
from tqdm import tqdm
from einops import rearrange

from optimizer import build_optimizer 
from scheduler import build_scheduler
from dataset import EndoscopyDataset, build_dataloader
from models.stmarigold_dc import SpatioTemporalMarigoldPipeline
from models.temporal_module import *
from loss.metric import AverageMeter, GeometricConsistencyLoss
from infer import *



if __name__ == '__main__':
    cfg, device = utils.initialization('configs/config.yaml')
    debug_dir = cfg['DEBUG']['ROOT']
    os.makedirs(debug_dir, exist_ok=True)

    train_loader = build_dataloader(cfg, EndoscopyDataset, mode="train")
    val_loader = build_dataloader(cfg, EndoscopyDataset, mode='val')

    logger.info("Loading checkpoint for Marigold-DC Model")
    dtype = eval(cfg['MARIGOLD_DC']['DTYPE'])
    pipe = SpatioTemporalMarigoldPipeline.from_pretrained(
        cfg['MARIGOLD_DC']['CHECKPOINT'], 
        torch_dtype=dtype,
        local_files_only=True
    )
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    if getattr(pipe, "empty_text_embedding", None) is None:
        seq_len = 77
        cross_attention_dim = pipe.unet.config.cross_attention_dim
        if cross_attention_dim is None:
            cross_attention_dim = 1024
        pipe.empty_text_embedding = torch.zeros(
            (1, seq_len, cross_attention_dim), 
            device=device, 
            dtype=dtype
        )
        logger.warning(
            f"Empty text embedding not found. Created ete with shape: {pipe.empty_text_embedding.shape}")

    generator = torch.Generator(device=device).manual_seed(42)
    _temporal_params_generator = pipe.inject_temporal_layers(
        num_frames=cfg['DATA']['CONTEXT_SIZE'],
        camera_embed_dim=cfg['DATA']['CAMERA_EMBED_DIM']
    )
    temporal_params = list(_temporal_params_generator)

    camera_encoder = CameraPoseEncoder().to(device)
    geo_criterion = GeometricConsistencyLoss(
        vae=pipe.vae, 
        scheduler=pipe.scheduler,
        sample_pairs=1  # 【关键】每次迭代只随机抽取 4 对相邻帧算几何Loss，防止爆显存
    ).to(device)

    optimizer = build_optimizer(cfg, temporal_params)
    optimizer.add_param_group({'params': camera_encoder.parameters(), 'lr': 1e-4})
    scheduler = build_scheduler(cfg, optimizer, train_loader)

    pipe.scheduler.set_timesteps(1000)
    pipe.unet.train()

    # logger.info(f"Start Training: {len(dataset)} videos, {len(train_loader)} batches per epoch.")
    for epoch in range(cfg['TRAIN']['EPOCH']):
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            color = batch['color'].to(device, dtype=dtype)
            sparse = batch['sparse'].to(device, dtype=dtype)
            gt_depth = batch['depth'].to(device, dtype=dtype)
            pose = batch['pose'].to(device)
            intrinsic = batch['intrinsic'].to(device)
            
            B, T, C, H, W = color.shape
            intr_flat, pose_flat = utils.normalize_flatten_camrea_info(intrinsic, pose, W, H)
            camera_embedding = camera_encoder(pose_flat, intr_flat)

            color_latents = pipe.encode_batch_rgb(color, generator)
            depth_latents = pipe.encode_batch_depth(gt_depth, generator)
            
            latent_h, latent_w = color_latents.shape[-2:]
            noise = pipe.generate_correlated_noise(
                batch_size=B, num_frames=T, latent_shape=(4, latent_h, latent_w),
                poses=pose, intrinsics=intrinsic, sparse_depths=sparse,
                generator=generator, correlation_strength=0.8, debug=False
            )

            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=device).long()
            timesteps_expanded = timesteps.repeat_interleave(T)

            depth_latents_flat = rearrange(depth_latents, 'b t c h w -> (b t) c h w')
            noise_flat = rearrange(noise, 'b t c h w -> (b t) c h w')
            
            noisy_depth_latents = pipe.scheduler.add_noise(
                depth_latents_flat, noise_flat, timesteps_expanded
            )

            color_latents_flat = rearrange(color_latents, 'b t c h w -> (b t) c h w')
            unet_input = prepare_unet_input(
                color_latents, 
                rearrange(noisy_depth_latents, '(b t) c h w -> b t c h w', b=B), 
                sparse, 
                latent_h, latent_w
            )

            empty_text_embed = pipe.empty_text_embedding.to(device, dtype=dtype) # [1, 2, 1024]
            batch_text_embed = empty_text_embed.repeat(B*T, 1, 1)

            model_pred = pipe.unet(
                unet_input, 
                timesteps_expanded, 
                encoder_hidden_states=batch_text_embed,
                cross_attention_kwargs={"camera_emb": camera_embedding}
            ).sample

            loss_mse = F.mse_loss(model_pred.float(), noise_flat.float(), reduction="mean")
            if cfg['TRAIN']['USE_GEO_LOSS']:
                loss_geo = geo_criterion(
                    noisy_latents=noisy_depth_latents, # 传入加噪后的 latents
                    model_pred=model_pred,             # 传入模型预测的 noise
                    timesteps=timesteps_expanded, 
                    pose=batch['pose'].to(device),          # [B, T, 4, 4]
                    intrinsic=batch['intrinsic'].to(device), # [B, T, 3, 3]
                    batch_size=B,
                    num_frames=T
                )
                loss = loss_mse + 0.05 * loss_geo
            else:
                loss = loss_mse
            loss.backward()

            all_params = [p for group in optimizer.param_groups for p in group['params']]
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_meter.update(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg': f"{loss_meter.avg:.4f}",
                'grad': f"{total_norm.item():.2f}",
                'lr': f"{current_lr:.2e}"
            })
        
        if epoch != 0 and epoch % 5 == 0:
            logger.debug("Executing denoising process...")
            run_inference_preview(
                pipe, 
                camera_encoder, 
                val_loader, 
                device, 
                dtype, 
                generator, 
                epoch,
                num_batches=5
            )


