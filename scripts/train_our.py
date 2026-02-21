import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from diffusers import DDPMScheduler, DDIMScheduler

# 添加路径
sys.path.append(os.getcwd())

from models.video_marigold import VideoMarigoldPipeline
from dataset.endo import EndoscopyDataset
from models.temporal_module import depth_transform

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=1, help="Video batch size")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")
    
    # 1. 初始化模型
    logger.info("Loading Video Marigold Model...")
    
    # [修改点 1] 这里不要急着 .to(device)
    pipe = VideoMarigoldPipeline.from_pretrained(
        cfg['MARIGOLD_DC']['CHECKPOINT'],
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # 初始化视频模块 (此时新插入的层在 CPU)
    pipe.setup_video_model(num_frames=cfg['DATA']['CONTEXT_SIZE'])
    
    # [修改点 2] 结构修改完成后，统一移动到 GPU
    pipe = pipe.to(device)
    
    # 开启梯度检查点 (省显存)
    pipe.unet.enable_gradient_checkpointing()
    
    # 2. 准备数据
    train_dataset = EndoscopyDataset(cfg, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 3. 优化器
    # 收集需要训练的参数: conv_in, temporal_layers, pose_encoder
    trainable_params = []
    trainable_params += list(pipe.unet.conv_in.parameters())
    
    for name, module in pipe.unet.named_modules():
        if hasattr(module, "temporal_layer"):
            trainable_params += list(module.temporal_layer.parameters())
            
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # 4. 训练循环
    logger.info(f"Start training with {len(trainable_params)} tensor groups...")
    global_step = 0
    
    for epoch in range(args.epochs):
        pipe.unet.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # train_step 返回: noise_pred, noise, gt_latents
            # 确保 batch 数据被正确移动到了 device (由 pipe.device 控制)
            noise_pred, noise, gt_latents = pipe.train_step(batch)
            
            # --- MSE Loss ---
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())
            
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            save_dir = cfg['DEBUG']['ROOT']
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            
            # 保存关键部分的权重
            torch.save({
                'unet_state': pipe.unet.state_dict(),
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()