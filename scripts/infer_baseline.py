import os
import sys
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDIMScheduler

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from models.marigold_dc import MarigoldDepthCompletionPipeline
from dataset.endo import EndoscopyDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def tensor_to_pil(tensor):
    """
    [C, H, W] Tensor (0-1) -> PIL Image
    """
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def tensor_to_numpy_depth(tensor):
    """
    [1, H, W] Tensor -> [H, W] Numpy array
    """
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    return tensor.squeeze(0).cpu().numpy()

def apply_cmap(depth, valid_mask=None):
    """
    将深度图归一化并上色 (Inferno)
    depth: [H, W] numpy array
    """
    if valid_mask is None:
        valid_mask = depth > 1e-6
        
    if valid_mask.sum() > 0:
        # 使用分位数进行鲁棒归一化 (排除极值噪点)
        d_min = np.percentile(depth[valid_mask], 1)
        d_max = np.percentile(depth[valid_mask], 99)
        norm = (depth - d_min) / (d_max - d_min + 1e-8)
        norm = np.clip(norm, 0, 1)
        norm[~valid_mask] = 0
    else:
        norm = np.zeros_like(depth)
        
    vis = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis[~valid_mask] = 0 # 背景置黑
    return vis

def visualize_result(rgb_pil, sparse_inv, gt_metric, pred_inv, save_path):
    """
    拼接 RGB, Sparse, GT, Pred 并保存
    注意：为了视觉对比方便，我们将 GT Metric 转为 Inverse Depth 再可视化
    """
    w, h = rgb_pil.size
    
    # 1. RGB
    rgb_np = np.array(rgb_pil)
    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    
    # 2. Sparse (Input is already Inverse Depth 1/d)
    sparse_vis = apply_cmap(sparse_inv, valid_mask=sparse_inv > 0)
    
    # 3. GT (Input is Metric Depth d -> Convert to 1/d for comparison)
    mask_gt = gt_metric > 1e-6
    gt_inv = np.zeros_like(gt_metric)
    gt_inv[mask_gt] = 1.0 / gt_metric[mask_gt]
    gt_vis = apply_cmap(gt_inv, valid_mask=mask_gt)
    
    # 4. Pred (Output matches Sparse, so it is 1/d)
    pred_vis = apply_cmap(pred_inv)
    
    # Resize consistency
    sparse_vis = cv2.resize(sparse_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_vis = cv2.resize(gt_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_vis = cv2.resize(pred_vis, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 添加文字标签
    def add_label(img, text):
        cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    rgb_bgr = add_label(rgb_bgr, "RGB")
    sparse_vis = add_label(sparse_vis, "Sparse Input (1/d)")
    gt_vis = add_label(gt_vis, "GT Depth (Inv 1/d)")
    pred_vis = add_label(pred_vis, "Marigold Pred")

    # Concatenate: 2x2 Grid or 1x4 Row? Let's do 1x4 Row
    combined = np.hstack([rgb_bgr, gt_vis, sparse_vis, pred_vis])
    cv2.imwrite(save_path, combined)

def main():
    # 1. 配置路径
    config_path = "configs/config.yaml" 
    output_dir = "debug/baseline_infer"
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备模型 (Marigold-DC)
    print("正在加载 Marigold-DC 模型...")
    checkpoint_path = cfg['MARIGOLD_DC']['CHECKPOINT']
    
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        checkpoint_path, 
        prediction_type="depth",
        local_files_only=True
    ).to(device, dtype=torch.float32)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. 准备数据
    print("正在初始化 Dataset...")
    dataset = EndoscopyDataset(cfg, mode='val') 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"开始推理，结果将保存至: {output_dir}")

    # 4. 推理循环
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= 10: break 
        
        T = batch['color'].shape[1]
        center_idx = T // 2
        
        # 提取数据
        rgb_tensor = batch['color'][0, center_idx]
        rgb_pil = tensor_to_pil(rgb_tensor)
        
        # Sparse (Inverse Depth)
        sparse_tensor = batch['sparse'][0, center_idx]
        sparse_numpy = tensor_to_numpy_depth(sparse_tensor)
        
        # GT (Metric Depth) - 新增
        gt_tensor = batch['depth'][0, center_idx]
        gt_numpy = tensor_to_numpy_depth(gt_tensor)
        
        try:
            # 运行 Marigold-DC (输入 Sparse 1/d, 输出 Pred 1/d)
            pred_depth = pipe(
                image=rgb_pil,
                sparse_depth=sparse_numpy,
                num_inference_steps=50, 
                ensemble_size=1,
                processing_resolution=768, 
                seed=2024
            )
            
            # 保存结果
            save_name = f"sample_{i:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            visualize_result(rgb_pil, sparse_numpy, gt_numpy, pred_depth, save_path)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print("测试完成！")

if __name__ == "__main__":
    main()