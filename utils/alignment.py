import torch
import numpy as np

def align_scale_shift_global(pred_video, target_video, mask_video):
    """
    计算全局最佳的 s, t 使得 s * pred_video + t ≈ target_video
    输入形状均为 [T, H, W] 或展平的一维 tensor
    """
    # 提取所有有效像素
    p = pred_video[mask_video].view(-1)
    t = target_video[mask_video].view(-1)
    
    # 保护机制：如果有效点太少，不对齐
    if p.numel() < 10:
        return 1.0, 0.0
    
    # 构造最小二乘线性方程 Ax = b
    # A = [p, 1]
    ones = torch.ones_like(p)
    A = torch.stack([p, ones], dim=1) # [N, 2]
    b = t
    
    # 求解
    try:
        X = torch.linalg.lstsq(A, b).solution
        s, t = X[0].item(), X[1].item()
    except Exception:
        s, t = 1.0, 0.0
        
    return s, t

def compute_global_metrics(pred_video, target_video, min_depth=1e-3, max_depth=10.0):
    """
    对整个视频序列进行一次性全局对齐，并计算指标
    pred_video: 网络预测的深度序列 [T, H, W]
    target_video: 真实深度序列 [T, H, W]
    """
    # 1. 获取全局有效 Mask
    valid_mask = (target_video > min_depth) & (target_video < max_depth)
    
    if valid_mask.sum() == 0:
        return {
            "mae": 0.0, "rmse": 0.0, 
            "pred_aligned": pred_video, 
            "scale": 1.0, "shift": 0.0
        }
    
    # 2. 计算全局 Scale & Shift
    s, t = align_scale_shift_global(pred_video, target_video, valid_mask)
    
    # 3. 将全局参数统一应用到整个视频 (保证时序一致性，不闪烁)
    pred_aligned = pred_video * s + t
    pred_aligned = torch.clamp(pred_aligned, min=min_depth, max=max_depth)
    
    # 4. 计算全局误差
    diff = pred_aligned[valid_mask] - target_video[valid_mask]
    abs_diff = diff.abs()
    
    global_mae = abs_diff.mean().item()
    global_rmse = (diff ** 2).mean().sqrt().item()
    
    return {
        "mae": global_mae, 
        "rmse": global_rmse, 
        "pred_aligned": pred_aligned,
        "scale": s,
        "shift": t
    }