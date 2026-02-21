import torch

def depth_transform(depth_map, min_val=0.001, max_val=0.2):
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