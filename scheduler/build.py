import torch
from loguru import logger
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

SCHEDULER_REGISTRY = {}

def register(name):
    def decorator(fn):
        SCHEDULER_REGISTRY[name] = fn
        return fn
    return decorator

@register("diffusers_cosine")
def _build_diffusers_cosine(cfg, optimizer, num_training_steps):
    sched_cfg = cfg.get('SCHEDULER', {})
    
    # 获取 Warmup 步数，支持百分比(0.1) 或 绝对步数(500)
    warmup_steps = sched_cfg.get('WARMUP_STEPS', 0)
    if isinstance(warmup_steps, float): # 如果是 0.1，则占总步数的 10%
        warmup_steps = int(num_training_steps * warmup_steps)
    
    logger.info(f"Building Scheduler: Diffusers Cosine | Warmup: {warmup_steps} | Total: {num_training_steps}")
    
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

@register("multistep")
def _build_multistep(cfg, optimizer, num_training_steps=None):
    sched_cfg = cfg['SCHEDULER']
    milestones = sched_cfg.get('MILESTONES', [10, 20])
    gamma = float(sched_cfg.get('GAMMA', 0.1))
    
    logger.info(f"Building Scheduler: MultiStepLR | Milestones: {milestones} | Gamma: {gamma}")
    
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

@register("cosine")
def _build_cosine(cfg, optimizer, num_training_steps):
    T_max = cfg['TRAIN']['EPOCH'] 
    eta_min = float(cfg['SCHEDULER'].get('MIN_LR', 1e-6))
    
    logger.info(f"Building Scheduler: CosineAnnealingLR | T_max: {T_max} | Min LR: {eta_min}")
    
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
    )

def build_scheduler(cfg, optimizer, train_loader=None):
    """
    Args:
        cfg: 全局配置
        optimizer: 创建好的优化器
        train_loader: 用于计算总训练步数 (epochs * len(loader))
    """
    sched_cfg = cfg.get('SCHEDULER', {})
    name = sched_cfg.get('NAME', 'diffusers_cosine') # 默认使用 diffusers_cosine

    if name not in SCHEDULER_REGISTRY:
        logger.warning(
            f"Scheduler '{name}' not registered. Using None. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
        return None

    num_training_steps = 0
    if train_loader is not None:
        num_epochs = cfg['TRAIN']['EPOCH']
        num_training_steps = len(train_loader) * num_epochs

    scheduler = SCHEDULER_REGISTRY[name](cfg, optimizer, num_training_steps)
    return scheduler