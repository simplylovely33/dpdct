import torch
import types
from loguru import logger

OPTIMIZER_REGISTRY = {}

def register(name):
    def decorator(fn):
        OPTIMIZER_REGISTRY[name] = fn
        return fn
    return decorator

@register("adamw")
def _build_adamw(cfg, model):
    opt_cfg = cfg['OPTIMIZER']
    if hasattr(model, 'parameters'):
        params = model.parameters()
    else:
        params = model
        if isinstance(params, (types.GeneratorType, filter)):
            params = list(params)
    return torch.optim.AdamW(
        params,
        lr=float(opt_cfg['LR']),
        weight_decay=float(opt_cfg['WEIGHT_DECAY']),
    )

@register("sgd")
def _build_sgd(cfg, model):
    opt_cfg = cfg['OPTIMIZER']
    if hasattr(model, 'parameters'):
        params = model.parameters()
    else:
        params = model
        if isinstance(params, (types.GeneratorType, filter)):
            params = list(params)
    return torch.optim.SGD(
        params,
        lr=float(opt_cfg['LR']),
        momentum=float(opt_cfg['MOMENTUM']),
    )

def build_optimizer(cfg, model):
    name = cfg['OPTIMIZER']['NAME']
    if name not in OPTIMIZER_REGISTRY:
        logger.error(
            f"Optimizer '{name}' not registered. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
        raise KeyError(name)

    optimizer = OPTIMIZER_REGISTRY[name](cfg, model)

    logger.info(
        f"Building Optimizer: {optimizer.__class__.__name__} | "
        f"lr={optimizer.param_groups[0]['lr']} | "
        f"weight_decay={optimizer.param_groups[0].get('weight_decay', 0)} | "
        f"momentum={optimizer.param_groups[0].get('momentum', None)}"
    )

    return optimizer
