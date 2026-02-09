import torch
from loguru import logger
from torch.utils.data import DataLoader

DATALOADER_REGISTRY = {}

def register(name):
    def decorator(fn):
        DATALOADER_REGISTRY[name] = fn
        return fn
    return decorator

@register("default")
def _build_default_dataloader(cfg, dataset, mode):
    dataset = dataset(cfg, mode=mode)
    dl_cfg = cfg['DATALOADER'][mode.upper()]
    return DataLoader(
        dataset,
        batch_size=dl_cfg['BATCH_SIZE'],
        num_workers=cfg['DATALOADER']['NUM_WORKERS'],
        shuffle=dl_cfg['SHUFFLE'],
        drop_last=dl_cfg['DROP_LAST'],
        pin_memory=True,
    )

def build_dataloader(cfg, dataset, mode):
    name = cfg['DATALOADER'].get('NAME', 'default')
    if name not in DATALOADER_REGISTRY:
        logger.error(
                f"Dataloader '{name}' not registered. "
                f"Available: {list(DATALOADER_REGISTRY.keys())}"
            )
        raise KeyError(name)
    
    dataloader = DATALOADER_REGISTRY[name](cfg, dataset, mode)

    logger.info(
        f"Building Dataloader : {mode} | "
        f"batch_size={dataloader.batch_size} | "
        f"num_workers={dataloader.num_workers} | "
        f"dataset_size={len(dataloader.dataset)}"
    )
    return dataloader