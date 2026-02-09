import shutil
from pathlib import Path
from loguru import logger

def clean_scared_generated_files(root_dir: str):
    root_path = Path(root_dir)
    if not root_path.exists():
        logger.error(f"路径不存在: {root_path}")
        return

    # 我们要删除的目标文件夹或文件名称
    targets_to_delete = {
        "left",
        "left_undistorted",
        "depthmap",
        "depthmap_undistorted"
    }

    # 遍历所有 keyframe 下的 data 目录
    # 结构: dataset_X / keyframe_Y / data
    data_dirs = list(root_path.rglob("data"))
    
    logger.info(f"找到 {len(data_dirs)} 个 data 目录，开始清理...")

    deleted_count = 0
    
    for data_dir in data_dirs:
        for item in data_dir.iterdir():
            if item.name in targets_to_delete:
                try:
                    if item.is_dir():
                        shutil.rmtree(item) # 删除文件夹及其内容
                        logger.info(f"🗑️ 已删除文件夹: {item}")
                    else:
                        item.unlink()       # 删除文件
                        logger.info(f"🗑️ 已删除文件:   {item}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"❌ 删除失败 {item}: {e}")

    logger.success(f"清理完成！共删除了 {deleted_count} 个项目。")
    logger.info("原始文件 (rgb.mp4, *.tar.gz) 已保留。")

if __name__ == "__main__":
    # 请修改为你的根目录
    root = "/home/wsco/local/yyz/data/scared"
    clean_scared_generated_files(root)