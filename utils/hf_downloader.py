import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import argparse
from loguru import logger
from huggingface_hub import snapshot_download


def download_huggingface_repo(repo_id, repo_type, save_path):
    logger.info(f"Configure mirror -> https://hf-mirror.com")
    logger.info(f"Prepare download -> ID: {repo_id} | Type: {repo_type}")
    logger.info(f"Save Path -> {save_path}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=save_path
        )
        logger.success(f"Download Successfully ")

    except Exception as e:
        logger.error(f"Download Failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace 数据集/模型下载工具 (自动使用镜像源)")
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True, 
        default="maxhallan7/scared", 
        help="HuggingFace的仓库ID: 一般是<用户名/仓库名>组合 网页左上角带复制图标的标题"
    )
    parser.add_argument(
        "--repo_type", 
        type=str, 
        default="dataset", 
        choices=["dataset", "model", "space"],
        help="HuggingFace的仓库类型: dataset(默认), model, space"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        default="/home/wsco/local/yyz/data/zip/scared",
        help="本地保存路径"
    )

    args = parser.parse_args()
    download_huggingface_repo(args.repo_id, args.repo_type, args.save_path)
    