import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed

# 还需要完善一下检查机制，就是重复运行的时候，哪些已经解压完或者已经得到结果的rgb图像就不再处理了

def worker_extract_archive(dir_root: Path):
    try:
        folder_name = dir_root.name.split('.')[0]
        extract_path = dir_root.parent / folder_name

        should_extract = True
        with tarfile.open(dir_root, "r:gz") as tar:
            members = tar.getmembers()
            tar_file_count = len([m for m in members if m.isfile()])
            
            if extract_path.exists():
                current_files = list(extract_path.glob("*"))
                current_file_count = len([f for f in current_files if f.is_file()])
                if current_file_count == tar_file_count and tar_file_count > 0:
                    should_extract = False
            
            if should_extract:
                extract_path.mkdir(parents=True, exist_ok=True)
                tar.extractall(path=extract_path, members=members)
                logger.success(f"Unzip Success: {dir_root}")
            else:
                logger.info(f"Skipped (Already Exists): {dir_root}")
    
    except Exception as e:
        logger.error(f"Unzip Failure: {dir_root.name} -> {e}")

def parallel_process_scared_data(dir_root: Path, max_workers=8):
    if not dir_root.exists():
        logger.error(f"SCARED data root doesn't exist: {dir_root}")
        return

    target_files = {"frame_data.tar.gz", "scene_points.tar.gz"}
    archive_tasks = [f for f in dir_root.rglob("*.tar.gz") if f.name in target_files]
    logger.info(f"🚀 Start Unzipping {len(archive_tasks)} files with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_extract_archive, f) for f in archive_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Unzipping"):
            result = future.result()
            #logger.info(result) 
    logger.success("🎉 All Parallel Tasks Finished!")
    
def main():
    parser = argparse.ArgumentParser(description="Extract the Archive from Downloaded SCARED Zip")
    parser.add_argument(
        "--scared_root", 
        default='/home/wsco/local/yyz/data/scared', 
        help="Path to the dataset root folder. Default is './data'."
    )
    args = parser.parse_args()

    parallel_process_scared_data(Path(args.scared_root), max_workers=4)

if __name__ == "__main__":
    main()