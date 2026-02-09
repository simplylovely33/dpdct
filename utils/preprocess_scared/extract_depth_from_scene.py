import cv2
import argparse
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed

# 还需要完善一下检查机制，就是重复运行的时候，哪些已经解压完或者已经得到结果的rgb图像就不再处理了

def worker_process_single_tiff(tiff_file: Path):
    try:
        parent_dir = tiff_file.parent
        output_dir = parent_dir / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = tiff_file.stem
        number_str = "".join(filter(str.isdigit, stem))
        if not number_str:
            number_str = stem
            
        save_path = output_dir / f"{number_str}.png"
        cloud = tifffile.imread(str(tiff_file))
        if cloud.ndim == 3 and cloud.shape[-1] == 3:
            depth_map = cloud[:, :, -1]
        else:
            depth_map = cloud
            
        depth_map = np.nan_to_num(depth_map, nan=0.0, neginf=0.0, posinf=0.0)
        depth_map[depth_map < 0] = 0
        depth_uint16 = depth_map.astype(np.uint16)
        cv2.imwrite(str(save_path), depth_uint16)
        
        return None

    except Exception as e:
        return f"❌ Error: {tiff_file.name} -> {e}"
    
def parallel_process_scared_data(dir_root: Path, max_workers=8):
    if not dir_root.exists():
        logger.error(f"SCARED data root doesn't exist: {dir_root}")
        return

    logger.info("🔍 Scanning all TIFF files...")
    all_tiff_files = []
    scene_folders = list(dir_root.rglob("scene_points"))
    for folder in scene_folders:
        all_tiff_files.extend(list(folder.glob("*.tiff")))
        
    total_files = len(all_tiff_files)
    if total_files == 0:
        logger.warning("No TIFF files found!")
        return

    logger.info(f"🚀 Found {total_files} images. Processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_process_single_tiff, f) for f in all_tiff_files]
        for future in tqdm(as_completed(futures), total=total_files, desc="Converting Depth", unit="img"):
            result = future.result()
            #logger.error(result)

    logger.success("🎉 All Parallel Tasks Finished!")


def main():
    parser = argparse.ArgumentParser(description="Extract the Archive from Downloaded SCARED Zip")
    parser.add_argument(
        "--scared_root", 
        default='/home/wsco/local/yyz/data/scared', 
        help="Path to the dataset root folder. Default is './data'."
    )
    args = parser.parse_args()
    parallel_process_scared_data(Path(args.scared_root), max_workers=16)
    return 0

if __name__ == "__main__":
    main()