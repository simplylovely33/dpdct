import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed

# 还需要完善一下检查机制，就是重复运行的时候，哪些已经解压完或者已经得到结果的rgb图像就不再处理了

def worker_extract_rgb_from_video(dir_root: Path):
    try:
        output_dir = dir_root.parent / "rgb"
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(dir_root))
        if not cap.isOpened():
            return f"❌ Open Video Failure: {dir_root.name}"

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mid_point = height // 2
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            left_img = frame[0:mid_point, :]
            save_name = f"{frame_idx:06d}.png"
            save_path = output_dir / save_name
            cv2.imwrite(str(save_path), left_img)
            
            frame_idx += 1
            
        cap.release()
        return f"✅ Video Processed: {dir_root.name} ({frame_idx} frames)"
    except Exception as e:
        return f"❌ Video Failure: {dir_root.name} -> {e}"

    
def parallel_process_scared_data(dir_root: Path, max_workers=8):
    if not dir_root.exists():
        logger.error(f"SCARED data root doesn't exist: {dir_root}")
        return

    video_tasks = list(dir_root.rglob("rgb.mp4"))
    logger.info(f"🚀 Start Processing {len(video_tasks)} videos with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_extract_rgb_from_video, f) for f in video_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Frames"):
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
    parallel_process_scared_data(Path(args.scared_root), max_workers=16)
    return 0

if __name__ == "__main__":
    main()