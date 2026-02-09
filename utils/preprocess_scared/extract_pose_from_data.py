import json
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

def process_poses_for_c3vd_format(data_dir: Path):
    """
    输入: data 文件夹路径 (包含 frame_data 子文件夹)
    输出: 在 data 目录下生成 poses.txt (C3VD 格式) 和 intrinsics.txt
    """
    json_dir = data_dir / "frame_data"
    if not json_dir.exists():
        return f"⚠️ No frame_data in {data_dir.name}"

    # 按文件名排序确保帧顺序正确 (000000.json, 000001.json...)
    json_files = sorted(list(json_dir.glob("*.json")))
    
    pose_lines = []
    
    # 用于保存内参 (只存一次)
    intrinsic_saved = False
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            # 1. 提取 Pose (左目, World -> Camera or Camera -> World)
            # JSON 里的通常是 [[R11, R12, R13, Tx], ...] (Row-Major)
            pose_matrix = np.array(content['camera-pose']) 
            
            # 【关键步骤】转换为 C3VD 格式
            # C3VD 示例特征: 0 在第 4, 8, 12 位，说明它是 Column-Major (即原矩阵的转置)
            # 比如: 第一行 [R11, R21, R31, 0]
            #pose_matrix_transposed = pose_matrix.T 
            
            # 展平为一维数组
            flat_pose = pose_matrix.flatten()
            
            # 格式化为字符串，逗号分隔
            line_str = ",".join(f"{x:.6f}" for x in flat_pose)
            pose_lines.append(line_str)
            
            # 2. 提取内参 (KL)
            if not intrinsic_saved and "camera-calibration" in content:
                calib = content["camera-calibration"]
                if "KL" in calib:
                    KL = np.array(calib["KL"])
                    # 保存为 standard 3x3 matrix format
                    np.savetxt(data_dir / "intrinsics.txt", KL, fmt="%.6f")
                    intrinsic_saved = True
                    
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")
            
    # 保存 poses.txt
    if pose_lines:
        output_file = data_dir / "poses.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(pose_lines))
        return f"✅ Generated poses.txt ({len(pose_lines)} frames) in {data_dir}"
    else:
        return f"⚠️ No valid poses found in {data_dir.parent}"

def batch_process_poses(root_dir: Path):
    root_path = Path(root_dir)
    # 找到所有的 'data' 文件夹 (每个 keyframe 下都有一个 data)
    data_dirs = list(root_path.rglob("data"))
    
    logger.info(f"Found {len(data_dirs)} sequences. Extracting poses...")
    
    for d in data_dirs:
        # 这里不需要多进程，因为 JSON 读取极快，IO 开销很小
        # 如果觉得慢，也可以套用之前的 ProcessPoolExecutor
        result = process_poses_for_c3vd_format(d)
        logger.info(result)

def main():
    parser = argparse.ArgumentParser(description="Extract the Archive from Downloaded SCARED Zip")
    parser.add_argument(
        "--scared_root", 
        #default='/home/wsco/local/yyz/data/scared', 
        help="Path to the dataset root folder. Default is './data'."
    )
    args = parser.parse_args()
    batch_process_poses(args.scared_root)
    return 0


if __name__ == "__main__":
    main()
    