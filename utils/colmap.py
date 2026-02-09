import os
import sys
import subprocess
import argparse
import numpy as np
import shutil
from pathlib import Path
from loguru import logger
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from collections import deque


def check_colmap_installed():
    try:
        subprocess.run(["colmap", "help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.success("COLMAP executable installed")
    except FileNotFoundError:
        logger.error("COLMAP executable not found in PATH! Please install COLMAP.")


def run_cmd_live(cmd, verbose=False):
    logger.info(f"Running: {' '.join(cmd)}")
    log_queue = deque(maxlen=6)
    full_log = [] 

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1 
    )

    with Live(refresh_per_second=10) as live:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                line = line.rstrip()
                log_queue.append(line)
                full_log.append(line)
                log_text = Text("\n".join(log_queue), style="dim green")
                live.update(Panel(log_text, border_style="dim white"))

    if process.returncode != 0:
        logger.error("Command failed!")
        print("\n".join(full_log)) 
        raise subprocess.CalledProcessError(process.returncode, cmd)
    else:
        logger.success(f"Finished: {' '.join(cmd)}")


def read_calibration(calib_file, camera_model):
    try:
        content = calib_file.read_text().replace(',', ' ').strip().split()
        vals = np.array([float(x) for x in content])
        matrix = vals.reshape(3, -1)
        fx, fy, cx, cy = 0, 0, 0, 0

        if len(vals) == 12 or len(vals) == 9:
            fx, fy = matrix[0, 0], matrix[1, 1]
            cx, cy = matrix[0, 2], matrix[1, 2]
        elif len(vals) == 4:
            fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
        else:
            logger.warning(f"Unknown calib format in {calib_file}, found {len(vals)} values. Expected 4 or 9.")
            return None
        
        if camera_model == "PINHOLE":
            params = f"{fx},{fy},{cx},{cy}"
        elif camera_model == "OPENCV":
            params = f"{fx},{fy},{cx},{cy}" + ",0.0,0.0,0.0,0.0"

        logger.info(f"Loaded intrinsics from file: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return params
    
    except Exception as e:
        logger.warning(f"Failed to parse {calib_file}: {e}")
        return None
    

def colmap_pipeline(
        input_root, save_root, mask_root, calib_file=None,
        use_gpu=True, use_sequential=False,
        camera_model="PINHOLE"
    ):

    img_dir = input_root / 'image01'
    data_name = input_root.name
    logger.info(f"Processing video: {data_name} | Images: {img_dir}")
    
    save_path = save_root / data_name
    db_path = save_path / "database.db"
    sparse_dir = save_path /"sparse"

    if db_path.exists():
        os.remove(db_path)
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    use_k = None
    if calib_file is not None and calib_file.exists():
        k = read_calibration(calib_file, camera_model)
    else:
        logger.warning(f"No calibration file provided or file not found. Using default/estimation.")

    use_mask = mask_root is not None and mask_root.exists()

    gpu_flag = "1" if use_gpu else "0"

    # Step 1: Feature Extraction
    extract_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--ImageReader.single_camera", "1", 
        "--ImageReader.camera_model", camera_model,
        "--SiftExtraction.use_gpu", gpu_flag
    ]
    extract_cmd += ["--ImageReader.camera_params", k] if use_k else []
    extract_cmd += ["--ImageReader.mask_path", str(mask_root)] if use_mask else []
    run_cmd_live(extract_cmd)

    # Step 2: Feature Matching (Exhaustive / Sequential)
    if not use_sequential:
        match_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
            "--ExhaustiveMatching.block_size", "50"
        ]
    else:
        match_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
            "--SequentialMatching.overlap", "20", 
            "--SequentialMatching.loop_detection", "0"
        ]
    run_cmd_live(match_cmd)

    # Step 3: Sparse Reconstruction
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001"
    ]

    if use_k and camera_model=="PINHOLE":
        logger.info("Using Hybrid Strategy: Fixing Intrinsics (f, c), Refining Distortion (k, p).")
    elif use_k and camera_model=="OPENCV":
        mapper_cmd.extend([
            "--BundleAdjustment.refine_focal_length", "0",
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", "1" 
        ])
    else:
        logger.info("Using Auto Strategy: Refining ALL parameters.")
    run_cmd_live(mapper_cmd)

    # Step 4: Convert Saved Format (BIN to TXT)
    model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not model_dirs:
        logger.error(f"Mapper failed! No models found in {sparse_dir}")
        return
    logger.info(f"Found {len(model_dirs)} sub-models: {[m.name for m in model_dirs]}")

    best_model = None
    max_points = -1

    for model_path in model_dirs:
        logger.info(f"Converting model {model_path.name}...")
        convert_bin2txt_cmd = [
            "colmap", "model_converter",
            "--input_path", str(model_path),
            "--output_path", str(model_path),
            "--output_type", "TXT"
        ]
        run_cmd_live(convert_bin2txt_cmd)
        
        points_file = model_path / "points3D.txt"
        if points_file.exists():
            score = points_file.stat().st_size 
            logger.info(f"Model {model_path.name} size score: {score/1024:.2f} KB")
            
            if score > max_points:
                max_points = score
                best_model = model_path

    if best_model:
        logger.success(f"Best reconstruction is likely: Model {best_model.name}")
        
        target_0 = sparse_dir / "0"
        for model_path in model_dirs:
            if model_path != best_model:
                shutil.rmtree(model_path)
                logger.info(f"Removed suboptimal model: {model_path.name}")

        if best_model != target_0:
            best_model.rename(target_0)
            logger.success(f"Renamed best model {best_model.name} to 0")
        else:
            logger.success("Best model is already 0, kept as is.")
        
        # Step 5: Convert Sparse Result to PLY
        ply_path = target_0 / "model.ply"
        convert_ply_cmd = [
                "colmap", "model_converter",
                "--input_path", str(target_0),
                "--output_path", str(ply_path), 
                "--output_type", "PLY"
            ]
        run_cmd_live(convert_ply_cmd)
        logger.success(f"Saved point cloud to {ply_path}")
    else:
        logger.warning("Models were found but conversion failed or empty.")


def main():
    parser = argparse.ArgumentParser(description="COLMAP Executable Process Script")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Root directory containing images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Root directory containing structure from motion result"
    )
    parser.add_argument(
        "--calib", 
        type=str, 
        #required=True, 
        help="Whether need to add the calibration information"
    )
    parser.add_argument(
        "--mask", 
        type=str, 
        #required=True, 
        help="Root directory containing masks"
    )
    parser.add_argument(
        "-us", "--use_sequential", 
        action="store_true",
        help="Use exhaustive matcher in COLMAP"
    )
    parser.add_argument(
        "-cm", "--camera_model", 
        default="PINHOLE",
        choices=["PINHOLE", "OPENCV"],
        help="Camera model"
    )
    args = parser.parse_args()

    check_colmap_installed()
    
    root_path, save_path = Path(args.input), Path(args.output)
    mask_path = Path(args.mask) if args.mask else None
    calib_file = Path(args.calib) if args.calib else None
    match_mode = args.use_sequential

    camera_model = args.camera_model
    colmap_pipeline(
        root_path, save_path, mask_path, calib_file, 
        use_sequential=match_mode, camera_model=camera_model
    )

if __name__ == "__main__":
    main()