import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from natsort import natsorted   # 推荐使用自然排序，避免 10 在 2 前面


def create_depth_video(
    input_folder,
    output_video_path,
    fps=20,
    colormap=cv2.COLORMAP_JET,
    extensions=(".png", ".tiff", ".tif")
):
    """
    读取文件夹下的 16-bit 深度图（支持 .png / .tiff / .tif），归一化 + 伪彩色，可视化成视频

    Args:
        input_folder (str/Path): 包含深度图的文件夹
        output_video_path (str/Path): 输出视频路径 (.mp4)
        fps (int): 帧率
        colormap: cv2.COLORMAP_xxx
        extensions: 支持的文件后缀（可自行增减）
    """
    input_path = Path(input_folder).expanduser().resolve()
    if not input_path.is_dir():
        logger.error(f"输入路径不是文件夹或不存在: {input_folder}")
        return

    # 收集所有支持的图片文件，并使用自然排序
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))  # 兼容大写后缀

    if not image_files:
        logger.error(f"在 {input_folder} 下未找到任何 {extensions} 文件")
        return

    # 非常重要：使用 natsort 进行文件名自然排序
    image_files = natsorted(image_files)

    logger.info(f"找到 {len(image_files)} 张深度图，准备生成视频...")

    # 读取第一帧，获取尺寸 & 确认能读
    first_frame = cv2.imread(str(image_files[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        logger.error(f"无法读取第一帧: {image_files[0]}")
        return

    if len(first_frame.shape) != 2:
        logger.warning(f"第一帧不是单通道图像，shape={first_frame.shape}，将继续尝试处理单通道切片")

    height, width = first_frame.shape[:2]
    logger.info(f"图像尺寸: {width} × {height}    dtype: {first_frame.dtype}")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # 部分系统更兼容，可替换试试
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error("视频写入器初始化失败，请检查路径/编码器/权限")
        return

    for i, img_path in enumerate(image_files, 1):
        # 关键：使用 IMREAD_UNCHANGED 读取原始位深（支持 16-bit TIFF/PNG）
        depth_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if depth_raw is None:
            logger.warning(f"无法读取 {img_path}，跳过")
            continue

        # 如果是三通道，尝试取第一个通道（常见于某些保存方式）
        if len(depth_raw.shape) == 3:
            if depth_raw.shape[2] in (1, 3, 4):
                depth_raw = depth_raw[..., 0]   # 通常深度图只用第一个通道
            else:
                logger.warning(f"通道数异常 {depth_raw.shape}，尝试继续处理")
                depth_raw = depth_raw[..., 0]

        # 确保是 2D 数组
        if len(depth_raw.shape) != 2:
            logger.warning(f"帧 {i} 不是单通道 2D 图像，跳过")
            continue

        # ------------------ 可视化核心 ------------------
        valid_mask = depth_raw > 0
        if valid_mask.any():
            min_val = depth_raw[valid_mask].min()
            max_val = depth_raw[valid_mask].max()
        else:
            min_val = max_val = 0

        if max_val - min_val < 1e-6:
            depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
        else:
            depth_norm = (depth_raw.astype(np.float32) - min_val) / (max_val - min_val)
            depth_norm = np.clip(depth_norm, 0, 1)

        depth_8bit = (depth_norm * 255).astype(np.uint8)

        # 应用伪彩色
        depth_color = cv2.applyColorMap(depth_8bit, colormap)

        # 写入
        out.write(depth_color)

        if i % 50 == 0:
            logger.info(f"已处理 {i}/{len(image_files)} 帧")

    out.release()
    logger.success(f"视频已保存至: {output_video_path.absolute()}")


if __name__ == "__main__":
    # 示例路径（请自行修改）
    
    #/home/wsco/local/yyz/data/scared/dataset_3/keyframe_1/data/depth_left
    depth_folder = "/home/wsco1/yyz/dataset/zip/edam/rectified01/depth01"
    output_video = "/home/wsco1/yyz/video/edm_r1_depth.mp4"

    create_depth_video(
        depth_folder,
        Path(output_video),
        fps=30,
        colormap=cv2.COLORMAP_JET,
        # 如果你的文件是 .TIFF 大写后缀，也可以直接包含
        extensions=(".png", ".tiff", ".tif", ".PNG", ".TIFF", ".TIF")
    )

