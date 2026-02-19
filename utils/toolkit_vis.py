import cv2
import ast
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils.toolkit import *
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def extract_keyframs_imgs(map_keyframes, input_folder, output_folder):
    """
    extract the keyframes images from the map_keyframes, saved in the output_folder
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_folder) ])
    
    print(f"[Info] Find {len(image_files)} image files")
    print(f"[Info] SLAM result has {len(map_keyframes)} keyframes")

    pbar = tqdm(total=len(map_keyframes))
    for kf in map_keyframes:
        index = kf['id']
        _, ext = os.path.splitext(image_files[index])
        src_path = os.path.join(input_folder, image_files[index])
        dst_path = os.path.join(output_folder, f'{index:06d}{ext}')
        shutil.copy(src_path, dst_path)
        pbar.set_description(f"[Info] Copying ")
        pbar.update(1)
    pbar.close()

    print(f"[Info] Copied keyframe images are already saved at: {output_folder}")

def convert_img_to_video(input_folder, output_video, fps=15):
    """
    Convert a sequence of PNG images in a folder to a video file
    
    Args:
        input_folder (str): Path to the folder containing PNG files
        output_video (str): Output video file path
        fps (int): Frames per second for the output video, default is 30
    """
    files = sorted([f for f in os.listdir(input_folder)])
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    if not files:
        print(f"Can not find any files in {input_folder}")
        return
    
    frame = cv2.imread(os.path.join(input_folder, files[0]))

    height, width, _ = frame.shape
    print(f"[Info] Image size: {width}x{height}")
    print(f"[Info] Find {len(files)} files")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    pbar = tqdm(total=len(files))
    for file in files:
        frame = cv2.imread(os.path.join(input_folder, file))
        video.write(frame)
        pbar.set_description(f"[Info] Working on {os.path.basename(file)}")
        pbar.update(1)
    pbar.close()
    video.release()

    cv2.destroyAllWindows()
    print(f"[Info] Video is already saved at: {output_video}")


def visualize_slam_map(map_keyframes, map_points):
    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract and plot map points with RGB colors
    points = np.array([p['pt'] for p in map_points])  # 3D coordinates [x, y, z]
    colors = np.array([p['color'] for p in map_points]) / 255.0  # Normalize RGB to [0, 1]

    # Plot point cloud
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors[:, [2, 1, 0]], s=1, alpha=0.5
    )

    # Extract and plot camera poses
    camera_poses = [np.array(ast.literal_eval(kf['pose'])) for kf in map_keyframes]
    camera_rotations, camera_positions = get_poses_component(camera_poses, Tcw=True)

    # Plot camera trajectory
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
            color='blue', linewidth=2, marker='o', markersize=5, label='Camera Trajectory')

    # Optional: Plot camera orientation (simplified as arrows)
    # for kf in map_keyframes:
    #     pos = kf['position']
    #     # Assuming rotation is a 3x3 matrix in 'rotation' key
    #     rot = np.array(kf['rotation'])
    #     # Define a small vector for visualization (e.g., z-axis of camera)
    #     length = 0.1  # Length of the arrow
    #     z_axis = rot @ np.array([0, 0, length])  # Transform z-axis
    #     ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], 
    #               color='red', linewidth=1)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SLAM Map Visualization')
    ax.legend()

    # Equalize axis scales for better visualization
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Show plot
    plt.show()


def generate_times_file(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频帧率: {fps}")
    print(f"总帧数: {frame_count}")

    with open(output_txt_path, 'w') as f:
        for i in range(frame_count):
            timestamp = i / fps
            formatted_time = "{:.6e}".format(timestamp)
            
            f.write(formatted_time + '\n')

    cap.release()
    print(f"生成完毕! 文件已保存至: {output_txt_path}")
