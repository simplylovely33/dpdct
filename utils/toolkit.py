import os
import sys
import yaml
import torch
import random
import numpy as np
from loguru import logger

def load_config(config_path):
    """
    Load .yaml Configuration File
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"WARNING: Loading Configuration file failure - {e}")
            return None
        
def device_confirmation():
    """
    Confirm the device availability information
    """
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        logger.info(f"Using {torch.cuda.get_device_name(device_id)} as default device.")
        device = torch.device('cuda')
    else:
        logger.info(f"Using {torch.device('cpu')} as default device.")
        device = torch.device('cpu')
    return device


def setup_logger():
    """
    Setup the logger configuration
    """
    logger.remove()
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level}</level> | "
        "<level>{message}</level>"
    )

def initialization(config_path):
    """
    Initialize the configuration
    """
    setup_logger()
    cfg = load_config(config_path)
    device = device_confirmation()
    return cfg, device

    
def load_config(config_path):
    """
    Load .yaml Configuration File
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"WARNING: Loading Configuration file failure - {e}")
            return None

def read_json_file(file_path: str):
    """
    read the json file and return the data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"File {file_path} is not a valid JSON format")
        return None


def get_specific_keys(data: list, keys_to_keep: list):
    """
    get the specific keys from the data
    """
    specific_data = []
    for item in data:
        filtered_item = {}
        for k in keys_to_keep:
            if k in item:
                filtered_item[k] = item[k]
        specific_data.append(filtered_item)
    return specific_data

def get_pose_component(camera_pose: np.ndarray, Tcw=False):
    """
    get the rotation and translation components from the camera pose
    """
    if Tcw:
        camera_pose = np.linalg.inv(camera_pose)
    else:
        pass
    R, t = camera_pose[:3, :3], camera_pose[:3, 3]

    return R, t


def get_poses_component(camera_poses, Tcw=False):
    """
    get the rotation and translation components from the camera poses
    """
    rotations, translations = [], []
    for pose in camera_poses:
        if Tcw:
            pose = np.linalg.inv(pose)
        else:
            pass
        R, t = pose[:3, :3], pose[:3, 3]
        rotations.append(R)
        translations.append(t)
    return np.array(rotations), np.array(translations)

def project_mapoints_to_img(map_keyframe, map_points):
    """
    project the map points to the image plane
    """
    camera = map_keyframe['camera']
    fx, fy, cx, cy = camera['fx'], camera['fy'], camera['cx'], camera['cy']
    camera_pose = np.array(ast.literal_eval(map_keyframe['pose']))
    R, t = get_pose_component(camera_pose, Tcw=False)

    projected_points = []
    for point in map_points:
        pt = np.array(point['pt'])
        pt_wc = R @ pt + t
        if pt_wc[2] <= 0:
            continue
        x = pt_wc[0] / pt_wc[2]
        y = pt_wc[1] / pt_wc[2]
        
        u = fx * x + cx
        v = fy * y + cy

        projected_point = {
            'id': point['id'],
            'pt_3d': pt,
            'pt_2d': np.array([u, v]),
            'depth': pt_wc[2],
            'color': point['color']
        }

        projected_points.append(projected_point)

    return projected_points

def create_depth_map(projected_points, map_keyframe, debug=False):
    """
    create the depth map from the projected points
    """
    camera = map_keyframe['camera']
    height, width = camera['height'], camera['width']
    depth_map = np.zeros((height, width), dtype=np.float32)
    for point in projected_points:
        u, v = point['pt_2d']
        u_int, v_int = int(round(u)), int(round(v))
         # Check if the point is within image bounds
        if 0 <= u_int < width and 0 <= v_int < height:
            # Update depth map with the depth value
            # If multiple points project to the same pixel, keep the closest one
            current_depth = depth_map[v_int, u_int]
            if current_depth == 0 or point['depth'] < current_depth:
                depth_map[v_int, u_int] = point['depth']
    if debug:
        current_file_path = os.path.abspath(__file__)
        output_folder = os.path.join(
            os.path.dirname(current_file_path), 'debug', 'slam_depth'
        )
        output_path = os.path.join(output_folder, f'{map_keyframe["id"]:06d}.png')
        if np.max(depth_map) > 0:
            normalized_depth = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
            # Save using OpenCV (uncomment if needed)
            cv2.imwrite(output_path, normalized_depth)
    else:
        pass
    return depth_map

def project_all_keyframes(map_keyframes, map_points, debug=True):
    """
    project all keyframes' map points to the image plane
    """
    keyframes_id, projected_points, depth_maps, = [], [], []
    for keyframe in map_keyframes:
        keyframes_id.append(keyframe['id'])
        projected_point = project_mapoints_to_img(keyframe, map_points)
        projected_points.append(projected_point)
        depth_map = create_depth_map(projected_point, keyframe, debug)
        depth_maps.append(depth_map)
    
    return {'keyframes_id': keyframes_id, 'projected_points': projected_points,'depth_map': depth_maps}


def save_npy(index, save_path):
    """
    save the numpy array to the specified path
    """
    np.save(save_path, index)


class EndoSplitGenerator:
    def __init__(self, cfg, random_seed=42):
        self.cfg = cfg['SPLIT']
        self.split_ratio = self.cfg['RATIO']
        self.output_dir = self.cfg['OUTPUT']
        self.data_root = os.path.dirname(self.output_dir)
        self.random_seed = random_seed
        
        self.folders = {
            "color": self.cfg['RGB_DIR'],
            "depth": self.cfg['DEPTH_DIR'],
            "pose": self.cfg['POSE_FILE']
        }
        
        self.exts = {
            "color": self.cfg['RGB_EXT'],
            "depth": self.cfg['DEPTH_EXT'],
        }

    def _has_files(self, folder_path, exts):
        if not os.path.exists(folder_path):
            return False
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in exts):
                return True
        return False

    def is_valid_sequence(self, seq_abs_path):
        rgb_path = os.path.join(seq_abs_path, self.folders["color"])
        if not self._has_files(rgb_path, self.exts["color"]):
            return False

        depth_path = os.path.join(seq_abs_path, self.folders["depth"])
        if not self._has_files(depth_path, self.exts["depth"]):
            return False

        pose_path = os.path.join(seq_abs_path, self.folders["pose"])
        if not os.path.exists(pose_path):
            return False
        return True

    def find_sequences(self):
        valid_sequences = []
        logger.info(f"Scanning {self.data_root} ...")
        
        for root, dirs, files in os.walk(self.data_root):
            if self.folders["color"] in dirs:
                seq_abs_path = root
                seq_rel_path = os.path.relpath(root, self.data_root)
                
                if self.is_valid_sequence(seq_abs_path):
                    valid_sequences.append(seq_rel_path)
                else:
                    logger.info(f"[Skip] Incomplete sequence: {seq_rel_path}")
                    
        return valid_sequences

    def write_txt(self, filename, seq_list):
        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, "w") as f:
            for seq_path in seq_list:
                f.write(f"{seq_path}\n")
        logger.info(f"Generated {filename}: {len(seq_list)} sequences.")

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        random.seed(self.random_seed)
        
        all_seqs = self.find_sequences()
        total = len(all_seqs)
        logger.info(f"Found {total} valid sequences.")
        
        if total == 0:
            logger.error("Error: No sequences found! Check your data_root and folder names.")
            return

        random.shuffle(all_seqs)
        
        n_val = int(total * self.split_ratio[1])
        n_test = int(total * self.split_ratio[2])
        
        if total >= 3:
            if n_val == 0: n_val = 1
            if n_test == 0: n_test = 1
            
        n_train = total - n_val - n_test
        
        train_seqs = all_seqs[:n_train]
        val_seqs = all_seqs[n_train : n_train + n_val]
        test_seqs = all_seqs[n_train + n_val :]
        
        self.write_txt("train_split.txt", train_seqs)
        self.write_txt("val_split.txt", val_seqs)
        self.write_txt("test_split.txt", test_seqs)

def normalize_flatten_camrea_info(intrinsics, pose, width, height):
    intr_norm = intrinsics.clone()
    intr_norm[..., 0, 0] /= width
    intr_norm[..., 0, 2] /= width
    intr_norm[..., 1, 1] /= height
    intr_norm[..., 1, 2] /= height
    intr_flat, pose_flat = intr_norm.flatten(2), pose[:, :, :3, :].flatten(2)
    return intr_flat, pose_flat

if __name__ == "__main__":
    setup_logger()
    cfg = load_config('configs/config.yaml')
    generator = EndoSplitGenerator(cfg)
    generator.run()