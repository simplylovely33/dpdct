import os
import re
import cv2
import ast
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from loguru import logger
from torchvision import transforms as T

import utils.toolkit as utils

class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.data_root = self.cfg['DATA']['ROOT']
        self.context_len = cfg['DATA']['CONTEXT_SIZE']
        self.sparse_range = self.cfg['DATA']['SPARSE_RANGE']
        self.dynamic_sampling = cfg['DATA']['DYNAMIC_SAMPLING']
        self.target_h, self.target_w = self.cfg['DATA']['HEIGHT'], self.cfg['DATA']['WIDTH']
        self.split_file = os.path.join(self.cfg['SPLIT']['OUTPUT'], f"{self.mode}_split.txt")

        self._samples = []
        self._pose_cache, self._video_frame_lists = {}, {}
        
        # [Test Mode] 缓存 SLAM 解析结果
        self._slam_cache = {}

        self._load_split()

        if self.mode == 'train':
            self.samples_per_video = 50
        elif self.mode == 'val':
            self.samples_per_video = 10
        else:
            self.samples_per_video = 1

    def _load_split(self):
        if os.path.exists(self.split_file):
            temp_samples = []
            with open(self.split_file, "r") as f:
                for line in f:
                    item = {"video_path": os.path.join(self.data_root, line.strip())}
                    item["center_id"] = None
                    temp_samples.append(item)
            
            # 对于 Test 模式，确保每一个提取出的关键帧都被作为中心帧评估
            if self.mode == 'test':
                for item in temp_samples:
                    video_path = item["video_path"]
                    self._lazy_init_video_meta(video_path)
                    
                    if video_path in self._video_frame_lists:
                        for fid in self._video_frame_lists[video_path]:
                            self._samples.append({
                                "video_path": video_path,
                                "center_id": fid
                            })
            else:
                self._samples = temp_samples
                
            logger.info(f"Loaded {len(self._samples)} sequences/keyframes from {self.split_file}")
        else:
            logger.error(f"Split file not found: {self.split_file}")

    def _extract_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    
    def _lazy_init_video_meta(self, video_path):
        if video_path in self._video_frame_lists:
            return

        # --- Test 模式：直接解析 map.json 提取关键帧和投影 ---
        if self.mode == 'test':
            map_setup_json = os.path.join(video_path, 'slam_result', 'slam_state', 'map.json')
            
            if not os.path.exists(map_setup_json):
                logger.error(f"Map file not found: {map_setup_json}")
                self._video_frame_lists[video_path] = []
                return
            
            logger.info(f"Parsing SLAM Map for {video_path}...")
            map_setup = utils.read_json_file(map_setup_json)
            map_data = map_setup['map']
            
            kf_keys = self.cfg.get('PYSLAM', {}).get('KEYFRAMS_KEYS', ['id', 'camera', 'pose', 'image_name'])
            pt_keys = self.cfg.get('PYSLAM', {}).get('POINTS_KEYS', ['id', 'pt', 'color'])
            
            map_keyframes = utils.get_specific_keys(map_data['keyframes'], kf_keys)
            map_points = utils.get_specific_keys(map_data['points'], pt_keys)
            
            # 使用 toolkit 预计算所有的投影深度
            projections = utils.project_all_keyframes(map_keyframes, map_points, debug=False)
            
            self._video_frame_lists[video_path] = projections['keyframes_id']
            self._slam_cache[video_path] = {
                'projections': projections,
                'map_keyframes': {kf['id']: kf for kf in map_keyframes}
            }
            return

        # --- Train/Val 模式：原始读取逻辑 ---
        rgb_dir = os.path.join(video_path, "color")
        if not os.path.exists(rgb_dir):
            logger.warning(f"Warning: RGB dir not found {rgb_dir}")
            self._video_frame_lists[video_path] = []
            return

        files = [f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        sorted_ids = sorted(list(set([self._extract_number(f) for f in files])))
        sorted_ids = [i for i in sorted_ids if i != -1]
        
        self._video_frame_lists[video_path] = sorted_ids

        pose_map = {}
        pose_file = os.path.join(video_path, self.cfg['SPLIT']['POSE_FILE'])
        if os.path.exists(pose_file):
            try:
                with open(pose_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                count = min(len(lines), len(sorted_ids))
                for i in range(count):
                    fid = sorted_ids[i]
                    line = lines[i]
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 16:
                        vals = [float(x) for x in parts[:16]]
                        mat = torch.tensor(vals).float().view(4, 4)
                        if mat[3, 3] == 1 and mat[3, 0] != 0: mat = mat.t()
                        pose_map[fid] = mat
            except Exception as e:
                logger.error(f"Error reading pose {pose_file}: {e}")
        self._pose_cache[video_path] = pose_map

    def _get_sampling_window(self, video_path, specified_center_id):
        valid_ids = self._video_frame_lists[video_path]
        total_frames = len(valid_ids)
        
        if total_frames < self.context_len:
            fallback_id = valid_ids[0] if total_frames > 0 else 0
            return [fallback_id] * self.context_len, fallback_id

        # --- Test 模式: 取连续关键帧组成窗口 ---
        if self.mode == 'test':
            center_id = specified_center_id
            if center_id is None: center_id = valid_ids[total_frames // 2]
            
            try: curr_idx = valid_ids.index(center_id)
            except ValueError: curr_idx, center_id = total_frames // 2, valid_ids[total_frames // 2]
                
            stride = 1 
            half = self.context_len // 2
            window_ids = []
            
            for offset in range(-half, half + 1):
                idx = max(0, min(curr_idx + (offset * stride), total_frames - 1))
                window_ids.append(valid_ids[idx])
                
            return window_ids, center_id

        # --- Train/Val 模式 ---
        if self.dynamic_sampling and specified_center_id is None:
            physical_max_stride = (total_frames - 1) // (self.context_len - 1)
            SLAM_MIN_STRIDE = 5
            SLAM_MAX_STRIDE = 15 
            
            upper_bound = min(physical_max_stride, SLAM_MAX_STRIDE)
            
            if upper_bound < SLAM_MIN_STRIDE: stride = max(1, upper_bound)
            else: stride = random.randint(SLAM_MIN_STRIDE, upper_bound)
            
            half = self.context_len // 2
            margin = half * stride
            min_idx, max_idx = margin, total_frames - 1 - margin
            
            if min_idx > max_idx:
                center_idx, stride = total_frames // 2, 1 
            else:
                center_idx = random.randint(min_idx, max_idx)
            
            window_ids = []
            for offset in range(-half, half + 1):
                idx = center_idx + (offset * stride)
                window_ids.append(valid_ids[idx])
            
            return window_ids, valid_ids[center_idx]
        else:
            center_id = specified_center_id
            if center_id is None: center_id = valid_ids[total_frames // 2]
            try: curr_idx = valid_ids.index(center_id)
            except: curr_idx, center_id = total_frames // 2, valid_ids[total_frames // 2]
            
            stride = 10 
            half = self.context_len // 2
            window_ids = []
            for offset in range(-half, half + 1):
                idx = max(0, min(curr_idx + (offset * stride), total_frames - 1))
                window_ids.append(valid_ids[idx])
            return window_ids, center_id

    def _read_rgb_and_size(self, video_path, frame_id):
            candidates = [f"{frame_id}_color.png", f"{frame_id:04d}_color.png"]
            
            if self.mode == 'test':
                if video_path in self._slam_cache:
                    kf = self._slam_cache[video_path]['map_keyframes'].get(frame_id)
                    if kf and 'image_name' in kf: candidates.insert(0, kf['image_name'])
                        
            img_path = None
            for c in candidates:
                p = os.path.join(video_path, "color", c)
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size 
                img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                return tensor, h, w
            else:
                return torch.zeros(3, self.target_h, self.target_w), 448, 448

    def _read_depth_and_size(self, video_path, frame_id, folder_name="depth"):
        candidates = [f"{frame_id}_depth.tiff",f"{frame_id:04d}_depth.tiff"]
        d_path = None
        for c in candidates:
            p = os.path.join(video_path, folder_name, c)
            if os.path.exists(p):
                d_path = p
                break
        if d_path:
            d = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
            if d is None: return torch.zeros(1, self.target_h, self.target_w)
            if d.dtype == np.uint16: d = d.astype(np.float32) * (0.1 / 65535.0)
            elif d.dtype == np.uint8: d = d.astype(np.float32) / 255.0 * 10.0
            d = cv2.resize(d, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(d).unsqueeze(0).float()
        return torch.zeros(1, self.target_h, self.target_w)

    def _read_intrinsics(self, video_path, original_h, original_w):
        calib_path = os.path.join(os.path.dirname(video_path), "calib.txt")
        K = torch.eye(3).float()
        
        loaded = False
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    content = f.read().replace(',', ' ').split()
                    vals = [float(x) for x in content if x]
                    if len(vals) == 4:
                        K[0,0], K[1,1] = vals[0], vals[1]
                        K[0,2], K[1,2] = vals[2], vals[3]
                        loaded = True
                    elif len(vals) >= 9:
                        K = torch.tensor(vals[:9]).view(3, 3).float()
                        loaded = True
            except: pass
        
        if not loaded:
            K[0, 0] = original_w * 1.0 
            K[1, 1] = original_w * 1.0
            K[0, 2] = original_w / 2.0
            K[1, 2] = original_h / 2.0

        scale_x = self.target_w / original_w
        scale_y = self.target_h / original_h
        
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        return K

    def _generate_consistent_sparse(
        self, depth_list, pose_list, k, n_points, debug_idx=0, rgb_list=None
    ):
        seq_len = len(depth_list)
        center_idx = seq_len // 2
        H, W = depth_list[0].shape[-2:]
        device = depth_list[0].device

        center_depth = depth_list[center_idx].squeeze(0)
        center_pose = pose_list[center_idx]             

        valid_mask = center_depth > 0
        valid_coords = torch.nonzero(valid_mask) 

        if valid_coords.shape[0] < n_points:
            selected_coords = valid_coords
        else:
            choice = torch.randperm(valid_coords.shape[0])[:n_points]
            selected_coords = valid_coords[choice]
        
        ys_source = selected_coords[:, 0] 
        xs_source = selected_coords[:, 1]
        zs_source = center_depth[ys_source, xs_source] 

        ones = torch.ones_like(zs_source)
        pixel_coords_col = torch.stack([xs_source, ys_source, ones], dim=0).float() 
        K_inv = torch.inverse(k)
        pts_cam_source = (K_inv @ pixel_coords_col) * zs_source.unsqueeze(0)

        pts_cam_source_homo = torch.cat([pts_cam_source, torch.ones(1, pts_cam_source.shape[1], device=device)], dim=0)
        pts_world = center_pose @ pts_cam_source_homo 

        if rgb_list is not None:
            debug_colors = np.random.randint(0, 255, (pts_world.shape[1], 3), dtype=np.uint8)
            debug_vis_imgs = []

        sparse_depth_list = []

        for i in range(seq_len):
            sparse_map = torch.zeros_like(depth_list[i])
            w2c = torch.inverse(pose_list[i]) 
            pts_cam_curr_homo = w2c @ pts_world 
            pts_cam_curr = pts_cam_curr_homo[:3, :] 
            pts_pix_homo = k @ pts_cam_curr 
            
            z_proj = pts_cam_curr[2, :] 
            eps = 1e-6
            u_proj = pts_pix_homo[0, :] / (z_proj + eps)
            v_proj = pts_pix_homo[1, :] / (z_proj + eps)

            mask_z = z_proj > 0.01
            u_round = torch.round(u_proj).long()
            v_round = torch.round(v_proj).long()
            mask_bound = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

            valid_idx = torch.nonzero(mask_z & mask_bound).squeeze()
            if valid_idx.dim() == 0 and valid_idx.numel() == 1: valid_idx = valid_idx.unsqueeze(0)
            elif valid_idx.numel() == 0: valid_idx = torch.tensor([], device=device, dtype=torch.long)

            final_indices = []
            
            if len(valid_idx) > 0:
                u_check = u_round[valid_idx]
                v_check = v_round[valid_idx]
                z_check_proj = z_proj[valid_idx] 

                z_gt = depth_list[i][0, v_check, u_check]

                rel_error = torch.abs(z_gt - z_check_proj) / (z_gt + 1e-6)
                is_consistent = (rel_error < 0.1) & (z_gt > 0)
                consistent_indices = valid_idx[is_consistent]
                
                if len(consistent_indices) > 0:
                    final_u = u_round[consistent_indices]
                    final_v = v_round[consistent_indices]
                    final_z = z_proj[consistent_indices]
                    final_inverse_z = 1.0 / (final_z + 1e-6) # 注意：这里如果原来用反深度，需要统一。如果是正深度应改为 final_z
                    sparse_map[0, final_v, final_u] = final_z
                    final_indices = consistent_indices

            sparse_depth_list.append(sparse_map)

        return sparse_depth_list

    def __getitem__(self, index):
        video_index = index % len(self._samples)
        sample = self._samples[video_index] 
        video_path = sample["video_path"]
        
        self._lazy_init_video_meta(video_path)
        window_ids, center_id = self._get_sampling_window(video_path, sample['center_id'])

        rgb_list, sparse_list, depth_list, pose_list, intrinsic_list = [], [], [], [], []

        _, h_orig, w_orig = self._read_rgb_and_size(video_path, window_ids[0])

        for fid in window_ids:
            # 1. RGB 
            img, _, _ = self._read_rgb_and_size(video_path, fid)
            rgb_list.append(img)

            # 2. GT Depth (供评估和训练时提取Sparse使用)
            depth = self._read_depth_and_size(video_path, fid, "depth")
            depth_list.append(depth)

            # --- Test 模式下的特征提取 ---
            if self.mode == 'test':
                kf = self._slam_cache[video_path]['map_keyframes'][fid]
                
                idx_in_proj = self._slam_cache[video_path]['projections']['keyframes_id'].index(fid)
                orig_depth_map = self._slam_cache[video_path]['projections']['depth_map'][idx_in_proj]
                
                if (orig_depth_map.shape[0] != self.target_h) or (orig_depth_map.shape[1] != self.target_w):
                    sparse_resized = cv2.resize(orig_depth_map, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
                else: sparse_resized = orig_depth_map
                sparse_list.append(torch.from_numpy(sparse_resized).unsqueeze(0).float())
                
                camera_pose = np.array(ast.literal_eval(kf['pose']))
                Twc = np.linalg.inv(camera_pose)
                pose_list.append(torch.from_numpy(Twc).float())
                
                camera = kf['camera']
                K = torch.eye(3).float()
                scale_x = self.target_w / camera['width']
                scale_y = self.target_h / camera['height']
                K[0, 0] = camera['fx'] * scale_x
                K[1, 1] = camera['fy'] * scale_y
                K[0, 2] = camera['cx'] * scale_x
                K[1, 2] = camera['cy'] * scale_y
                intrinsic_list.append(K)
                
            else:
                if fid in self._pose_cache[video_path]:
                    pose_list.append(self._pose_cache[video_path][fid])
                else: pose_list.append(torch.eye(4).float())

        # 如果不是 test 模式，进行几何投影生成一致性 Sparse Depth
        if self.mode != 'test':
            k = self._read_intrinsics(video_path, h_orig, w_orig)
            intrinsic_list = [k] * self.context_len
            
            n_points = random.randint(self.sparse_range[0], self.sparse_range[1]) if self.mode == "train" else 300
            sparse_list = self._generate_consistent_sparse(
                depth_list, pose_list, k, n_points, debug_idx=index, rgb_list=None
            )

        colors, poses = torch.stack(rgb_list), torch.stack(pose_list)
        sparses, depths = torch.stack(sparse_list), torch.stack(depth_list)
        intrinsics = torch.stack(intrinsic_list)
        
        # 相对 Pose 计算
        center_idx = self.context_len // 2
        center_pose_inv = torch.inverse(poses[center_idx])
        rel_poses = torch.matmul(center_pose_inv, poses)

        # ==============================================================
        # 核心改动：动态 Min-Max 归一化 (Dynamic Relative Scale Transformation)
        # ==============================================================
        
        # 1. 对整个窗口内的 Sparse Depth 进行统一的 Min-Max 归一化
        valid_sp = sparses > 1e-4
        if valid_sp.sum() > 0:
            min_sp = sparses[valid_sp].min()
            max_sp = sparses[valid_sp].max()
            sparses[valid_sp] = (sparses[valid_sp] - min_sp) / (max_sp - min_sp + 1e-6)

        # 2. 在 Train/Val 时对 GT Depth 进行相同的统一归一化，训练网络预测相对深度
        # 注：在 Test 时保留绝对尺度 GT，交由 eval 脚本统一执行全局最小二乘法进行对齐验证
        if self.mode != 'test':
            valid_gt = depths > 1e-4
            if valid_gt.sum() > 0:
                min_gt = depths[valid_gt].min()
                max_gt = depths[valid_gt].max()
                depths[valid_gt] = (depths[valid_gt] - min_gt) / (max_gt - min_gt + 1e-6)
        # ==============================================================

        return {
            "color": colors,           
            "sparse": sparses,
            "depth": depths,
            "pose": rel_poses,      
            "intrinsic": intrinsics,        
            "video_path": video_path,
            "center_id": center_id
        }

    def __len__(self):
        return len(self._samples) * self.samples_per_video