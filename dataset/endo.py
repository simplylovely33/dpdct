import os
import re
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from loguru import logger
from torchvision import transforms as T


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

        self._load_split()

        if self.mode == 'train':
            self.samples_per_video = 1
        elif self.mode == 'val':
            self.samples_per_video = 10
        else:
            self.samples_per_video = 1

    def _load_split(self):
        if os.path.exists(self.split_file):
            with open(self.split_file, "r") as f:
                for line in f:
                    item = {"video_path": os.path.join(self.data_root, line.strip())}
                    item["center_id"] = None
                    self._samples.append(item)
            logger.info(f"Loaded {len(self._samples)} sequences from {self.split_file}")
        else:
            logger.error(f"Split file not found: {self.split_file}")

    def _extract_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    
    def _lazy_init_video_meta(self, video_path):
        if video_path in self._video_frame_lists:
            return
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
        
        # 0. 基础保护：如果视频比窗口还短，降级处理
        if total_frames < self.context_len:
            fallback_id = valid_ids[0] if total_frames > 0 else 0
            return [fallback_id] * self.context_len, fallback_id

        # --- 自适应步长逻辑 ---
        if self.dynamic_sampling and specified_center_id is None:
            # 1. 计算理论最大物理步长
            # 意思是：如果我要从头取到尾，每一步最大能跨多少？
            # 公式: (总帧数 - 1) / (窗口长度 - 1)
            # 例如: 200帧, context=7 -> (199 / 6) = 33
            physical_max_stride = (total_frames - 1) // (self.context_len - 1)
            
            # 2. 定义 SLAM 约束边界 (关键参数)
            # MIN_STRIDE: 保证最小视差 (比如 5帧 ≈ 0.15秒)
            # MAX_STRIDE: 保证几何重叠 (比如 30帧 ≈ 1.0秒)
            # 内窥镜移动慢，30-45 通常是几何重叠的极限，再大就完全没交集了
            SLAM_MIN_STRIDE = 5
            SLAM_MAX_STRIDE = 15 
            
            # 3. 确定最终的随机范围上限
            # 取 "物理极限" 和 "几何极限" 的较小值
            # 如果视频很短(30帧)，physical=4，则上限被限制在 4
            # 如果视频很长(2000帧)，physical=300，则上限被限制在 30
            upper_bound = min(physical_max_stride, SLAM_MAX_STRIDE)
            
            # 4. 随机选择步长
            if upper_bound < SLAM_MIN_STRIDE:
                # 视频太短，无法满足最小视差要求，只能尽可能大或者降级为1
                stride = max(1, upper_bound)
            else:
                # 在 [最小视差, 几何极限] 之间随机，实现数据多样性
                stride = random.randint(SLAM_MIN_STRIDE, upper_bound)
            
            # 5. 确定中心点范围 (基于选定的 stride)
            half = self.context_len // 2
            margin = half * stride
            
            min_idx = margin
            max_idx = total_frames - 1 - margin
            
            # 再次检查边界 (防止极端情况)
            if min_idx > max_idx:
                center_idx = total_frames // 2
                stride = 1 # 回退
            else:
                center_idx = random.randint(min_idx, max_idx)
            
            # 6. 生成窗口 ID
            window_ids = []
            for offset in range(-half, half + 1):
                idx = center_idx + (offset * stride)
                window_ids.append(valid_ids[idx])
            
            return window_ids, valid_ids[center_idx]

        else:
            # --- 验证/测试模式 (保持确定性) ---
            center_id = specified_center_id
            if center_id is None:
                center_id = valid_ids[total_frames // 2]
            
            try:
                curr_idx = valid_ids.index(center_id)
            except ValueError:
                curr_idx = total_frames // 2
                center_id = valid_ids[curr_idx]
            
            # 验证时使用固定的中等步长 (例如 10)，确保公平对比
            stride = 10 
            half = self.context_len // 2
            window_ids = []
            
            for offset in range(-half, half + 1):
                idx = max(0, min(curr_idx + (offset * stride), total_frames - 1))
                window_ids.append(valid_ids[idx])
                
            return window_ids, center_id

    def _read_rgb_and_size(self, video_path, frame_id):
            candidates = [f"{frame_id}_color.png", f"{frame_id:04d}_color.png"]
            
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
            if d is None: 
                return torch.zeros(1, self.target_h, self.target_w)
            # c3vd data format
            if d.dtype == np.uint16:
                d = d.astype(np.float32) * (0.1 / 65535.0)
            elif d.dtype == np.uint8:
                d = d.astype(np.float32) / 255.0 * 10.0
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
            except:
                pass
        
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
        """
        基于纯几何投影和深度一致性校验生成 Sparse Depth。
        不包含任何光度（RGB）计算。
        """
        seq_len = len(depth_list)
        center_idx = seq_len // 2
        H, W = depth_list[0].shape[-2:]
        device = depth_list[0].device

        # --- 1. 准备中心帧数据 (Source) ---
        center_depth = depth_list[center_idx].squeeze(0) # [H, W]
        center_pose = pose_list[center_idx]              # [4, 4] C2W

        valid_mask = center_depth > 0
        valid_coords = torch.nonzero(valid_mask) # [N_valid, 2] (y, x)

        # 随机采样 n_points 个点
        if valid_coords.shape[0] < n_points:
            selected_coords = valid_coords
        else:
            # 随机打乱取前 n 个
            choice = torch.randperm(valid_coords.shape[0])[:n_points]
            selected_coords = valid_coords[choice]
        
        # 获取源像素坐标和深度
        # 注意: selected_coords 是 (y, x)
        ys_source = selected_coords[:, 0] 
        xs_source = selected_coords[:, 1]
        zs_source = center_depth[ys_source, xs_source] # [N]

        # --- 2. 反投影: 2D Pixel -> 3D World ---
        # 构造齐次像素坐标 [3, N]
        ones = torch.ones_like(zs_source)
        # 像素坐标列向量: [u, v, 1]^T
        pixel_coords_col = torch.stack([xs_source, ys_source, ones], dim=0).float() 

        # K_inv @ pixel * depth -> Camera Coordinate
        K_inv = torch.inverse(k)
        # [3, N] = [3, 3] @ [3, N] * [1, N]
        pts_cam_source = (K_inv @ pixel_coords_col) * zs_source.unsqueeze(0)

        # Camera -> World
        # [4, N] (Homogeneous)
        pts_cam_source_homo = torch.cat([pts_cam_source, torch.ones(1, pts_cam_source.shape[1], device=device)], dim=0)
        # P_world = T_c2w @ P_cam
        pts_world = center_pose @ pts_cam_source_homo # [4, N]

        # --- Debug 准备 ---
        if rgb_list is not None:
            # 给每个点分配固定的随机颜色，用于追踪视觉一致性
            debug_colors = np.random.randint(0, 255, (pts_world.shape[1], 3), dtype=np.uint8)
            debug_vis_imgs = []

        # --- 3. 重投影循环: 3D World -> All Frames ---
        sparse_depth_list = []

        for i in range(seq_len):
            # 初始化空稀疏图
            sparse_map = torch.zeros_like(depth_list[i]) # [1, H, W]
            
            # 获取当前帧的 World-to-Camera 矩阵
            # T_w2c = inv(T_c2w)
            w2c = torch.inverse(pose_list[i]) # [4, 4]

            # A. 变换到当前相机坐标系
            # P_curr = T_w2c @ P_world
            pts_cam_curr_homo = w2c @ pts_world # [4, N]
            pts_cam_curr = pts_cam_curr_homo[:3, :] # [3, N]

            # B. 投影到像素坐标
            # P_pix = K @ P_curr
            pts_pix_homo = k @ pts_cam_curr # [3, N]
            
            # 透视除法
            # z_proj 是投影计算出的深度
            z_proj = pts_cam_curr[2, :] # [N]
            eps = 1e-6
            u_proj = pts_pix_homo[0, :] / (z_proj + eps)
            v_proj = pts_pix_homo[1, :] / (z_proj + eps)

            # --- 4. 几何一致性校验 (The Filter) ---
            
            # Condition 1: 手性检查 (必须在相机前方)
            mask_z = z_proj > 0.01

            # Condition 2: 图像边界检查
            u_round = torch.round(u_proj).long()
            v_round = torch.round(v_proj).long()
            mask_bound = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

            # 组合基础 Mask
            valid_idx = torch.nonzero(mask_z & mask_bound).squeeze()
            if valid_idx.dim() == 0 and valid_idx.numel() == 1: valid_idx = valid_idx.unsqueeze(0)
            elif valid_idx.numel() == 0: valid_idx = torch.tensor([], device=device, dtype=torch.long)

            # Condition 3: 深度一致性 (几何遮挡检测)
            # 只有当 投影深度 ≈ 真实深度 时，才认为该点可见
            final_indices = []
            
            if len(valid_idx) > 0:
                # 提取通过边界检查的点的坐标
                u_check = u_round[valid_idx]
                v_check = v_round[valid_idx]
                z_check_proj = z_proj[valid_idx] # 投影过来的理论深度

                # 读取当前帧该位置的 Ground Truth 深度
                z_gt = depth_list[i][0, v_check, u_check]

                # --- 核心判断 ---
                # 允许误差: 相对误差 < 10% (可以根据数据质量调整，比如 0.05 或 0.15)
                # 绝对误差 < 0.05 (防止极小深度时的除法不稳定)
                # 如果 z_gt 是 0 (无效区域)，也认为不一致
                
                # 相对误差公式
                rel_error = torch.abs(z_gt - z_check_proj) / (z_gt + 1e-6)
                
                # 你可以根据需要放宽或收紧这个阈值
                is_consistent = (rel_error < 0.1) & (z_gt > 0)
                
                # 筛选出最终一致的点的原始索引
                consistent_indices = valid_idx[is_consistent]
                
                if len(consistent_indices) > 0:
                    # 填入 Sparse Map
                    # 使用 z_check_proj (投影深度) 也就是输入时的几何约束深度
                    # 这保证了输入的 sparse depth 和输入的 pose 是严格数学对应的
                    final_u = u_round[consistent_indices]
                    final_v = v_round[consistent_indices]
                    final_z = z_proj[consistent_indices]
                    final_inverse_z = 1.0 / (final_z + 1e-6)
                    sparse_map[0, final_v, final_u] = final_inverse_z
                    final_indices = consistent_indices

            sparse_depth_list.append(sparse_map)

            # --- Debug 可视化 ---
            if rgb_list is not None:
                img_vis = rgb_list[i].permute(1, 2, 0).cpu().numpy() * 255.0
                img_vis = img_vis.astype(np.uint8).copy()
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

                # 绘制所有 Consistent 的点
                # 注意：我们必须使用 consistent_indices 来索引 debug_colors
                # 这样同一个 3D 点在不同帧才会显示同一种颜色
                if len(final_indices) > 0:
                    for idx in final_indices:
                        idx = idx.item() # 原始采样点的序号
                        cx, cy = int(u_round[idx]), int(v_round[idx])
                        color = debug_colors[idx].tolist()
                        cv2.circle(img_vis, (cx, cy), 3, color, -1)
                
                cv2.putText(img_vis, f"Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                debug_vis_imgs.append(img_vis)

        # --- 保存 Debug 图片 ---
        if rgb_list is not None and len(debug_vis_imgs) > 0:
            concat_img = np.hstack(debug_vis_imgs)
            save_dir = "./debug"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"sample_{debug_idx}.jpg")
            cv2.imwrite(save_path, concat_img)

        return sparse_depth_list

    def __getitem__(self, index):
        video_index = index % len(self._samples)
        sample = self._samples[video_index] 
        video_path = sample["video_path"]
        
        self._lazy_init_video_meta(video_path)
 
        max_retries = 3
        for attempt in range(max_retries):
            window_ids, center_id = self._get_sampling_window(video_path, sample['center_id'])

            rgb_list, sparse_list, depth_list, pose_list = [], [], [], []

            for fid in window_ids:
                img, h, w = self._read_rgb_and_size(video_path, fid)
                rgb_list.append(img)

                depth = self._read_depth_and_size(video_path, fid, "depth")
                depth_list.append(depth)

                if fid in self._pose_cache[video_path]:
                    pose_list.append(self._pose_cache[video_path][fid])
                else:
                    pose_list.append(torch.eye(4).float())
            
            k = self._read_intrinsics(video_path, h, w)
            ks = k.unsqueeze(0).repeat(self.context_len, 1, 1)

            if self.mode == "train":
                n_points = random.randint(self.sparse_range[0], self.sparse_range[1])
            else:
                n_points = 300

            sparse_list_check = self._generate_consistent_sparse(
                depth_list,
                pose_list,
                k,
                n_points,
                debug_idx=index,
                rgb_list=None,
            )

            center_idx = self.context_len // 2
            valid_points_count = (sparse_list_check[center_idx] > 0).sum().item()
            
            # 阈值：如果少于 50 个点，说明几何投影崩了
            if valid_points_count > 50:
                sparse_list = sparse_list_check
                break # 成功，跳出重试
            else:
                # 失败，logger 记录一下 (可选)
                # logger.warning(f"Low sparse points ({valid_points_count}), retrying...")
                
                # 【关键】这里需要一种方式让下一次 _get_sampling_window 变乖
                # 但由于 _get_sampling_window 是随机的，我们其实只要重试大概率会随到小的 stride
                # 或者你可以手动强制下一次 stride = 1 (看你的实现复杂度)
                if attempt == max_retries - 1:
                    # 最后一次还是失败，就勉强用吧，或者返回 stride=1 的结果
                    sparse_list = sparse_list_check


        colors, poses = torch.stack(rgb_list), torch.stack(pose_list)
        sparses, depths = torch.stack(sparse_list), torch.stack(depth_list)
        
        center_idx = self.context_len // 2
        center_pose_inv = torch.inverse(poses[center_idx])
        rel_poses = torch.matmul(center_pose_inv, poses)

        return {
            "color": colors,           
            "sparse": sparses,
            "depth": depths,
            "pose": rel_poses,      
            "intrinsic": ks,        
            "video_path": video_path,
            "center_id": center_id
        }

    def __len__(self):
        return len(self._samples) * self.samples_per_video