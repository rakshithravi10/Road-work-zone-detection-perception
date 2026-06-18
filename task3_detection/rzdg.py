import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_points, bbox_camera2lidar
from pointpillars.dataset import point_range_filter, data_augment


class RZDG(Dataset):
    """
    RZDG dataset loader for PointPillars
    Dataset contains: barriers and road_beacons
    """

    CLASSES = {
        'barrier': 0,
        'road_beacon': 1
    }

    def __init__(self, data_root, split='train', pts_prefix='velodyne'):
        """
        Args:
            data_root: root directory containing 'training' or 'testing' folders
            split: 'train', 'val', 'trainval', or 'test'
            pts_prefix: prefix for point cloud files (default 'velodyne')
        """
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        
        # Determine dataset directory
        if split == 'test':
            dataset_dir = os.path.join(data_root, 'testing')
        else:
            dataset_dir = os.path.join(data_root, 'training')
        
        self.dataset_dir = dataset_dir
        
        # Load all sample IDs
        image_dir = os.path.join(dataset_dir, 'image_0')
        all_ids = sorted([f.replace('.png', '') for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Split train/val if needed
        if split == 'trainval':
            self.sample_ids = all_ids
        elif split == 'train':
            # Use 90% for training
            split_idx = int(0.9 * len(all_ids))
            self.sample_ids = all_ids[:split_idx]
        elif split == 'val':
            # Use 10% for validation
            split_idx = int(0.9 * len(all_ids))
            self.sample_ids = all_ids[split_idx:]
        else:  # test
            self.sample_ids = all_ids
        
        # Data augmentation configuration
        self.data_aug_config = dict(
            db_sampler=None,  # No database sampling for RZDG
            object_noise=dict(
                num_try=100,
                translation_std=[0.1, 0.1, 0.1],
                rot_range=[-0.15707963267, 0.15707963267]
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            ),
            point_range_filter=[0, -40, -3, 70, 40, 1],
            object_range_filter=[0, -40, -3, 70, 40, 1]
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        
        # Load point cloud
        pts_path = os.path.join(self.dataset_dir, self.pts_prefix, f'{sample_id}.bin')
        pts = read_points(pts_path)
        
        # Load calibration
        calib_path = os.path.join(self.dataset_dir, 'calib', f'{sample_id}.txt')
        calib_info = self._load_calib(calib_path)
        
        # Load image info
        image_path = os.path.join(self.dataset_dir, 'image_0', f'{sample_id}.png')
        image_info = {'image_path': image_path}
        
        # Load annotations if not test set
        if self.split != 'test':
            label_path = os.path.join(self.dataset_dir, 'label_0', f'{sample_id}.txt')
            annos_info = self._load_annotations(label_path)
            
            # Convert camera coordinates to LiDAR coordinates
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)
            
            gt_bboxes_3d = bbox_camera2lidar(annos_info['gt_bboxes'], tr_velo_to_cam, r0_rect)
            
            data_dict = {
                'pts': pts,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels': annos_info['gt_labels'].astype(np.int64),
                'gt_names': annos_info['gt_names'],
                'difficulty': np.zeros(len(annos_info['gt_names'])),
                'image_info': image_info,
                'calib_info': calib_info
            }
            
            # Apply data augmentation for training
            if self.split in ['train', 'trainval']:
                data_dict = data_augment(self.CLASSES, self.dataset_dir, data_dict, self.data_aug_config)
            else:
                data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
        else:
            # Test set - no annotations
            data_dict = {
                'pts': pts,
                'image_info': image_info,
                'calib_info': calib_info
            }
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
        
        return data_dict

    def _load_calib(self, calib_path):
        """Load calibration file and return calibration info dict"""
        calib_info = {}
        
        with open(calib_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                
                key = parts[0].strip()
                value = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
                
                if key == 'P0':
                    calib_info['P0'] = value.reshape(3, 4)
                elif key == 'Tr_velo_to_cam':
                    calib_info['Tr_velo_to_cam'] = value.reshape(3, 4)
                elif key == 'R0_rect':
                    calib_info['R0_rect'] = value.reshape(3, 3)
        
        # Ensure required keys exist
        if 'Tr_velo_to_cam' not in calib_info:
            calib_info['Tr_velo_to_cam'] = np.eye(3, 4, dtype=np.float32)
        if 'R0_rect' not in calib_info:
            calib_info['R0_rect'] = np.eye(3, dtype=np.float32)
        if 'P0' not in calib_info:
            calib_info['P0'] = np.zeros((3, 4), dtype=np.float32)
        
        return calib_info

    def _load_annotations(self, label_path):
        """Load label file in KITTI format and return annotations"""
        gt_bboxes = []
        gt_labels = []
        gt_names = []
        
        if not os.path.exists(label_path):
            return {
                'gt_bboxes': np.zeros((0, 7), dtype=np.float32),
                'gt_labels': np.zeros(0, dtype=np.int64),
                'gt_names': []
            }
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_type = parts[0]
                
                # Skip if not a known class
                if obj_type not in self.CLASSES:
                    continue
                
                # Parse KITTI format: type truncated occluded alpha bbox dimensions location rotation_y score
                truncated = int(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])
                
                # bbox in image (unused for 3D detection)
                bbox_left = float(parts[4])
                bbox_top = float(parts[5])
                bbox_right = float(parts[6])
                bbox_bottom = float(parts[7])
                
                # Dimensions: height, width, length
                height = float(parts[8])
                width = float(parts[9])
                length = float(parts[10])
                
                # Location: x, y, z (in camera frame)
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                
                # Rotation around y-axis (yaw)
                rotation_y = float(parts[14])
                
                # Score (optional, for test results)
                score = float(parts[15]) if len(parts) > 15 else 1.0
                
                # Create bounding box in camera frame: [x, y, z, h, w, l, ry]
                bbox_camera = np.array([x, y, z, height, width, length, rotation_y], dtype=np.float32)
                gt_bboxes.append(bbox_camera)
                gt_labels.append(self.CLASSES[obj_type])
                gt_names.append(obj_type)
        
        return {
            'gt_bboxes': np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 7), dtype=np.float32),
            'gt_labels': np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros(0, dtype=np.int64),
            'gt_names': gt_names
        }


if __name__ == '__main__':
    rzdg_data = RZDG(data_root='/home/rax/Documents/team_project/RZDG_real_object',
                     split='train')
    print(f"Dataset size: {len(rzdg_data)}")
    sample = rzdg_data[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Point cloud shape: {sample['pts'].shape}")
    print(f"GT bboxes shape: {sample['gt_bboxes_3d'].shape}")
    print(f"GT labels: {sample['gt_labels']}")
    print(f"GT names: {sample['gt_names']}")
