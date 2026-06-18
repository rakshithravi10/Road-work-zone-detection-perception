import argparse
import cv2
import numpy as np
import os
import torch
import pdb

from pointpillars.utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from pointpillars.model import PointPillars


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def filter_valid_boxes(camera_bboxes, labels, scores, image_shape, max_depth=60, min_depth=1):
    """
    Filter out boxes that would cause long projection lines.
    
    Args:
        camera_bboxes: (N, 7) camera coordinates [x, y, z, w, l, h, ry]
        labels: (N,) class labels
        scores: (N,) confidence scores
        image_shape: (H, W) image dimensions
        max_depth: maximum allowed depth (z) in meters
        min_depth: minimum allowed depth (z) in meters
    
    Returns:
        Filtered camera_bboxes, labels, scores
    """
    if len(camera_bboxes) == 0:
        return camera_bboxes, labels, scores
    
    # Filter by depth (z coordinate in camera frame)
    depth = camera_bboxes[:, 2]
    depth_mask = (depth > min_depth) & (depth < max_depth)
    
    # Filter by reasonable dimensions (avoid huge boxes)
    dims = camera_bboxes[:, 3:6]  # w, l, h
    max_dim = np.max(dims, axis=1)
    dim_mask = max_dim < 10  # No dimension larger than 10m
    
    # Filter by reasonable x, y position (not too far from center)
    x_pos = np.abs(camera_bboxes[:, 0])
    y_pos = np.abs(camera_bboxes[:, 1])
    pos_mask = (x_pos < 50) & (y_pos < 10)  # Reasonable lateral and vertical range
    
    # Combine all masks
    keep_mask = depth_mask & dim_mask & pos_mask
    
    return camera_bboxes[keep_mask], labels[keep_mask], scores[keep_mask] if len(scores) > 0 else scores


def vis_img_3d_filtered(img, image_points, labels, image_shape, rt=True):
    """
    Visualize 3D boxes on image, filtering out boxes with points far outside image.
    
    Args:
        img: input image
        image_points: (N, 8, 2) corners projected to image
        labels: (N,) labels
        image_shape: (H, W)
        rt: whether to return image
    """
    H, W = image_shape
    margin = 500  # Allow some margin outside image
    
    # Color map
    colors = {
        0: (0, 0, 255),    # barrier - red
        1: (0, 255, 0),    # road_beacon - green
        -1: (0, 255, 0),   # GT - green
    }
    
    for i, (corners, label) in enumerate(zip(image_points, labels)):
        # Check if any corner is way outside the image (causes long lines)
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        
        # Skip if any point is too far outside image bounds
        if np.any(x_coords < -margin) or np.any(x_coords > W + margin):
            continue
        if np.any(y_coords < -margin) or np.any(y_coords > H + margin):
            continue
        
        # Skip if the box spans too large a range (indicates projection error)
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        if x_range > W * 2 or y_range > H * 2:
            continue
        
        # Get color for this label
        color = colors.get(label, (255, 255, 255))
        
        # Draw the 3D box edges
        # Bottom face: 0-1-2-3
        # Top face: 4-5-6-7
        corners = corners.astype(np.int32)
        
        # Bottom face
        for j in range(4):
            pt1 = tuple(corners[j])
            pt2 = tuple(corners[(j + 1) % 4])
            cv2.line(img, pt1, pt2, color, 2)
        
        # Top face
        for j in range(4):
            pt1 = tuple(corners[j + 4])
            pt2 = tuple(corners[(j + 1) % 4 + 4])
            cv2.line(img, pt1, pt2, color, 2)
        
        # Vertical edges
        for j in range(4):
            pt1 = tuple(corners[j])
            pt2 = tuple(corners[j + 4])
            cv2.line(img, pt1, pt2, color, 2)
    
    if rt:
        return img


import glob

def main():
    CLASSES = {
        'barrier': 0, 
        'road_beacon': 1, 
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # Hardcoded base paths
    base_root = "/home/rax/Documents/team_project/RZDG_real_object/testing"
    velo_dir = os.path.join(base_root, 'velodyne')
    calib_dir = os.path.join(base_root, 'calib')
    gt_dir = os.path.join(base_root, 'label_0')
    img_dir = os.path.join(base_root, 'image_0')
    ckpt_path = '/home/rax/Documents/team_project/PointPillars/best_model.pth'
    results_dir = 'results3'  # Changed output directory
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    no_cuda = False 

    # Load model once
    if not no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(ckpt_path))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    # Get all .bin files
    pc_files = glob.glob(os.path.join(velo_dir, '*.bin'))
    pc_files.sort()
    
    print(f"Found {len(pc_files)} files to process.")

    for pc_path in pc_files:
        file_id = os.path.basename(pc_path).split('.')[0]
        
        calib_path = os.path.join(calib_dir, f'{file_id}.txt')
        gt_path = os.path.join(gt_dir, f'{file_id}.txt')
        img_path = os.path.join(img_dir, f'{file_id}.png')
        
        print(f"Processing {file_id}...")

        if not os.path.exists(pc_path):
            print(f"Skipping {file_id}: PC not found")
            continue
            
        pc = read_points(pc_path)
        pc = point_range_filter(pc)
        pc_torch = torch.from_numpy(pc)
        
        if os.path.exists(calib_path):
            calib_info = read_calib(calib_path)
        else:
            calib_info = None
        
        if os.path.exists(gt_path):
            gt_label = read_label(gt_path)
        else:
            gt_label = None

        if os.path.exists(img_path):
            img = cv2.imread(img_path, 1)
        else:
            img = None

        with torch.no_grad():
            if not no_cuda:
                pc_torch = pc_torch.cuda()
            
            result_filter = model(batched_pts=[pc_torch], 
                                  mode='test')[0]
        
        if calib_info is not None and img is not None:
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)
            P2 = calib_info['P0'].astype(np.float32)

            image_shape = img.shape[:2]
            result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

        result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        
        # 3D visualization of point cloud and detected boxes
        # vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

        if img is not None and calib_info is not None:
            bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
            
            # Filter noisy predictions using confidence scores and physical constraints
            camera_bboxes, labels_filtered, scores_filtered = filter_noisy_predictions(
                camera_bboxes, labels, scores,
                score_threshold=0.35,  # Adjust this value (0.1-0.7)
                max_depth=60,
                min_depth=1,
                max_dimension=10,
                lateral_range=50,
                vertical_range=10
            )
            
            if len(camera_bboxes) > 0:
                bboxes_corners = bbox3d2corners_camera(camera_bboxes)
                image_points = points_camera2image(bboxes_corners, P2)
                img = vis_img_3d_filtered(img, image_points, labels_filtered, img.shape[:2], rt=True)

        if calib_info is not None and gt_label is not None and img is not None:
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)

            dimensions = gt_label['dimensions']
            location = gt_label['location']
            rotation_y = gt_label['rotation_y']
            gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
            sel = gt_labels != -1
            gt_labels = gt_labels[sel]
            bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
            gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
            bboxes_camera = bboxes_camera[sel]
            gt_lidar_bboxes = gt_lidar_bboxes[sel]

            # P2 is needed for projection
            P2 = calib_info['P0'].astype(np.float32)
            
            # Filter GT boxes - same filtering
            gt_dummy_scores = np.ones(len(bboxes_camera))
            gt_vis_labels = np.array([-1] * len(bboxes_camera))
            bboxes_camera_filtered, gt_vis_labels_filtered, _ = filter_valid_boxes(
                bboxes_camera, gt_vis_labels, gt_dummy_scores, img.shape[:2]
            )
            
            if len(bboxes_camera_filtered) > 0:
                bboxes_corners = bbox3d2corners_camera(bboxes_camera_filtered)
                image_points = points_camera2image(bboxes_corners, P2)
                img = vis_img_3d_filtered(img, image_points, gt_vis_labels_filtered, img.shape[:2], rt=True)
        
        if img is not None:
            output_name = os.path.join(results_dir, f'result_{file_id}.png')
            cv2.imwrite(output_name, img)
            print(f"Result saved to {output_name}")
            
        
def filter_noisy_predictions(camera_bboxes, labels, scores, 
                            score_threshold=0.3,
                            class_score_thresholds=None,
                            max_depth=60, min_depth=1,
                            max_dimension=10,
                            lateral_range=50,
                            vertical_range=10):
    """
    Comprehensive filtering for noisy 3D predictions.
    
    Args:
        score_threshold: Global confidence threshold
        class_score_thresholds: Dict of {class_id: threshold} for per-class filtering
        Other params: Physical constraints
    """
    if len(camera_bboxes) == 0:
        return camera_bboxes, labels, scores
    
    # 1. Score thresholding
    score_mask = scores > score_threshold
    
    # 2. Class-specific thresholding (if provided)
    if class_score_thresholds:
        class_mask = np.zeros(len(labels), dtype=bool)
        for class_id, threshold in class_score_thresholds.items():
            class_mask |= (labels == class_id) & (scores >= threshold)
        score_mask = score_mask & class_mask
    
    # 3. Physical constraints
    depth = camera_bboxes[:, 2]
    depth_mask = (depth > min_depth) & (depth < max_depth)
    
    dims = camera_bboxes[:, 3:6]
    max_dim = np.max(dims, axis=1)
    dim_mask = max_dim < max_dimension
    
    x_pos = np.abs(camera_bboxes[:, 0])
    y_pos = np.abs(camera_bboxes[:, 1])
    pos_mask = (x_pos < lateral_range) & (y_pos < vertical_range)
    
    # 4. Combine all filters
    keep_mask = score_mask & depth_mask & dim_mask & pos_mask
    
    return camera_bboxes[keep_mask], labels[keep_mask], scores[keep_mask]


if __name__ == '__main__':
    main()