# Road Work Zone Detection Perception

Camera and LiDAR calibration, semantic segmentation, and 3D object detection for road work zone objects (construction barriers, traffic cones), built on a real vehicle sensor rig at Technische Hochschule Ingolstadt (THI).

This was a team project where each member completed individual technical tasks. The work documented here covers my individual contributions across three stages of the pipeline: sensor calibration, semantic segmentation, and 3D object detection.

## Hardware Setup

- 7 cameras (1920x1200 resolution) mounted on a research vehicle
- Ouster LiDAR sensor
- Real-world data captured on streets in and around Ingolstadt, Germany

## Pipeline Overview

### Task 1 — Multi-Camera Calibration and LiDAR Fusion
`task1_calibration/`

- Performed intrinsic calibration of all 7 cameras using the checkerboard method
- Achieved mean reprojection errors as low as 0.12 pixels (cam1), with most cameras under 0.35 pixels
- Computed LiDAR-to-camera extrinsic transformation matrices (4x4 SE3) using manual 3D-to-2D point correspondences
- Verified full sensor fusion by projecting the LiDAR point cloud onto camera images and visualising in RViz

| Camera | Mean Reprojection Error (px) |
|---|---|
| cam1 | 0.12 |
| cam7 | 0.20 |
| cam3 | 0.22 |
| cam4 | 0.25 |
| cam5 | 0.35 |
| cam2 | 0.43 |
| cam6 | 0.56 |

### Task 2 — Semantic Segmentation
`task2_segmentation/`

- Annotated 87 real-world road work zone images captured on the THI vehicle rig
- Two classes: construction barriers and traffic cones/pylons
- Trained and evaluated a DeepLabV3+ model on the annotated dataset
- Generated raw segmentation masks and overlay visualisations

### Task 3 — 3D Object Detection
`task3_detection/`

- Ran 3D object detection using PointPillars via OpenPCDet on LiDAR point cloud data
- Generated KITTI-format labels for detected barriers and cones
- Built an inference and visualisation pipeline projecting 3D bounding boxes onto camera images
- Processed all 87 scenes end to end

## Repository Structure

```
road-work-zone-detection-perception/
├── task1_calibration/
│   ├── camera_intrinsic_all_cams.py      # intrinsic calibration script
│   ├── pnp_extrinsic_calibration.py      # extrinsic calibration script
│   ├── cam1.yaml ... cam7.yaml           # per-camera intrinsic results
│   └── extrinsic_cam1__TEST.json ...     # per-camera extrinsic (LiDAR-to-camera) results
├── task2_segmentation/
│   ├── main.py                           # DeepLabV3+ training entry point
│   ├── rzdg_dataset.py                   # custom dataset loader for the THI road work zone data
│   ├── training.log                      # real training run log
│   └── sample_results/                   # example raw images, masks, and overlays
├── task3_detection/
│   ├── train_rzdg.py                     # PointPillars training script
│   ├── test.py                           # inference script
│   ├── rzdg.py                           # dataset handling for detection
│   ├── voxel_module.py                   # voxelization module
│   └── sample_results/                   # example KITTI-format labels and 3D box visualizations
└── assets/                               # calibration and fusion verification screenshots
```

## Sample Results

`assets/` contains LiDAR point cloud and camera fusion verification screenshots from RViz.

`task2_segmentation/sample_results/` contains raw camera images, generated segmentation masks, and DeepLabV3+ overlay outputs.

`task3_detection/sample_results/` contains KITTI-format label files and camera images with projected 3D bounding boxes for detected barriers and cones.

## Tech Stack

Python, OpenCV, PyTorch, DeepLabV3+, PointPillars, OpenPCDet, ROS2, RViz, KITTI format

## Notes

This repository contains the code and results for my individual contributions to a team project. Cloned third-party libraries (OpenPCDet, DeepLabV3Plus-Pytorch base implementation, PointPillars base implementation) are not included here — only the scripts, configs, and results I produced.
