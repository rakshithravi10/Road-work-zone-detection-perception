import cv2
import numpy as np
import glob
import os

# =====================================================
# CONFIGURATION
# =====================================================

CHESSBOARD_SIZE = (8, 5)     # inner corners
SQUARE_SIZE = 0.025          # meters (scale not critical)

CAMERAS = [
    "cam1", "cam2", "cam3",
    "cam4", "cam5", "cam6", "cam7"
]

BASE_IMAGE_PATH = "../intrinsic"
OUTPUT_BASE = "outputs"

# =====================================================
# CREATE OUTPUT ROOT
# =====================================================

os.makedirs(OUTPUT_BASE, exist_ok=True)

# =====================================================
# INTRINSIC CALIBRATION LOOP
# =====================================================

for cam in CAMERAS:
    print(f"\n========== Calibrating {cam} ==========")

    image_path = os.path.join(
        BASE_IMAGE_PATH, cam, "calibrationdata", "*.png"
    )
    images = sorted(glob.glob(image_path))[:30]

    if len(images) < 10:
        print(f"Not enough images for {cam}. Skipping.")
        continue

    cam_output = os.path.join(OUTPUT_BASE, cam)
    os.makedirs(cam_output, exist_ok=True)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS +
                 cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners_refined)

            cv2.drawChessboardCorners(
                img,
                CHESSBOARD_SIZE,
                corners_refined,
                ret
            )

            cv2.imwrite(
                os.path.join(cam_output, f"corners_{idx:02d}.png"),
                img
            )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            K,
            dist
        )
        error = cv2.norm(
            imgpoints[i],
            imgpoints_proj,
            cv2.NORM_L2
        ) / len(imgpoints_proj)
        total_error += error

    mean_error = total_error / len(objpoints)

    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist)
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")

    np.save(os.path.join(cam_output, "K.npy"), K)
    np.save(os.path.join(cam_output, "dist.npy"), dist)

print("\nAll cameras intrinsic calibration completed.")
