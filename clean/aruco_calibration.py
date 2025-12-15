#!/usr/bin/env python3
"""
ArUco board detection and camera-to-world calibration functions.

Provides functions for detecting ArUco boards and computing transformation matrices
from camera frame to world frame.
"""

import numpy as np
import cv2
import json
import os
from typing import Optional, Dict

from config import CONFIG


def detect_aruco_board(image: np.ndarray, camera_K: np.ndarray,
                       dist_coeffs: Optional[np.ndarray] = None) -> Dict:
    """
    Detect a 3x4 ArUco board in an image and estimate its pose.

    Detects a 3x4 board of ArUco markers (DICT_4X4_50) with 30mm markers and
    10mm separation. Returns the detected markers' corners, IDs, and the estimated
    board pose using solvePnP.

    Args:
        image: Input image (BGR or grayscale)
        camera_K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (5,) or (8,). If None, assumes no distortion.

    Returns:
        Dictionary with keys:
            - success (bool): Whether detection and pose estimation succeeded
            - rvec (np.ndarray): Rotation vector (3,) if successful, None otherwise
            - tvec (np.ndarray): Translation vector (3,) if successful, None otherwise
            - corners (list): List of detected marker corners if successful, None otherwise
            - ids (np.ndarray): Array of detected marker IDs if successful, None otherwise
    """
    result = {
        "success": False,
        "rvec": None,
        "tvec": None,
        "corners": None,
        "ids": None
    }

    # Ensure camera matrix is float32
    camera_K = np.asarray(camera_K, dtype=np.float32)

    # Handle distortion coefficients
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5,), dtype=np.float32)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize ArUco detector with compatibility for old and new APIs
    aruco = cv2.aruco

    # Get dictionary
    dict_id = CONFIG["aruco"]["dictionary_enum"]
    try:
        # New API (OpenCV 4.7+)
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        # Old API (OpenCV 4.6 and earlier)
        aruco_dict = aruco.Dictionary_get(dict_id)
        detector_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    # Check if markers were detected
    if ids is None or len(ids) == 0:
        return result

    # Build object points and image points from detected markers
    obj_pts = []
    img_pts = []

    for marker_id, corner in zip(ids.flatten(), corners):
        obj_3d = _get_marker_corners_3d(marker_id)
        if obj_3d is None:
            continue

        img_2d = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
        obj_pts.append(obj_3d)
        img_pts.append(img_2d)

    if len(obj_pts) == 0:
        return result

    # Concatenate all points
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)

    # Solve PnP using IPPE first (more stable), then fallback to EPNP
    success = False
    rvec = None
    tvec = None

    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_K, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        pass

    # Fallback to EPNP
    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
        except cv2.error:
            pass

    # Final fallback to default method
    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_K, dist_coeffs
            )
        except cv2.error:
            return result

    if not success or rvec is None or tvec is None:
        return result

    # Success: populate result dictionary
    result["success"] = True
    result["rvec"] = rvec.reshape(3)
    result["tvec"] = tvec.reshape(3)
    result["corners"] = corners
    result["ids"] = ids

    return result


def calibrate_camera_to_world(image: np.ndarray, camera_K: np.ndarray,
                              dist_coeffs: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Calibrate camera to world transformation from ArUco board detection.

    Detects an ArUco board in the image and computes a 4x4 homogeneous transformation
    matrix that transforms points from world coordinates (board frame) to camera frame.

    Args:
        image: Input image (BGR or grayscale)
        camera_K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (5,) or (8,). If None, assumes no distortion.

    Returns:
        4x4 transformation matrix T_world_camera, or None if detection fails.
        The matrix transforms points as: p_camera = T_world_camera @ p_world
    """
    detection_result = detect_aruco_board(image, camera_K, dist_coeffs)

    if not detection_result["success"]:
        return None

    rvec = detection_result["rvec"]
    tvec = detection_result["tvec"]

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    # Construct 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3] = tvec.astype(np.float64)

    return T


def save_calibration(T_matrix: np.ndarray, filepath: str) -> bool:
    """
    Save a 4x4 transformation matrix to a JSON file.

    Saves the rotation matrix (3x3) and translation vector (3,) extracted from
    the homogeneous transformation matrix.

    Args:
        T_matrix: 4x4 homogeneous transformation matrix
        filepath: Path to save the JSON file

    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Ensure it's 4x4
        if T_matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {T_matrix.shape}")

        # Extract rotation and translation
        R = T_matrix[:3, :3].tolist()
        t = T_matrix[:3, 3].tolist()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to JSON
        calibration_data = {
            "R": R,
            "t": t
        }

        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving calibration to {filepath}: {e}")
        return False


def load_calibration(filepath: str) -> Optional[np.ndarray]:
    """
    Load a 4x4 transformation matrix from a JSON file.

    Loads the rotation matrix (3x3) and translation vector (3,) from a JSON file
    and reconstructs a 4x4 homogeneous transformation matrix.

    Args:
        filepath: Path to the JSON calibration file

    Returns:
        4x4 transformation matrix, or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            calibration_data = json.load(f)

        # Extract R and t
        R = np.asarray(calibration_data["R"], dtype=np.float64)
        t = np.asarray(calibration_data["t"], dtype=np.float64)

        # Validate shapes
        if R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3, 3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"Expected t shape (3,), got {t.shape}")

        # Construct 4x4 matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    except Exception as e:
        print(f"Error loading calibration from {filepath}: {e}")
        return None


def _get_marker_corners_3d(marker_id: int) -> Optional[np.ndarray]:
    """
    Get 3D corner positions for a marker on the board.

    Computes the world-frame 3D coordinates of the four corners of a marker
    on the ArUco board, based on its ID.

    Args:
        marker_id: Marker ID (0 to ARUCO_ROWS*ARUCO_COLS-1)

    Returns:
        4x3 array of 3D corner positions in meters, or None if invalid ID
    """
    aruco_rows = CONFIG["aruco"]["rows"]
    aruco_cols = CONFIG["aruco"]["cols"]
    marker_size = CONFIG["aruco"]["marker_size_m"]
    separation = CONFIG["aruco"]["marker_separation_m"]

    if marker_id < 0 or marker_id >= aruco_rows * aruco_cols:
        return None

    row, col = divmod(marker_id, aruco_cols)

    x0 = col * (marker_size + separation)
    y0 = row * (marker_size + separation)

    return np.array([
        [x0, y0, 0.0],
        [x0 + marker_size, y0, 0.0],
        [x0 + marker_size, y0 + marker_size, 0.0],
        [x0, y0 + marker_size, 0.0]
    ], dtype=np.float32)
