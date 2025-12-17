#!/usr/bin/env python3
"""
ArUco marker detection and board pose estimation.
"""

import numpy as np
import cv2 as cv
from typing import Optional, Tuple, Dict
import config


class ArucoDetector:
    """Detects ArUco markers and estimates board pose."""

    def __init__(self):
        """Initialize ArUco detector."""
        self.aruco_dict = None
        self.detector_params = None
        self.detector = None
        self.api_version = None

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize ArUco detection objects with version compatibility."""
        if not hasattr(cv, "aruco"):
            raise RuntimeError("OpenCV ArUco module not available")

        aruco = cv.aruco

        # Get dictionary
        dict_id = getattr(aruco, config.ARUCO_DICT)
        try:
            self.aruco_dict = aruco.getPredefinedDictionary(dict_id)
        except AttributeError:
            self.aruco_dict = aruco.Dictionary_get(dict_id)

        # Get detector parameters
        try:
            self.detector_params = aruco.DetectorParameters()
        except AttributeError:
            self.detector_params = aruco.DetectorParameters_create()

        # Create detector (new API) or use params directly (old API)
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self.api_version = "new"
        else:
            self.detector = self.detector_params
            self.api_version = "old"

        print(f"[ArUco] Initialized with {self.api_version} API")

    def detect_markers(self, image: np.ndarray) -> Tuple[Optional[list], Optional[np.ndarray]]:
        """
        Detect ArUco markers in image.

        Args:
            image: BGR image

        Returns:
            Tuple of (corners, ids) or (None, None) if no markers detected
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if self.api_version == "new":
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector
            )

        if ids is None or len(ids) == 0:
            return None, None

        return corners, ids

    def estimate_board_pose(
        self,
        corners: list,
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate board pose from detected markers.

        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            Tuple of (rvec, tvec) or None if pose estimation fails
        """
        if ids is None or len(ids) == 0:
            return None

        # Build object points and image points from all detected markers
        obj_pts = []
        img_pts = []

        for marker_id, corner in zip(ids.flatten(), corners):
            obj_3d = self._get_marker_corners_3d(marker_id)
            if obj_3d is None:
                continue

            img_2d = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
            obj_pts.append(obj_3d)
            img_pts.append(img_2d)

        if len(obj_pts) == 0:
            return None

        # Concatenate all points
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        # Solve PnP
        success, rvec, tvec = cv.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv.SOLVEPNP_IPPE
        )

        if not success:
            # Fallback to default method
            success, rvec, tvec = cv.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs
            )

        if not success:
            return None

        return rvec.reshape(3), tvec.reshape(3)

    def _get_marker_corners_3d(self, marker_id: int) -> Optional[np.ndarray]:
        """
        Get 3D corner positions for a marker on the board.

        Args:
            marker_id: Marker ID

        Returns:
            4x3 array of 3D corner positions or None if invalid ID
        """
        if marker_id < 0 or marker_id >= config.ARUCO_ROWS * config.ARUCO_COLS:
            return None

        row, col = divmod(marker_id, config.ARUCO_COLS)

        x0 = col * (config.MARKER_SIZE_M + config.SEPARATION_M)
        y0 = row * (config.MARKER_SIZE_M + config.SEPARATION_M)

        return np.array([
            [x0,                      y0,                      0.0],
            [x0 + config.MARKER_SIZE_M, y0,                      0.0],
            [x0 + config.MARKER_SIZE_M, y0 + config.MARKER_SIZE_M, 0.0],
            [x0,                      y0 + config.MARKER_SIZE_M, 0.0]
        ], dtype=np.float32)

    @staticmethod
    def create_default_camera_matrix(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create default camera intrinsics for given image size.

        Args:
            width: Image width
            height: Image height

        Returns:
            Tuple of (camera_matrix, dist_coeffs)
        """
        focal_length = 0.8 * max(width, height)
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist = np.zeros((5, 1), dtype=np.float32)

        return K, dist

    @staticmethod
    def pose_to_transformation_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert pose (rvec, tvec) to 4x4 transformation matrix.

        Args:
            rvec: Rotation vector (3,)
            tvec: Translation vector (3,)

        Returns:
            4x4 transformation matrix
        """
        R, _ = cv.Rodrigues(rvec.reshape(3, 1))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T
