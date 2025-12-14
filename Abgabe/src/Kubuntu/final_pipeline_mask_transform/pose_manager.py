"""
Pose Manager - Headset Pose Streaming and ArUco Calibration
Handles continuous headset pose updates and one-time ArUco-based calibration
"""

import numpy as np
import cv2 as cv
import json
import time
from typing import Optional, Dict, Tuple
from collections import deque
from scipy.spatial.transform import Rotation

from .config import ARUCO_CONFIG, CALIBRATION_FILES


class PoseManager:
    """
    Manages headset poses and calibration
    - Streams headset pose continuously
    - Performs one-time ArUco calibration
    - Maintains pose history for filtering
    """

    def __init__(self):
        """Initialize pose manager"""
        # ArUco detection setup
        self.aruco_dict, self.detector, self.api = self._setup_aruco()

        # Pose history (for filtering/smoothing)
        self.pose_history = deque(maxlen=30)  # Keep last 30 poses

        # Current headset pose
        self.current_headset_pose = None
        self.last_headset_update = None

        # Baseline headset pose (saved at calibration time)
        # Used for drift correction when ArUco is not visible
        self.baseline_headset_pose = None
        self.baseline_timestamp = None

        # Calibration data
        self.calibration_data = {
            "headset_to_world": None,  # T_world_headset
            "realsense_to_world": None,  # T_world_realsense
            "avp_to_realsense": None  # T_realsense_avp (derived)
        }

        # Load existing calibration if available
        self._load_calibration()

    def _setup_aruco(self) -> Tuple:
        """Setup ArUco detection"""
        if not hasattr(cv, "aruco"):
            print("[ERROR] cv2.aruco not available")
            return None, None, None

        aruco = cv.aruco

        # Get dictionary
        dict_name = ARUCO_CONFIG["dictionary"]
        try:
            aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        except:
            aruco_dict = aruco.Dictionary_get(getattr(aruco, dict_name))

        # Get detector parameters
        try:
            params = aruco.DetectorParameters()
        except:
            params = aruco.DetectorParameters_create()

        # Create detector
        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(aruco_dict, params)
            api = "new"
        else:
            detector = params
            api = "old"

        return aruco_dict, detector, api

    def update_headset_pose(self, pose: Dict):
        """
        Update current headset pose from streaming data

        Args:
            pose: Dictionary with 'position' [x, y, z] and 'rotation' [rx, ry, rz] or quaternion
        """
        self.current_headset_pose = pose
        self.last_headset_update = time.time()

        # Add to history
        self.pose_history.append({
            "pose": pose.copy(),
            "timestamp": self.last_headset_update
        })

    def get_current_headset_pose(self) -> Optional[Dict]:
        """Get latest headset pose"""
        return self.current_headset_pose

    def get_pose_history(self, duration_sec: float = 1.0) -> list:
        """
        Get pose history within time window

        Args:
            duration_sec: Time window in seconds

        Returns:
            List of poses within time window
        """
        if not self.pose_history:
            return []

        current_time = time.time()
        cutoff_time = current_time - duration_sec

        return [p for p in self.pose_history if p["timestamp"] >= cutoff_time]

    def detect_aruco_in_image(self, image: np.ndarray, camera_matrix: np.ndarray,
                               dist_coeffs: np.ndarray) -> Optional[Dict]:
        """
        Detect ArUco markers and estimate pose

        Args:
            image: Input image (grayscale or color)
            camera_matrix: Camera intrinsics K
            dist_coeffs: Distortion coefficients

        Returns:
            Dictionary with detected markers and poses
        """
        if self.aruco_dict is None:
            return None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers
        if self.api == "new":
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector
            )

        if ids is None or len(ids) == 0:
            return None

        # Estimate poses for all detected markers
        marker_poses = []
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i]

            # Get marker size
            if marker_id == ARUCO_CONFIG["headset_marker_id"]:
                marker_size = ARUCO_CONFIG["headset_marker_size_m"]
            else:
                marker_size = ARUCO_CONFIG["marker_size_m"]

            # Estimate pose
            obj_points = np.array([
                [-marker_size/2, marker_size/2, 0],
                [marker_size/2, marker_size/2, 0],
                [marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv.solvePnP(
                obj_points,
                marker_corners.reshape(-1, 2),
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_IPPE
            )

            if success:
                marker_poses.append({
                    "id": int(marker_id),
                    "rvec": rvec.flatten(),
                    "tvec": tvec.flatten(),
                    "corners": marker_corners
                })

        return {
            "markers": marker_poses,
            "num_detected": len(marker_poses),
            "timestamp": time.time()
        }

    def detect_board_pose(self, image: np.ndarray, camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect ArUco board and estimate its pose

        Args:
            image: Input image
            camera_matrix: Camera intrinsics
            dist_coeffs: Distortion coefficients

        Returns:
            (rvec, tvec) of board pose, or None
        """
        detection = self.detect_aruco_in_image(image, camera_matrix, dist_coeffs)
        if not detection or detection["num_detected"] == 0:
            return None

        # Collect all 3D-2D correspondences from board markers
        obj_pts = []
        img_pts = []

        rows = ARUCO_CONFIG["board_rows"]
        cols = ARUCO_CONFIG["board_cols"]
        marker_size = ARUCO_CONFIG["marker_size_m"]
        separation = ARUCO_CONFIG["separation_m"]

        for marker in detection["markers"]:
            marker_id = marker["id"]

            # Skip headset marker
            if marker_id == ARUCO_CONFIG["headset_marker_id"]:
                continue

            # Check if marker is part of board
            if marker_id >= rows * cols:
                continue

            # Get board position of this marker
            row, col = divmod(marker_id, cols)
            x0 = col * (marker_size + separation)
            y0 = row * (marker_size + separation)

            # 3D corners of this marker on the board
            marker_obj_pts = np.array([
                [x0, y0, 0],
                [x0 + marker_size, y0, 0],
                [x0 + marker_size, y0 + marker_size, 0],
                [x0, y0 + marker_size, 0]
            ], dtype=np.float32)

            # 2D corners in image
            marker_img_pts = marker["corners"].reshape(-1, 2).astype(np.float32)

            obj_pts.append(marker_obj_pts)
            img_pts.append(marker_img_pts)

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
            success, rvec, tvec = cv.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs
            )

        if success:
            return rvec.flatten(), tvec.flatten()

        return None

    def calibrate_with_aruco(self, realsense_camera, headset_image: np.ndarray,
                            headset_camera_matrix: np.ndarray,
                            headset_dist_coeffs: np.ndarray) -> bool:
        """
        Perform one-time ArUco-based calibration

        Args:
            realsense_camera: RealSenseDepth instance
            headset_image: Image from headset with ArUco marker visible
            headset_camera_matrix: Headset camera intrinsics
            headset_dist_coeffs: Headset distortion coefficients

        Returns:
            True if calibration successful
        """
        print("[Calibration] Starting ArUco calibration...")

        # Step 1: Detect headset marker in headset view
        print("[Calibration] Detecting headset marker...")
        headset_detection = self.detect_aruco_in_image(
            headset_image, headset_camera_matrix, headset_dist_coeffs
        )

        if not headset_detection or headset_detection["num_detected"] == 0:
            print("[ERROR] No markers detected in headset view")
            return False

        # Find headset marker
        headset_marker = None
        for marker in headset_detection["markers"]:
            if marker["id"] == ARUCO_CONFIG["headset_marker_id"]:
                headset_marker = marker
                break

        if headset_marker is None:
            print("[ERROR] Headset marker not found")
            return False

        # Get T_world_headset (headset to world transform)
        T_world_headset = self._pose_to_matrix(headset_marker["rvec"], headset_marker["tvec"])

        # Step 2: Detect board in RealSense view
        print("[Calibration] Capturing RealSense frame...")
        rs_data = realsense_camera.capture_frame()
        if not rs_data:
            print("[ERROR] Failed to capture RealSense frame")
            return False

        print("[Calibration] Detecting board in RealSense view...")
        board_pose_rs = self.detect_board_pose(
            rs_data["rgb"],
            rs_data["intrinsics"]["K"],
            rs_data["intrinsics"]["dist"]
        )

        if board_pose_rs is None:
            print("[ERROR] Board not detected in RealSense view")
            return False

        # Get T_realsense_world (world to RealSense transform)
        T_realsense_world = self._pose_to_matrix(board_pose_rs[0], board_pose_rs[1])

        # Step 3: Compute T_realsense_avp
        # T_realsense_avp = T_realsense_world * T_world_headset
        T_realsense_avp = T_realsense_world @ T_world_headset

        # Save calibration
        self.calibration_data["headset_to_world"] = T_world_headset
        self.calibration_data["realsense_to_world"] = T_realsense_world
        self.calibration_data["avp_to_realsense"] = T_realsense_avp

        # Save baseline headset pose for drift correction
        if self.current_headset_pose is not None:
            self.baseline_headset_pose = self.current_headset_pose.copy()
            self.baseline_timestamp = time.time()
            print("[Calibration] Saved baseline headset pose for drift correction")
        else:
            print("[WARNING] No current headset pose available - drift correction disabled")

        self._save_calibration()

        print("[Calibration] Calibration successful!")
        print(f"[Calibration] T_world_headset:\n{T_world_headset}")
        print(f"[Calibration] T_realsense_world:\n{T_realsense_world}")
        print(f"[Calibration] T_realsense_avp:\n{T_realsense_avp}")

        return True

    def _pose_to_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert rvec, tvec to 4x4 transformation matrix"""
        R, _ = cv.Rodrigues(rvec.reshape(3, 1))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def _matrix_to_pose(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 4x4 transformation matrix to rvec, tvec"""
        R = T[:3, :3]
        tvec = T[:3, 3]
        rvec, _ = cv.Rodrigues(R)
        return rvec.flatten(), tvec

    def _pose_dict_to_matrix(self, pose: Dict) -> np.ndarray:
        """
        Convert pose dictionary (from streamed headset data) to 4x4 transformation matrix

        Args:
            pose: Dictionary with 'position' [x, y, z] and 'rotation' [rx, ry, rz] or 'quaternion'

        Returns:
            4x4 transformation matrix
        """
        position = np.array(pose["position"])

        # Handle rotation representation
        if "quaternion" in pose:
            # Quaternion format: need to determine if it's [qw, qx, qy, qz] or [qx, qy, qz, qw]
            # Assuming [qw, qx, qy, qz] format (scalar first)
            quat = np.array(pose["quaternion"])
            # Convert quaternion to rotation matrix using scipy
            r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy expects [qx, qy, qz, qw]
            R = r.as_matrix()
        else:
            # Rotation vector (Rodrigues)
            rvec = np.array(pose["rotation"])
            R, _ = cv.Rodrigues(rvec.reshape(3, 1))

        # Construct transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def _save_calibration(self):
        """Save calibration data to files"""
        for key, filepath in CALIBRATION_FILES.items():
            if self.calibration_data[key] is not None:
                data = {
                    "matrix": self.calibration_data[key].tolist(),
                    "timestamp": time.time()
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"[Calibration] Saved {key} to {filepath}")

    def _load_calibration(self):
        """Load calibration data from files"""
        for key, filepath in CALIBRATION_FILES.items():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.calibration_data[key] = np.array(data["matrix"], dtype=np.float64)
                print(f"[Calibration] Loaded {key} from {filepath}")
            except Exception as e:
                print(f"[INFO] No calibration found for {key}: {e}")

    def is_calibrated(self) -> bool:
        """Check if calibration is available"""
        return self.calibration_data["avp_to_realsense"] is not None

    def get_transform_avp_to_realsense(self) -> Optional[np.ndarray]:
        """
        Get AVP to RealSense transformation matrix (base calibration, no drift correction)

        For drift-corrected transform, use get_corrected_transform_avp_to_realsense()
        """
        return self.calibration_data["avp_to_realsense"]

    def get_corrected_transform_avp_to_realsense(self) -> Optional[np.ndarray]:
        """
        Get AVP to RealSense transformation with drift correction

        When ArUco markers are not visible, this method corrects the base calibration
        transform using the deviation between the current streamed head pose and the
        baseline head pose captured at calibration time.

        This compensates for drift in the AVP's pose tracking, but is subject to
        accumulated drift over time if the head pose data drifts from the true position.

        Returns:
            Drift-corrected transformation matrix, or base transform if correction unavailable
        """
        base_transform = self.calibration_data["avp_to_realsense"]
        if base_transform is None:
            return None

        # If no baseline or current head pose, return base transform without correction
        if self.baseline_headset_pose is None or self.current_headset_pose is None:
            return base_transform

        try:
            # Convert poses to transformation matrices
            H_baseline = self._pose_dict_to_matrix(self.baseline_headset_pose)
            H_current = self._pose_dict_to_matrix(self.current_headset_pose)

            # Compute delta: how much has the headset moved since calibration?
            # Delta represents the relative motion in the AVP's local coordinate frame
            Delta = H_current @ np.linalg.inv(H_baseline)

            # Apply correction to the base transform
            # The AVP camera has moved by Delta in its local frame, so we compensate
            # by applying the inverse of Delta to maintain the correct RS-AVP relationship
            corrected_transform = base_transform @ np.linalg.inv(Delta)

            return corrected_transform

        except Exception as e:
            print(f"[WARNING] Failed to apply drift correction: {e}")
            return base_transform


if __name__ == "__main__":
    print("Testing Pose Manager...")
    manager = PoseManager()

    # Test pose update
    test_pose = {
        "position": [0.1, 0.2, 0.3],
        "rotation": [0.0, 0.0, 0.0]
    }
    manager.update_headset_pose(test_pose)
    print(f"Current pose: {manager.get_current_headset_pose()}")

    print(f"Calibrated: {manager.is_calibrated()}")
    print("Test complete!")
